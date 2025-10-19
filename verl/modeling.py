import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from verl.utils.torch_dtypes import PrecisionType
from typing import Union
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoConfig
from verl.utils import get_tokenizer, get_processor
from verl.utils.model_utils import set_trainable
from typing import Optional, Dict
from segment_anything import sam_model_registry
from transformers.utils import ModelOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)


@dataclass
class Qwen2_5_VLWithSegOutput(Qwen2_5_VLCausalLMOutputWithPast):
    seg_outputs: Optional[Dict[str, torch.Tensor]] = None


class VLRefSegCore(nn.Module):
    """
    Args:
        llm:
            Causal LM / VLM backbone. Must support `forward(**llm_inputs)`
            and return a `transformers.ModelOutput` with `hidden_states`
            when `output_hidden_states=True`.
        heatmap_head:
            Module that maps (image_embed, ref_vec) -> {'ref_logits_256': [B,1,256,256], ...}.
            Required for segmentation.
        prompt_encoder:
            SAM-style prompt encoder. Must implement `__call__(masks)` returning
            (sparse_embed, dense_embed) and `get_dense_pe()`. Required for segmentation.
        mask_decoder:
            SAM-style mask decoder taking image_embeddings, image_pe, sparse/dense prompts.
            Required for segmentation.
        special_token_id:
            Vocabulary id for the `<REF_POS>` token used to extract the reference
            embedding from LLM hidden states. **Must be provided**.

    Behavior:
        - Registers `self.ref_pos_id` as a non-persistent buffer (not saved in checkpoints).
        - You can instantiate with only `llm` for text-only usage; segmentation path
          requires `heatmap_head`, `prompt_encoder`, and `mask_decoder`.
        - Visual `image_embed` is NOT created here; it must be provided at `forward(...)`.
    """
    def __init__(
        self,
        llm: nn.Module = None,
        heatmap_head: nn.Module = None,
        prompt_encoder: nn.Module = None,
        mask_decoder: nn.Module = None,
        special_token_id: int = None,
    ):
        super().__init__()
        self.llm = llm
        self.heatmap_head = heatmap_head
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        # SAFETY: special_token_id must be provided
        if special_token_id is None:
            raise ValueError("special_token_id must not be None. Pass tokenizer.convert_tokens_to_ids('<REF_POS>')")
        # Keep on-module device; not saved in checkpoint (persistent=False).
        self.register_buffer("ref_pos_id", torch.tensor(special_token_id, dtype=torch.long), persistent=False)

    def forward(self, forward_seg=True, image_embed=None, unpadded_hw=None, **llm_inputs,) -> ModelOutput:
        """
        Forward pass of VLRefSegCore.

        Args:
            forward_seg (bool, default=True):
                Whether to run the segmentation branch (heatmap head + SAM).
            image_embed (Tensor or None):
                Precomputed vision/SAM image embeddings of shape [B, C, H', W'].
                Required when `forward_seg=True`.
            unpadded_hw (Tensor or None):
                Unpadded height and width of to decide valid mask positions, of shape [B, 2]
            **llm_inputs:
                Arguments forwarded to `self.llm.forward(...)`. Must include:
                  - input_ids: LongTensor [B, T]
                  - attention_mask: Bool/LongTensor [B, T]
                  - position_ids: LongTensor [B, T] or [B, 3, T] (for mRoPE)
                Optional: pixel_values / other MM kwargs expected by the LLM.
                Notes: If not provided, this function sets `use_cache=False` and
                       `output_hidden_states=forward_seg`.

        Returns:
            If `forward_seg=False`:
                transformers.ModelOutput from the LLM, unchanged.

            If `forward_seg=True`:
                transformers.ModelOutput from the LLM with an extra field
                `seg_outputs` (a dict) containing:
                  - ref_vec: Tensor [B, D], embedding at the **last** `<REF_POS>`
                             per sample (zeros if `<REF_POS>` absent).
                  - ref_logits_256: Tensor [B, 1, 256, 256], heatmap head logits.
                  - mask_logits: Tensor [B, 1, Hm, Wm], decoder logits.
                  - mask_sigmoid: Tensor [B, 1, Hm, Wm], sigmoid(mask_logits).

        Notes:
            - Requires `special_token_id` at init; `<REF_POS>` id is `self.ref_pos_id`.
            - LLM outputs (logits, past, etc.) are not altered; only `seg_outputs`
              is added when segmentation is executed.
        """

        # Use no cache so hidden states align with full sequence (esp. training).
        llm_inputs.setdefault("use_cache", False)

        # Call the LLM. Note: do NOT forward image_embed unless model expects it in kwargs.
        # If your LLM expects visual inputs in llm_inputs (e.g., pixel_values), include them there before this call.
        llm_outputs = self.llm.forward(output_hidden_states=forward_seg, **llm_inputs,)

        if not forward_seg:
            return llm_outputs

        if "input_ids" not in llm_inputs:
            raise KeyError("llm_inputs must contain 'input_ids' to locate <REF_POS>.")
        input_ids = llm_inputs["input_ids"]  # (B, T)
        assert image_embed is not None, "Must input image embed if forward seg"

        # Validate hidden states exist.
        if not hasattr(llm_outputs, "hidden_states") or llm_outputs.hidden_states is None:
            raise RuntimeError("LLM did not return hidden_states. Ensure output_hidden_states=True.")
        # Take the last layer hidden states: (B, T, D)
        last_hidden_state = llm_outputs.hidden_states[-1]

        seg_outputs = self._forward_seg(
            image_embed=image_embed,
            llm_input_ids=input_ids,
            last_hidden_state=last_hidden_state,
            unpadded_hw=unpadded_hw,
        )
        llm_output_dict = dict(llm_outputs)

        return Qwen2_5_VLWithSegOutput(**llm_output_dict, seg_outputs=seg_outputs)

    def _forward_seg(self, image_embed, llm_input_ids, last_hidden_state, unpadded_hw):
        # We need llm_input_ids to locate <REF_POS> positions.
        B, T = llm_input_ids.shape
        device = llm_input_ids.device

        # Boolean mask of where <REF_POS> appears per token.
        sp_mask = (llm_input_ids == self.ref_pos_id.to(device))  # (B, T) bool

        # Strategy: select the LAST occurrence of <REF_POS> per sample.
        # Build a time index grid: [0, 1, ..., T-1] repeated per batch.
        time_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)

        # Replace non-matching positions with 0; argmax then gives the last matching index
        # because later positions have larger time indices.
        masked_idx = torch.where(sp_mask, time_idx, torch.zeros_like(time_idx))  # (B, T)
        last_pos = masked_idx.argmax(dim=-1)  # (B,)

        # has_ref tells which samples actually contain <REF_POS>.
        has_ref = sp_mask.any(dim=-1)  # (B,)

        # Gather the hidden state at the selected index for each batch item: (B, D), D = last_hidden_state.size(-1)
        # For samples without <REF_POS>, replace with zeros to avoid picking index 0 accidentally.
        last_hidden_state = last_hidden_state.to(device=device)
        ref_vec = last_hidden_state[torch.arange(B, device=device), last_pos]  # (B, D)
        ref_vec = torch.where(
            has_ref.unsqueeze(-1),
            ref_vec,
            torch.zeros_like(ref_vec),
        )

        pe_dtype = next(self.prompt_encoder.parameters()).dtype
        pe_device = next(self.prompt_encoder.parameters()).device
        image_embed = image_embed.to(pe_device, dtype=pe_dtype)
        ref_vec = ref_vec.to(pe_device, dtype=pe_dtype)

        # forward by HeatmapHead
        seg_outputs = self.heatmap_head(
            image_embed=image_embed,
            ref_vec=ref_vec,
        )
        seg_outputs["ref_vec"] = ref_vec

        ref_logits_256 = seg_outputs["ref_logits_256"].to(pe_device, dtype=pe_dtype)

        # pad ref_logits_256 to -50.0 by unpadded_hw
        B, _, H, W = ref_logits_256.shape
        up_hw = unpadded_hw.to(pe_device)
        h256 = ((up_hw[:, 0].clamp(min=0) + 3) // 4).clamp(max=H)  # (B,)
        w256 = ((up_hw[:, 1].clamp(min=0) + 3) // 4).clamp(max=W)  # (B,)
        unpad_mask = (torch.arange(H, device=pe_device)[None, None, :, None] < h256[:, None, None, None]) & \
               (torch.arange(W, device=pe_device)[None, None, None, :] < w256[:, None, None, None])
        ref_logits_256 = ref_logits_256.masked_fill(~unpad_mask, ref_logits_256.new_tensor(-50.0))

        # forward by sam
        sparse_embed, dense_embed = self.prompt_encoder(
            points=None, boxes=None, masks=ref_logits_256
        )  # [B, 1, 256, 256] -> [B, 256, 64, 64]
        decoder_dtype = next(self.mask_decoder.parameters()).dtype
        image_pe = self.prompt_encoder.get_dense_pe().to(dtype=decoder_dtype)
        image_embed = image_embed.to(dtype=decoder_dtype)
        sparse_embed = sparse_embed.to(dtype=decoder_dtype)
        dense_embed = dense_embed.to(dtype=decoder_dtype)

        low_res_masks_ls = []
        for i in range(B):
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embed[i:i+1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embed[i:i+1],
                dense_prompt_embeddings=dense_embed[i:i+1],
                multimask_output=False,
            )
            low_res_masks_ls.append(low_res_masks)
        low_res_masks = torch.cat(low_res_masks_ls, dim=0)

        seg_outputs["mask_logits"] = low_res_masks
        seg_outputs["mask_sigmoid"] = torch.sigmoid(low_res_masks)

        return seg_outputs


class MHAttnHeatmapHead(nn.Module):
    """
    Inputs:
      image_embed: (B, 256, 64, 64)  — precomputed SAM image embedding
      ref_vec:     (B, D_llm)        — hidden state of the <REF_POS> token from Qwen
    Output:
      dict with:
        - "ref_logits_256": (B, 1, 256, 256)  — low-res logits for BCEWithLogits
    """
    def __init__(self,
                 llm_dim: int,
                 sam_dim: int = 256,
                 H: int = 64,
                 W: int = 64,
                 out_size: int = 256,
                 num_heads: int = 4,
                 refine: bool = True
                 ):
        super().__init__()
        assert sam_dim % num_heads == 0, "sam_dim must be divisible by num_heads"
        self.sam_dim = sam_dim
        self.H, self.W = H, W
        self.out_size = out_size
        self.num_heads = num_heads
        d_head = sam_dim // num_heads
        self.d_head = d_head

        # (1) Project <REF_POS> (query) into H heads
        self.q_proj = nn.Linear(llm_dim, num_heads * d_head, bias=False)

        # (2) Project SAM features (keys) with a 1x1 conv into H heads
        self.k_proj = nn.Conv2d(sam_dim, num_heads * d_head, kernel_size=1, bias=False)

        # (3) Tiny head-wise fusion/refiner at 64x64 -> 1 channel
        self.refine = refine
        if refine:
            self.fuser = nn.Sequential(
                nn.Conv2d(num_heads, num_heads, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(num_heads, 1, 3, padding=1),
            )
            # Make initial masks slightly empty (negative bias)
            nn.init.constant_(self.fuser[-1].bias, -1.8)
        else:
            self.fuser = nn.Conv2d(num_heads, 1, kernel_size=1)
            nn.init.constant_(self.fuser.bias, -1.8)

    def forward(self, image_embed: torch.Tensor, ref_vec: torch.Tensor):
        """
        image_embed: (B, 256, 64, 64)
        ref_vec:     (B, D_llm)
        """
        B, C, H, W = image_embed.shape
        assert C == self.sam_dim and H == self.H and W == self.W, \
            f"Expected image_embed {(B,self.sam_dim,self.H,self.W)}, got {(B,C,H,W)}"
        # dtype/device alignment
        ref_vec = ref_vec.to(dtype=image_embed.dtype, device=image_embed.device)
        assert ref_vec.dim() == 2 and ref_vec.size(0) == B, "ref_vec must be (B, D_llm)"

        # --- Multi-head attention *logit* map at 64x64 (no softmax!) ---
        # Q: (B, Hh, d_head, 1, 1)
        Q = self.q_proj(ref_vec).view(B, self.num_heads, self.d_head, 1, 1)
        # K: (B, Hh, d_head, 64, 64)
        K = self.k_proj(image_embed).view(B, self.num_heads, self.d_head, H, W)

        # Head-wise dot products -> logits per head: (B, Hh, 64, 64)
        # Scaled by sqrt(d_head) for stability
        logits_64_heads = (Q * K).sum(dim=2) / math.sqrt(self.d_head)

        # Head fusion/refine to a single logit map at 64x64: (B, 1, 64, 64)
        logits_64 = self.fuser(logits_64_heads)  # conv treats 'heads' as channels

        # Upsample to 256x256 (logits)
        logits_256 = F.interpolate(
            logits_64, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
        )

        return {
            "ref_logits_256": logits_256
        }


class RefPosHeatmapHead(nn.Module):
    """
    Inputs:
      sam_embed: (B, 256, 64, 64)  — precomputed SAM image embedding
      ref_vec:   (B, D_llm)        — hidden state of the <REF_POS> token from Qwen
    Output:
      dict with:
        - "ref_logits_256": (B, 1, 256, 256)  — low-res logits for BCEWithLogits
    """
    def __init__(self, llm_dim, sam_dim=256, H=64, W=64, out_size=256, n_blocks=2):
        super().__init__()
        self.sam_dim = sam_dim
        self.H, self.W = H, W
        self.out_size = out_size

        # (1) Project <REF_POS> vector to sam_dim for FiLM modulation
        self.to_gamma = nn.Sequential(
            nn.Linear(llm_dim, sam_dim),
            nn.SiLU(),
            nn.Linear(sam_dim, sam_dim)
        )
        self.to_beta  = nn.Sequential(
            nn.Linear(llm_dim, sam_dim),
            nn.SiLU(),
            nn.Linear(sam_dim, sam_dim)
        )

        # (2) Lightweight conv stack (can be repeated as residual-style blocks)
        blocks = []
        for _ in range(n_blocks):
            blocks += [
                nn.Conv2d(sam_dim, sam_dim, 3, padding=1),
                nn.GroupNorm(32, sam_dim),
                nn.SiLU(),
            ]
        self.conv = nn.Sequential(*blocks)

        # (3) Output head: produce 64×64 logits and upsample to 256×256
        self.head = nn.Conv2d(sam_dim, 1, kernel_size=1)
        # Optional: bias init to a small negative value to start with “emptier” masks
        nn.init.constant_(self.head.bias, -1.8)

    def forward(self, image_embed, ref_vec):
        """
        image_embed: (B, 256, 64, 64)
        ref_vec:   (B, D_llm)
        """
        B, C, H, W = image_embed.shape
        assert C == self.sam_dim and H == self.H and W == self.W
        ref_vec = ref_vec.to(dtype=image_embed.dtype, device=image_embed.device)
        assert ref_vec.dim() == 2 and ref_vec.size(0) == B

        # FiLM: y = (1 + gamma) * x + beta
        gamma = self.to_gamma(ref_vec).view(B, C, 1, 1)
        beta  = self.to_beta(ref_vec).view(B, C, 1, 1)
        feat  = (1 + gamma) * image_embed + beta

        feat      = self.conv(feat)  # (B, 256, 64, 64)
        logits_64 = self.head(feat)  # (B, 1, 64, 64)
        logits_256 = F.interpolate(
            logits_64, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
        )

        return {"ref_logits_256": logits_256}


def build_llm(
        model_path: str,
        supp_bf16: bool = True,
        device_map: str = "auto",
        dtype: str = "bf16",
):
    if supp_bf16:
        attn_impl = "flash_attention_2"
    else:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
        attn_impl = "sdpa"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=PrecisionType.to_dtype(dtype, supp_bf16=supp_bf16),
        attn_implementation=attn_impl,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    tokenizer = get_tokenizer(model_path)
    processor = get_processor(model_path)

    return model, tokenizer, processor


def build_VLRefSegCore(
        llm: nn.Module,
        tokenizer,
        token_embed_dim=None,
        special_token: str = "<REF_POS>",
        init_sam_ckpt: Optional[str] = None,
        is_trainable=True,
):
    if token_embed_dim is None:
        assert llm.get_input_embeddings().embedding_dim == getattr(llm.config, "hidden_size")
        token_embed_dim = llm.get_input_embeddings().embedding_dim

    heatmap_head = MHAttnHeatmapHead(llm_dim=token_embed_dim)

    _sam = sam_model_registry["vit_h"](checkpoint=None)
    prompt_encoder = _sam.prompt_encoder
    mask_decoder = _sam.mask_decoder
    del _sam

    if init_sam_ckpt is not None:
        sd = torch.load(init_sam_ckpt, map_location="cpu")
        pe_sd = {k.split("prompt_encoder.", 1)[1]: v for k, v in sd.items() if k.startswith("prompt_encoder.")}
        md_sd = {k.split("mask_decoder.", 1)[1]: v for k, v in sd.items() if k.startswith("mask_decoder.")}
        prompt_encoder.load_state_dict(pe_sd, strict=False)
        mask_decoder.load_state_dict(md_sd, strict=False)

    ref_pos_id = int(tokenizer.convert_tokens_to_ids(special_token))

    core = VLRefSegCore(
        llm=llm,
        heatmap_head=heatmap_head,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        special_token_id=ref_pos_id,
    )
    if is_trainable:
        core.train()
    else:
        core.eval()
    return core
