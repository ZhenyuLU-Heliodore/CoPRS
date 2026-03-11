# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
import json
import numpy as np
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer import core_algos
from verl.utils.ulysses import compute_iou
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor.base import BasePPOActor
from verl.workers.actor.config import ActorConfig
from verl.utils.rl_dataset import clamp_llava_image_tokens


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
            self,
            config: ActorConfig,
            actor_module: nn.Module,
            actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        self.global_steps = 0

    def _forward_micro_batch(
            self, micro_batch: Dict[str, torch.Tensor], meta_info: Dict = None, forward_seg: bool = False,
    ) -> Tuple:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        temperature = meta_info["temperature"]
        llm_version, image_token_id, replace_token_id = (
            meta_info["llm_version"], meta_info["image_token_id"], meta_info["replace_token_id"])
        input_ids = micro_batch["input_ids"] if "llava" not in llm_version else \
            clamp_llava_image_tokens(micro_batch["input_ids"], image_token_id, replace_token_id)

        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        image_embed = micro_batch["sam_embed"] if forward_seg else None
        unpadded_hw = micro_batch["unpadded_hw"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        vision_inputs = {}
        if "pixel_values" in micro_batch:
            vision_inputs["pixel_values"] = torch.cat(micro_batch["pixel_values"], dim=0)
            vision_inputs["image_grid_thw"] = torch.cat(micro_batch["image_grid_thw"], dim=0) \
                if micro_batch["image_grid_thw"][0] is not None else None

        if self.config.padding_free:
            raise NotImplementedError
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **vision_inputs,
                use_cache=False,
                forward_seg=forward_seg,
                image_embed=image_embed,
                unpadded_hw=unpadded_hw,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1: -1, :]  # (bsz, response_length, vocab_size)
            log_probs = logprobs_from_logits(logits, responses)  # (bsz, response_length)
            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

        if not forward_seg:
            return entropy, log_probs

        return entropy, log_probs, output.seg_outputs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        self.actor_optimizer.step()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto, gen_mask: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys
                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.
                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.
                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.
                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.
                ``sam_embed``: optional, if gen_mask = True, required.

            gen_mask: whether to generate segmentation mask for computing reward function

        Returns:
            if gen_mask is False:
                torch.Tensor: the log_prob tensor
            if gen_mask is True:
                torch.Tensor: the log_prob tensor
                torch.Tensor: mask_sigmoid_detach

        """
        self.actor_module.eval()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "sam_embed", "unpadded_hw"]
        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["pixel_values", "image_grid_thw"]
        else:
            non_tensor_select_keys = None

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst, mask_sigmoid_detach_lst = [], []
        for micro_batch in tqdm(micro_batches, desc="Compute log probs", disable=(self.rank != 0)):
            micro_batch.to("cuda")
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            forward_output = self._forward_micro_batch(model_inputs, meta_info=data.meta_info, forward_seg=gen_mask)

            log_probs_lst.append(forward_output[1])
            if gen_mask:
                mask_sigmoid_detach_lst.append(forward_output[2]["mask_sigmoid"])

        log_probs = torch.concat(log_probs_lst, dim=0)

        if not gen_mask:
            return log_probs

        mask_sigmoid_detach = torch.concat(mask_sigmoid_detach_lst, dim=0)
        return log_probs, mask_sigmoid_detach

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages",
                       "sam_embed", "mask_float_256", "unpadded_hw"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["pixel_values", "image_grid_thw"]
        else:
            non_tensor_select_keys = None

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        n = len(mini_batches)
        for i, mini_batch in enumerate(mini_batches):
            gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
            )
            micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

            self.actor_optimizer.zero_grad()

            for micro_batch in tqdm(micro_batches, desc=f"Update policy [{i + 1}/{n}]", disable=(self.rank != 0)):
                micro_batch.to("cuda")
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                responses = model_inputs["responses"]
                response_length = responses.size(1)
                attention_mask = model_inputs["attention_mask"]
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = model_inputs["old_log_probs"]
                advantages = model_inputs["advantages"]

                clip_ratio = self.config.clip_ratio
                entropy_coef = self.config.entropy_coef

                # all return: (bsz, response_length)
                entropy, log_prob, seg_outputs = self._forward_micro_batch(
                    model_inputs, meta_info=data.meta_info, forward_seg=True
                )

                ref_logits_256, gt_mask_256 = seg_outputs["ref_logits_256"], model_inputs["mask_float_256"]
                mask_logits, mask_sigmoid = seg_outputs["mask_logits"], seg_outputs["mask_sigmoid"]
                gt_mask_256 = gt_mask_256.to(device=ref_logits_256.device, dtype=ref_logits_256.dtype)

                # ========== seg loss ===========
                if self.config.ignore_pad_loss:
                    dev = ref_logits_256.device
                    up_hw = torch.as_tensor(model_inputs["unpadded_hw"], device=dev, dtype=torch.long)  # (B,2)
                    if up_hw.ndim != 2 or up_hw.size(-1) != 2:
                        raise ValueError(f"'unpadded_hw' shape must be (B,2), got {tuple(up_hw.shape)}")

                    # 1024 → 256 的上取整等价式：(x + 3) // 4
                    h256 = ((up_hw[:, 0].clamp(min=0) + 3) // 4).clamp(max=256)  # (B,)
                    w256 = ((up_hw[:, 1].clamp(min=0) + 3) // 4).clamp(max=256)  # (B,)

                    B, H, W = up_hw.shape[0], 256, 256
                    yy = torch.arange(H, device=dev).view(1, H, 1)  # (1,H,1)
                    xx = torch.arange(W, device=dev).view(1, 1, W)  # (1,1,W)
                    valid_256 = (yy < h256.view(B, 1, 1)) & (xx < w256.view(B, 1, 1))  # (B,H,W) bool
                    valid_256 = valid_256.float().unsqueeze(1)  # (B,1,H,W)

                    bce_map = F.binary_cross_entropy_with_logits(ref_logits_256, gt_mask_256, reduction="none")
                    heatmap_bce = (bce_map * valid_256).sum() / (valid_256.sum() + 1e-6)
                    focal_map = sigmoid_focal_loss(mask_logits, gt_mask_256, reduction="none")
                    focal_loss = (focal_map * valid_256).sum() / (valid_256.sum() + 1e-6)
                else:
                    heatmap_bce = F.binary_cross_entropy_with_logits(ref_logits_256, gt_mask_256, reduction="mean")
                    focal_loss = sigmoid_focal_loss(mask_logits, gt_mask_256, reduction="mean")
                    valid_256 = None

                heatmap_dice_loss = 1.0 - soft_dice(
                    torch.sigmoid(ref_logits_256), gt_mask_256, valid=valid_256, reduction="mean"
                )
                dice_loss = 1.0 - soft_dice(mask_sigmoid, gt_mask_256, valid=valid_256, reduction="mean")

                if self.config.adjust_loss_step < 0 or self.global_steps < self.config.adjust_loss_step:
                    dice_coef, focal_coef = self.config.dice_loss_coef, self.config.focal_loss_coef
                else:
                    dice_coef, focal_coef = self.config.dice_loss_coef_new, self.config.focal_loss_coef_new
                heatmap_dice_coef = dice_coef * self.config.heatmap_dice_rate
                seg_loss = (heatmap_bce
                            + dice_coef * dice_loss
                            + focal_coef * focal_loss
                            + heatmap_dice_loss * heatmap_dice_coef)

                pg_loss, pg_clipfrac, ppo_kl, ppo_ratio = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    eos_mask=response_mask,
                    cliprange=clip_ratio,
                )

                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coef

                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    # compute kl loss
                    kld = core_algos.kl_penalty(
                        logprob=log_prob,
                        ref_logprob=ref_log_prob,
                        kl_penalty=self.config.kl_loss_type,
                    )
                    kl_loss = verl_F.masked_mean(kld, response_mask)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                loss = (self.config.pg_loss_coef * policy_loss
                        + self.config.seg_loss_coef * seg_loss) / gradient_accumulation

                loss.backward()

                batch_metrics = {
                    "actor/policy_loss": policy_loss.detach().item(),
                    "actor/seg_loss": seg_loss.detach().item(),
                    "actor/dice_loss": dice_loss.detach().item(),
                    "actor/heatmap_bce": heatmap_bce.detach().item(),
                    "actor/focal_loss": focal_loss.detach().item(),
                    "actor/ratio_max": ppo_ratio.detach().max().item(),
                    "actor/pg_loss": pg_loss.detach().item(),
                    "actor/pg_abs": pg_loss.abs().detach().item(),
                    "actor/kl_loss": kl_loss.detach().item() if self.config.use_kl_loss else 0.0,
                    "actor/entropy": entropy_loss.detach().item(),
                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    "dice_coef": dice_coef,
                    "focal_coef": focal_coef,
                    "seg_coef": self.config.seg_loss_coef,
                    "kl_coef": self.config.kl_loss_coef
                }
                append_to_dict(metrics, batch_metrics)

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics

    def evaluate(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.eval()
        save_metric_path = os.path.join(data.meta_info["write_eval_dir"], "rank_"+str(self.rank)+".txt")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "mask_bool_gt_padded",
                       "image_id", "sam_embed", "unpadded_hw", "original_hw"]
        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["pixel_values", "image_grid_thw"]
        else:
            non_tensor_select_keys = None
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        eval_metrics = defaultdict(list)
        inters, unions = 0.0, 0.0
        for i, mini_batch in enumerate(mini_batches):
            micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

            for micro_batch in micro_batches:
                micro_batch.to("cuda")
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                _, _, seg_outputs = self._forward_micro_batch(
                    model_inputs, meta_info=data.meta_info, forward_seg=True
                )

                mask_logits, image_ids, ref_logits = (
                    seg_outputs["mask_logits"], model_inputs["image_id"], seg_outputs["ref_logits_256"])
                up_hws, orig_hws = model_inputs["unpadded_hw"],  model_inputs["original_hw"]
                masks_gt_padded = model_inputs["mask_bool_gt_padded"].to(device=mask_logits.device, dtype=torch.bool)
                for b in range(mask_logits.shape[0]):
                    mask_logit_256, mask_gt_padded, image_id, ref_logit_256 = (mask_logits[b].unsqueeze(0),
                        masks_gt_padded[b].unsqueeze(0), image_ids[b].item(), ref_logits[b].unsqueeze(0))

                    # ====== postprocess ======
                    up_h, up_w = up_hws[b][0].item(), up_hws[b][1].item()
                    orig_h, orig_w = orig_hws[b][0].item(), orig_hws[b][1].item()
                    mask_gt = mask_gt_padded[..., :orig_h, :orig_w]

                    mask_logit = F.interpolate(
                        mask_logit_256, (1024, 1024), mode="bilinear", align_corners=False,
                    )
                    mask_logit = mask_logit[..., :up_h, :up_w]
                    mask_logit = F.interpolate(mask_logit, (orig_h, orig_w), mode="bilinear", align_corners=False)
                    mask_bool = (mask_logit > 0.0)  # [1, 1, h, w]
                    ref_logit = F.interpolate(
                        ref_logit_256, (1024, 1024), mode="bilinear", align_corners=False,
                    )
                    ref_logit = ref_logit[..., :up_h, :up_w]
                    ref_logit = F.interpolate(ref_logit, (orig_h, orig_w), mode="bilinear", align_corners=False)
                    heatmap_bool = (ref_logit > 0.0)  # [1, 1, h, w]

                    iou, inter, union = compute_iou(mask_bool, mask_gt)
                    heatmap_iou, _, _ = compute_iou(heatmap_bool, mask_gt)
                    eval_metrics[image_id].append([iou[0].item(), heatmap_iou[0].item()])
                    inters += inter.item()
                    unions += union.item()

        eval_metrics["inters"], eval_metrics["unions"] = inters, unions
        with open(save_metric_path, "a", encoding="utf-8") as f:
            f.write(f"--------- eval_metrics ---------\n")
            f.write(json.dumps(eval_metrics, ensure_ascii=False, indent=2))
            f.write("\n\n")

        return eval_metrics


def soft_dice(
        prob: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        eps: float = 1e-6,
        reduction: str = "mean",
) -> torch.Tensor:
    """Compute soft Dice score over (B, C, H, W) probabilities and targets, then mean-reduce over batch."""
    target = target.to(dtype=prob.dtype, device=prob.device)

    # --- minimal change: apply valid mask if provided ---
    if valid is not None:
        valid = valid.to(dtype=prob.dtype, device=prob.device)
        if valid.ndim == prob.ndim - 1:  # allow (B,H,W)
            valid = valid.unsqueeze(1)
        prob = prob * valid
        target = target * valid

    # sum over channel/spatial dims, keep batch
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter) / (union + eps)

    if reduction == "mean":
        return dice.mean()
    if reduction == "sum":
        return dice.sum()
    return dice
