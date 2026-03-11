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
import os
import math
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl import DataProto

import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import pad_sequence_to_length
from verl.models.transformers.qwen2_5_vl import get_rope_index


class GetCollate:
    def __init__(self, tokenizer, llm_version):
        self.tokenizer = tokenizer
        self.llm_version = llm_version

    def collate_fn(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tensors = defaultdict(list)
        non_tensors = defaultdict(list)
        for feature in features:
            for key, value in feature.items():
                if isinstance(value, torch.Tensor):
                    tensors[key].append(value)
                else:
                    non_tensors[key].append(value)

        for key, value in tensors.items():
            if key not in ["pixel_values", "image_grid_thw"]:
                tensors[key] = torch.stack(value, dim=0)

        if "llava" in self.llm_version.lower():
            tensors["input_ids"] = clamp_llava_image_tokens(tensors["input_ids"], self.tokenizer)

        return {**tensors, **non_tensors}


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw"]:
            tensors[key] = torch.stack(value, dim=0)

    return {**tensors, **non_tensors}


def process_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
            self,
            data_anno_path: str,
            tokenizer: PreTrainedTokenizer,
            processor: Optional[ProcessorMixin],
            llm_version: str,  # in "qwen-2.5" or "llava-1.5"
            sam_embed_dir: str = "",
            k_max_objects: int = 1,
            mode: str = "train",  # in "train" and "eval"
            prompt_key="prompt",
            max_prompt_length=1024,
            truncation="right",
            system_prompt=None,
            max_pixels=None,
            min_pixels=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.llm_version = llm_version.lower()
        self.k_max_objects = k_max_objects
        self.mode = mode
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        self.is_multi_object = (k_max_objects > 1)
        self.sam_embed_root = sam_embed_dir if os.path.isdir(sam_embed_dir) else None
        self.dataset = load_from_disk(data_anno_path)

        if self.is_multi_object:
            k = int(self.k_max_objects)
            ds_ids = self.dataset.select_columns(["ann_id"])
            self._valid_idx = [i for i, x in enumerate(ds_ids) if len(x["ann_id"]) <= k]
            self.dataset_len = len(self._valid_idx)

            pos_tokens = " ".join(["<REF_POS>"] * k)
            tmpl = (
                "<image> Please find '{Question}' in the image.\n"
                "Compare the difference between object(s) and find the most closely matched object(s).\n"
                "Output the thinking process in <think> </think>.\n"
                "Then output exactly {K} reference position tokens, each written as <REF_POS>.\n"
                "These special tokens will be used to predict segmentation masks.\n"
                "Format:\n"
                "<think> your reasoning here </think>\n"
                "Here are the {K} reference positions (K={K}):\n"
                "{POS_TOKENS}"
            )
            self.user_prompt = tmpl.replace("{K}", str(k)).replace("{POS_TOKENS}", pos_tokens)

        else:
            self.dataset_len = len(self.dataset)
            self.user_prompt = (
                "<image> Please find '{Question}' in the image.\n"
                "Compare the difference between objects and find the most closely matched one.\n"
                "Output the thinking process in <think> </think>.\n"
                "Then generate one reference position token: <REF_POS>\n"
                "This special token will be used to predict a segmentation mask.\n"
                "Format:\n"
                "<think> your reasoning here </think>\n"
                "Here is the reference position: <REF_POS>"
            )

    def __len__(self):
        return self.dataset_len

    def _arrange_common_keys(self, row_dict):
        if self.sam_embed_root is not None:
            embed_filename = os.path.join(self.sam_embed_root, row_dict["embed_path"])
            row_dict["sam_embed"] = torch.load(embed_filename, map_location="cpu")
        elif "sam_embed" in row_dict:
            row_dict["sam_embed"] = torch.as_tensor(row_dict["sam_embed"], dtype=torch.float16)

        row_dict["image_id"] = torch.as_tensor(int(row_dict["image_id"]), dtype=torch.long)
        row_dict["original_hw"] = torch.as_tensor([int(x) for x in row_dict["original_hw"]], dtype=torch.long)
        row_dict["unpadded_hw"] = torch.as_tensor([int(x) for x in row_dict["unpadded_hw"]], dtype=torch.long)

        mask_256 = torch.as_tensor(row_dict["mask_float_256"], dtype=torch.float32)
        if mask_256.ndim == 2:
            mask_256 = mask_256.unsqueeze(0)

        if self.mode == "eval":  # read mask for evaluation
            mask = torch.as_tensor(row_dict["mask"], dtype=torch.bool)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

        # mask_float_256: [K, 256, 256]. Only during eval, gt_mask: [K, 8192, 8192]
        if self.is_multi_object:
            n_multi_objects = mask_256.shape[0]
            mask_256_padded = torch.zeros(self.k_max_objects, 256, 256, dtype=torch.float32)
            if n_multi_objects > 0:
                mask_256_padded[:n_multi_objects, :, :] = mask_256
            row_dict["mask_float_256"] = mask_256_padded.contiguous()
            if self.mode == "eval":
                mask_padded = torch.zeros((self.k_max_objects, 8192, 8192), dtype=torch.bool)
                if n_multi_objects > 0:
                    h, w = mask.shape[-2:]
                    mask_padded[:n_multi_objects, :h, :w] = mask
                row_dict["mask_bool_gt_padded"] = mask_padded

        else:
            n_multi_objects = -1
            row_dict["mask_float_256"] = mask_256.contiguous()
            if "ann_id" in row_dict:
                row_dict["ann_id"] = torch.as_tensor(int(row_dict["ann_id"]), dtype=torch.long)
            if self.mode == "eval":
                mask_padded = torch.zeros((1, 8192, 8192), dtype=torch.bool)
                h, w = mask.shape[-2:]
                mask_padded[..., :h, :w] = mask[..., :, :]
                row_dict["mask_bool_gt_padded"] = mask_padded

        row_dict.pop("mask", None)
        row_dict["n_multi_objects"] = torch.as_tensor(n_multi_objects, dtype=torch.int64)
        return row_dict

    def _arrange_qwen_tokens(self, row_dict, prob_key):
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt.format(Question=row_dict[prob_key])}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if "images" not in row_dict:
            raw_prompt, image_grid_thw = prompt, None
            image_inputs = {"pixel_values": None, "image_grid_thw": None}
        else:
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size ** 2
                index = 0
                while "<image>" in prompt:
                    num_placeholder = int((image_grid_thw[index].prod() // merge_length).item())
                    prompt = prompt.replace(
                        "<image>", "<|vision_start|>" + "<|placeholder|>" * num_placeholder + "<|vision_end|>", 1,
                    )
                    index += 1
                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)

        row_dict.update(image_inputs)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )
        if "images" in row_dict:
            position_ids = get_rope_index(
                self.processor, input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        return input_ids, attention_mask, position_ids, raw_prompt

    def _arrange_llava_tokens(self, row_dict, prob_key):
        vl_messages = [{"role": "system", "content": self.system_prompt},
                       {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row_dict[prob_key]}, ]}, ]

        prompt = self.processor.apply_chat_template(vl_messages, add_generation_prompt=True, tokenize=False)
        raw_prompt = prompt

        proc = self.processor(
            text=[prompt], images=row_dict["images"] if "images" in row_dict else None, return_tensors="pt",
            padding="max_length", truncation=True, max_length=self.max_prompt_length,
        )
        if "images" in row_dict:
            image_inputs = {"pixel_values": proc["pixel_values"], "image_grid_thw": None}
        else:
            image_inputs = {"pixel_values": None, "image_grid_thw": None}
        row_dict.update(image_inputs)

        return (
            proc["input_ids"][0],
            proc["attention_mask"][0],
            torch.clip(proc["attention_mask"][0].cumsum(dim=0) - 1, min=0, max=None),
            raw_prompt
        )

    def __getitem__(self, index):
        idx = self._valid_idx[index] if self.is_multi_object else index
        row_dict: Dict = self.dataset[idx]
        prob_key = self.prompt_key if self.prompt_key in row_dict else ("problem" if "problem" in row_dict else "text")
        texts = row_dict[prob_key]
        if isinstance(texts, list):
            chosen = next((s for s in texts if isinstance(s, str) and s.strip()), "")
            row_dict[prob_key] = chosen

        if "image" in row_dict:
            row_dict["images"] = [row_dict["image"]]
        if "images" in row_dict:  # expand image token
            row_dict["images"] = [process_image(img, self.max_pixels, self.min_pixels) for img in row_dict["images"]]

        if self.llm_version in ["qwen-2.5"]:
            input_ids, attention_mask, position_ids, raw_prompt = self._arrange_qwen_tokens(row_dict, prob_key)
        elif self.llm_version in ["llava-1.5"]:
            input_ids, attention_mask, position_ids, raw_prompt = self._arrange_llava_tokens(row_dict, prob_key)
        else:
            raise ValueError("Wrong type of LLM!")

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt"] = raw_prompt
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        row_dict = self._arrange_common_keys(row_dict)
        return row_dict


def clamp_llava_image_tokens(input_ids, image_id, replace_id):
    is_image = (input_ids == image_id)
    extra_mask = (is_image.to(torch.int32).cumsum(dim=1) > 576) & is_image

    replaced = int(extra_mask.sum().item())
    if replaced > 0:
        input_ids.masked_fill_(extra_mask, replace_id)
        print(f"WARNING: trimmed <image> beyond 576. Total replaces {replaced}")

    return input_ids
