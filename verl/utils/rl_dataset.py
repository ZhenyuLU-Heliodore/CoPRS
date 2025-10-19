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

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index


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
        sam_embed_dir: str = "",
        prompt_key="prompt",
        max_prompt_length=1024,
        truncation="error",
        system_prompt=None,
        max_pixels=None,
        min_pixels=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        self.sam_embed_root = sam_embed_dir if os.path.isdir(sam_embed_dir) else None
        self.dataset = load_from_disk(data_anno_path)  # you can load from disk if you have already downloaded the dataset
        
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
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataset[index]

        prob_key = self.prompt_key if self.prompt_key in row_dict else ("problem" if "problem" in row_dict else "text")
        if "text" in row_dict and isinstance(row_dict["text"], List):
            row_dict["text"] = row_dict["text"][0]
        question = row_dict[prob_key]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(
                Question=question.lower().strip("."),
            )},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if "image" in row_dict:
            row_dict["images"] = [row_dict["image"]]
        if "images" in row_dict:  # expand image token
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            row_dict["images"] = [
                process_image(image, self.max_pixels, self.min_pixels) for image in row_dict["images"]
            ]
            image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict.update(image_inputs)

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt:
                    prompt = prompt.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
        else:
            raw_prompt = prompt

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if "images" in row_dict:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )  # (3, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        if self.sam_embed_root is not None:
            embed_filename = os.path.join(self.sam_embed_root, row_dict["embed_path"])
            row_dict["sam_embed"] = torch.load(embed_filename, map_location="cpu")
        elif "sam_embed" in row_dict:
            row_dict["sam_embed"] = torch.as_tensor(row_dict["sam_embed"], dtype=torch.float16)
        if "ann_id" in row_dict:
            row_dict["ann_id"] = torch.as_tensor(int(row_dict["ann_id"]), dtype=torch.long)
        if "image_id" in row_dict:
            row_dict["image_id"] = torch.as_tensor(int(row_dict["image_id"]), dtype=torch.long)
        if "original_hw" in row_dict:
            row_dict["original_hw"] = torch.as_tensor([int(x) for x in row_dict["original_hw"]], dtype=torch.long)
        if "unpadded_hw" in row_dict:
            row_dict["unpadded_hw"] = torch.as_tensor([int(x) for x in row_dict["unpadded_hw"]], dtype=torch.long)
        if "mask_float_256" in row_dict:
            mask = torch.as_tensor(row_dict["mask_float_256"], dtype=torch.float32)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            row_dict["mask_float_256"] = mask.contiguous()
        # ====== pad gt mask to 4096 =======
        if "mask" in row_dict:
            mask = torch.as_tensor(row_dict["mask"], dtype=torch.bool)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            h, w = mask.shape[-2:]
            padded = torch.zeros((1, 8192, 8192), dtype=torch.bool)
            padded[..., :h, :w] = mask[..., :h, :w]
            row_dict["mask_bool_gt_padded"] = padded
            row_dict.pop("mask", None)

        return row_dict
