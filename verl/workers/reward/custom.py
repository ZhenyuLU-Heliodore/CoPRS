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


import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import (
    rseg_cot_compute_score,
)


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        if compute_score == "rseg_cot":
            self.compute_score = rseg_cot_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            # valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            # prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            if valid_response_length <= 0:
                continue
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            gt_mask = data_item.batch["mask_float_256"]
            pred_mask = data_item.batch["mask_sigmoid_detach"]
            if pred_mask.ndim == 3:  # -> (B=1,1,256,256)
                pred_mask = pred_mask.unsqueeze(0)
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.unsqueeze(0)

            score = self.compute_score(response_str, pred_mask, gt_mask)

            reward_tensor[i, valid_response_length - 1] = score

            # if already_print < self.num_examine:
            #     already_print += 1
            #     print("[prompt]", prompt_str)
            #     print("[response]", response_str)
            #     print("[ground_truth]", ground_truth)
            #     print("[score]", score)

        return reward_tensor
