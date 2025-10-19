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
from typing import List, Optional
import warnings

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType, FullStateDictConfig
)
from transformers import PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    If model._fsdp_wrapped_module contains an FSDP Model and other models,
    We must input extra_module_names and llm_fsdp

    A checkpoint manager that saves and loads (SPMD)
    - model (LLM shards + refseg_module parts)
    - optimizer (sharded)
    - lr_scheduler (full)
    - extra_states (rng, etc.)

    Optional (rank 0 only):
    - exports a flat state_dict of specified refseg submodules (e.g., heatmap_head/prompt_encoder/mask_decoder)
      to “refseg_modules.pt” with keys like 'heatmap_head.xxx'. No extra manifest/json needed.
    - save fsdp
    """

    def __init__(
            self,
            model: FSDP,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
            tokenizer: PreTrainedTokenizer,
            processor: ProcessorMixin,
            extra_module_names: Optional[List[str]] = None,
            llm_fsdp: Optional[FSDP] = None,
            *args,
            **kwargs,
    ):
        super().__init__(model, optimizer, lr_scheduler, tokenizer, processor)
        self.llm_fsdp = llm_fsdp if llm_fsdp is not None else model
        self.extra_module_names = extra_module_names
        self._core = getattr(model, "_fsdp_wrapped_module", getattr(model, "module", model))

    def load_checkpoint(self, path=None, load_init_llm_hf=False, *args, **kwargs):
        if path is None:
            return

        # ---------- every rank download its own checkpoint ----------
        if load_init_llm_hf:
            print("Skip loadding LLM checkpoint because load_init_llm_hf is True!")
        else:
            local_model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
            local_optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
            local_extra_state_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
            print(
                f"[rank-{self.rank}]: Loading from {local_model_path} and {local_optim_path} and {local_extra_state_path}"
            )
            model_state_dict = torch.load(local_model_path, map_location="cpu")
            extra_state_dict = torch.load(local_extra_state_path, map_location="cpu", weights_only=False)
            optimizer_state_dict = torch.load(local_optim_path, map_location="cpu")
            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]

            state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                self.model.load_state_dict(model_state_dict)
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(optimizer_state_dict)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])

            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

        # ---------- load refseg submodules (required when extra_module_names is set) ---------------
        if self.extra_module_names is not None:
            extra_core_path = os.path.join(path, "refseg_modules.pt")
            if not os.path.isfile(extra_core_path):
                raise FileNotFoundError(
                    f"[rank-{self.rank}] Missing refseg_modules.pt at {extra_core_path} "
                    f"(extra_module_names={self.extra_module_names})."
                )
            extra_sd = torch.load(extra_core_path, map_location="cpu")

            for name in self.extra_module_names:
                sub = getattr(self._core, name, None)
                prefix = f"{name}."
                sub_sd = {k[len(prefix):]: v for k, v in extra_sd.items() if k.startswith(prefix)}
                if sub is None:
                    print("\n" + "=" * 72 +
                          f"\n[WARN][rank-{self.rank}] refseg submodule '{name}' NOT found on core. Skip loading.\n" +
                          "=" * 72)
                    continue
                if not sub_sd:
                    print("\n" + "-" * 72 +
                          f"\n[WARN][rank-{self.rank}] No weights for '{name}' in refseg_modules.pt. Skip loading.\n" +
                          "-" * 72)
                    continue

                missing, unexpected = sub.load_state_dict(sub_sd, strict=False)
                if missing or unexpected:
                    print("\n" + "~" * 72 +
                          f"\n[WARN][rank-{self.rank}] load_state_dict('{name}') "
                          f"missing={list(missing)} unexpected={list(unexpected)}\n" +
                          "~" * 72)

    def save_checkpoint(self, local_path: str, global_step: int,
                        save_llm_hf=False, remove_previous_ckpt=False, *args, **kwargs):
        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        # TODO: shall we remove previous ckpt every save?
        if remove_previous_ckpt:
            self.remove_previous_save_local_path()
        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                if self.optimizer is not None:
                    optimizer_state_dict = self.optimizer.state_dict()
                else:
                    optimizer_state_dict = None
                if self.lr_scheduler is not None:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict()
                else:
                    lr_scheduler_state_dict = None

                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                print(f"[rank-{self.rank}]: Saving checkpoint to {os.path.abspath(model_path)}", flush=True)
                print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}", flush=True)
                torch.save(model_state_dict, model_path, _use_new_zipfile_serialization=False)
                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        # ----------- Rank-0: export refseg submodules  ----------------
        if self.rank == 0 and self.extra_module_names:
            extra_sd = {}
            for name in self.extra_module_names:
                sub = getattr(self._core, name, None)
                if sub is None:
                    print(f"[WARN][rank-{self.rank}] submodule '{name}' NOT found; skip.")
                    continue
                for k, v in sub.state_dict().items():
                    extra_sd[f"{name}.{k}"] = v.detach().cpu()

            torch.save(extra_sd, os.path.join(local_path, "refseg_modules.pt"))

        torch.distributed.barrier()

        # ------------ Rank-0 save llm config, processor and tokenizer in hf directory --------------
        wrapped_llm = getattr(self.llm_fsdp, "_fsdp_wrapped_module", getattr(self.llm_fsdp, "module", None))
        assert wrapped_llm is not None, "llm_fsdp must be an FSDP-wrapped HF model (inner module not found)"
        hf_local_path = os.path.join(local_path, "llm_hf")
        if self.rank == 0:
            os.makedirs(hf_local_path, exist_ok=True)
            print(f"[rank-{self.rank}]: Saving llm config/processor/tokenizer to {os.path.abspath(hf_local_path)}",
                  flush=True)
            wrapped_llm.config.save_pretrained(hf_local_path)
            if self.processor:
                self.processor.save_pretrained(hf_local_path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(hf_local_path)

        # ------------- (optional) save llm as HF format --------------
        if save_llm_hf:
            fsd_cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
            with FSDP.state_dict_type(self.llm_fsdp, StateDictType.FULL_STATE_DICT, fsd_cfg):
                llm_full_sd = self.llm_fsdp.state_dict()

            if self.rank == 0:
                print(f"[rank-{self.rank}]: Saving llm_hf to {os.path.abspath(hf_local_path)}", flush=True)
                wrapped_llm.save_pretrained(
                    hf_local_path,
                    state_dict=llm_full_sd,
                    safe_serialization=True,
                )
                del llm_full_sd

        torch.distributed.barrier()
        self.previous_save_local_path = local_path
