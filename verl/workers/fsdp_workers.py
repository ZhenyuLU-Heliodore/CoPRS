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
The main entry point to run the PPO algorithm
"""

from typing import Literal

import torch
import os
import torch.distributed as dist
from torch.optim.lr_scheduler import OneCycleLR
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.modeling_utils import no_init_weights

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import get_tokenizer, get_processor
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
    materialize_meta_,
)
from verl.utils.model_utils import print_model_size, set_trainable
from verl.utils.performance import log_gpu_memory_usage
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_constant_schedule_with_warmup
from verl.workers.actor import DataParallelPPOActor
from verl.workers.config import FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from verl.workers.critic import DataParallelPPOCritic
from verl.workers.rollout.vllm_rollout import vLLMRollout
from verl.workers.sharding_manager import FSDPVLLMShardingManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.modeling import build_VLRefSegCore


class FSDPWorker(Worker):
    def __init__(
            self,
            config: WorkerConfig,
            role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # build device mesh for FSDP
        # TODO: support FSDP hybrid shard for larger model
        world_size = dist.get_world_size()
        self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_sequence_parallel_size = self.config.actor.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // self.ulysses_sequence_parallel_size, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_critic = self.role == "critic"
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._use_param_offload = False
        self._use_optimizer_offload = False
        if self._is_actor:
            self._use_param_offload = self.config.actor.offload.param_offload
            self._use_optimizer_offload = self.config.actor.offload.optimizer_offload
        elif self._is_critic:
            self._use_param_offload = self.config.critic.offload.param_offload
            self._use_optimizer_offload = self.config.critic.offload.optimizer_offload
        elif self._is_ref:
            # NOTE: it seems that manual offload is slowly than FSDP offload
            self._use_param_offload = self.config.ref.offload.param_offload

        # normalize config
        if self._is_actor:
            self.config.actor.global_batch_size *= self.config.rollout.n
            self.config.actor.global_batch_size_per_device = (
                    self.config.actor.global_batch_size // self.device_mesh.shape[
                0] * self.ulysses_sequence_parallel_size
            )
            assert (
                    self.config.actor.global_batch_size_per_device
                    % self.config.actor.micro_batch_size_per_device_for_update
                    == 0
            )
        elif self._is_critic:
            self.config.critic.global_batch_size *= self.config.rollout.n
            self.config.critic.global_batch_size_per_device = (
                    self.config.critic.global_batch_size // self.device_mesh.shape[
                0] * self.ulysses_sequence_parallel_size
            )
            assert (
                    self.config.critic.global_batch_size_per_device
                    % self.config.critic.micro_batch_size_per_device_for_update
                    == 0
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_global_step(self, step: int):
        if not self._is_actor:
            return
        self.actor.global_steps = int(step)

    def _build_model_optimizer(
            self,
            model_config: ModelConfig,
            fsdp_config: FSDPConfig,
            optim_config: OptimConfig,
            padding_free: bool = False,
    ) -> None:
        self.tokenizer = get_tokenizer(model_config.tokenizer_path, trust_remote_code=model_config.trust_remote_code)
        self.processor = get_processor(model_config.tokenizer_path)
        self.llm_model_config = AutoConfig.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_config.override_config,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
        except Exception:
            self.generation_config = GenerationConfig.from_model_config(self.llm_model_config)

        if padding_free:
            raise NotImplementedError("Padding free is not implemented yet.")

        torch_dtype = PrecisionType.to_dtype(
            "bf16" if fsdp_config.torch_dtype is None else fsdp_config.torch_dtype, supp_bf16=self.config.supp_bf16
        )
        self.llm_model_config.torch_dtype = torch_dtype

        model_map_keys = AutoModelForVision2Seq._model_mapping.keys()
        if self._is_critic:
            auto_class = AutoModelForTokenClassification
        elif type(self.llm_model_config) in model_map_keys or isinstance(self.llm_model_config, tuple(model_map_keys)):
            auto_class = AutoModelForVision2Seq
        else:
            auto_class = AutoModelForCausalLM

        # --------------------------- build llm -----------------------------
        if self.config.supp_bf16:
            attn_impl = "flash_attention_2"
        else:
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
            attn_impl = "sdpa"
        if self.rank == 0:
            model = auto_class.from_pretrained(
                model_config.model_path,
                config=self.llm_model_config,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = auto_class.from_config(
                    self.llm_model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl,
                    trust_remote_code=model_config.trust_remote_code,
                )
        set_trainable(model, True)

        assert isinstance(model, PreTrainedModel)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        dist.barrier()
        if self.rank == 0:
            print_model_size(model)

        log_gpu_memory_usage("After init from huggingface model")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype, self.config.supp_bf16),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype, self.config.supp_bf16),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype, self.config.supp_bf16),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        if fsdp_config.enable_full_shard:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.param_offload or fsdp_config.optimizer_offload:
            cpu_offload = CPUOffload(offload_params=fsdp_config.param_offload)
        else:
            cpu_offload = None

        if self.rank == 0:
            print(f"FSDP wrap policy: {auto_wrap_policy}.")

        #  ---------------------- build refseg core and FSDP wrap ------------------------
        self.fsdp_llm = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=get_init_fn(model, device="cuda") if self.rank != 0 else None,
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            forward_prefetch=False,
            use_orig_params=True,
            device_mesh=self.device_mesh,
        )

        if self._is_rollout and not (self._is_actor or self._is_critic or self._is_ref):
            return

        log_gpu_memory_usage("After Actor FSDP init")
        core_model = build_VLRefSegCore(
            llm=self.fsdp_llm,
            tokenizer=self.tokenizer,
            init_sam_ckpt=(model_config.init_sam_ckpt if self.rank == 0 else None),
            is_trainable=True,
        )
        set_trainable(core_model.heatmap_head, True)
        set_trainable(core_model.prompt_encoder, True)
        set_trainable(core_model.mask_decoder, True)

        self.fsdp_module = FSDP(
            core_model,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            auto_wrap_policy=None,
            cpu_offload=None,
            mixed_precision=mixed_precision,
            param_init_fn=None,
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            use_orig_params=True,
            device_mesh=self.device_mesh,
        )

        # -----------------------  build optimizer ---------------------------
        base_lr = optim_config.base_lr
        lr_llm, lr_head = base_lr, base_lr * optim_config.lr_head_rate
        lr_pe, lr_decoder = base_lr * optim_config.lr_pe_rate, base_lr * optim_config.lr_decoder_rate
        param_groups = []

        def trainable_params(m):
            return [p for p in m.parameters() if p.requires_grad]
        core = getattr(self.fsdp_module, "_fsdp_wrapped_module",
                       getattr(self.fsdp_module, "module", self.fsdp_module))
        ps = trainable_params(core.heatmap_head)
        if ps: param_groups.append({"params": ps, "lr": lr_head, "name": "head"})
        ps = trainable_params(core.prompt_encoder)
        if ps: param_groups.append({"params": ps, "lr": lr_pe, "name": "pe"})
        ps = trainable_params(core.mask_decoder)
        if ps: param_groups.append({"params": ps, "lr": lr_decoder, "name": "dec"})
        ps = trainable_params(core.llm)
        if ps: param_groups.append({"params": ps, "lr": lr_llm, "name": "llm"})

        if self._is_actor or self._is_critic:
            self.optimizer = torch.optim.AdamW(
                param_groups,
                betas=optim_config.betas,
                weight_decay=optim_config.weight_decay,
            )
            max_lrs = [g["lr"] for g in param_groups]
            self.lr_scheduler = OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                total_steps=optim_config.training_steps+1,
                pct_start=0.0,
                anneal_strategy='cos',
                div_factor=1.0,
                final_div_factor=optim_config.lr_final_div_factor,
                cycle_momentum=False
            )
        else:
            self.optimizer, self.lr_scheduler = None, None

        log_gpu_memory_usage("After actor optimizer init")

    def _build_rollout(self) -> None:
        # V100, do not support bf16 and flash_attn
        if not self.config.supp_bf16:
            os.environ["VLLM_ATTENTION_BACKEND"] = "SDPA"
            os.environ["VLLM_USE_TRITON"] = "0"
            os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        assert self.world_size % tp_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by tp_size: {tp_size}"
        )
        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=["dp", "tp"])
        log_gpu_memory_usage("Before building vllm rollout")

        self.config.rollout.use_raw_prompt = False if "qwen" in self.config.llm_version else True
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
        )
        log_gpu_memory_usage("After building vllm rollout")

        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_llm,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
        )
        log_gpu_memory_usage("After building sharding manager")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._is_critic:
            model_config = self.config.critic.model
            fsdp_config = self.config.critic.fsdp
            optim_config = self.config.critic.optim
            padding_free = self.config.critic.padding_free
        else:
            model_config = self.config.actor.model
            fsdp_config = self.config.actor.fsdp
            optim_config = self.config.actor.optim
            padding_free = self.config.actor.padding_free

        if self._is_actor or self._is_critic or self._is_ref or self._is_rollout:
            self._build_model_optimizer(
                model_config=model_config,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                padding_free=padding_free,
            )
            # get the original unwrapped module
            if hasattr(self, "fsdp_module"):
                self.unwrapped_model = self.fsdp_module._fsdp_wrapped_module
                if self._use_optimizer_offload and not self._is_critic:
                    offload_fsdp_optimizer(optimizer=self.optimizer)
                    log_gpu_memory_usage("After offload actor optimizer during init")

        if self._is_actor:
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._is_critic:
            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._is_rollout:
            self._build_rollout()

        if self._is_ref:
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.fsdp_module)

        if self._is_actor or self._is_critic:
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                tokenizer=self.tokenizer,
                processor=self.processor,
                extra_module_names=["heatmap_head", "prompt_encoder", "mask_decoder"],
                llm_fsdp=self.fsdp_llm,
            )

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, global_step: int = 0, remove_previous_ckpt: bool = False, save_llm_hf=False):
        assert self._is_actor or self._is_critic
        if self._use_param_offload:
            # load_fsdp_model(self.fsdp_llm)
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=path,
            global_step=global_step,
            remove_previous_ckpt=remove_previous_ckpt,
            save_llm_hf=save_llm_hf,
        )
        dist.barrier()
        if self._use_param_offload:
            # offload_fsdp_model(self.fsdp_llm)
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str, load_optim: bool = True):
        if self._use_param_offload:
            # load_fsdp_model(self.fsdp_llm)
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path=path, load_optim=load_optim)
        dist.barrier()
        if self._use_param_offload:
            # offload_fsdp_model(self.fsdp_llm)
            offload_fsdp_model(self.fsdp_module)

    """ActorRolloutRefWorker"""

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        log_gpu_memory_usage("Before update policy")
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            self.lr_scheduler.step()

            pgs = self.actor.actor_optimizer.param_groups
            for i, pg in enumerate(pgs):
                name = pg.get("name", f"g{i}")
                metrics[f"lr_{name}"] = float(pg["lr"])
            log_gpu_memory_usage("After update policy")

            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def evaluate_actor(self, data: DataProto):
        assert self._is_actor

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with torch.inference_mode():
                eval_metrics = self.actor.evaluate(data=data)

            output = DataProto(meta_info={"eval_metrics": eval_metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_llm)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            # after parameters sync with rollout, offload actor model to CPU
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_llm)

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)

            log_gpu_memory_usage("After entering rollout sharding manager")

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage("After rollout generation")

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        torch.cuda.empty_cache()  # clear kv cache
        log_gpu_memory_usage("After recompute log prob")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        """
        Simultaneously compute the mask_sigmoid_detach for reward_fn
        """
        assert self._is_actor
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            log_prob, mask_sigmoid_detach = self.actor.compute_log_prob(data=data, gen_mask=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": log_prob, "mask_sigmoid_detach": mask_sigmoid_detach},
                meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        torch.cuda.empty_cache()
        log_gpu_memory_usage("After compute_log_prob and mask_sigmoid_detach")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        torch.cuda.empty_cache()
        log_gpu_memory_usage("After compute_ref_log_prob")
        return output

    """CriticWorker"""

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._is_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        torch.cuda.empty_cache()
        return output
