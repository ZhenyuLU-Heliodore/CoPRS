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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os, re
import json
import uuid
import shutil
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
from codetiming import Timer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer import core_algos
from verl.trainer.config import PPOConfig
from verl.utils.rl_dataset import RLHFDataset, GetCollate, collate_fn
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import Tracking
from verl.workers.fsdp_workers import FSDPWorker
from pathlib import Path


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


def latest_ckpt_path(save_ckpt_dir: Optional[str], load_ckpt_path: Optional[str]):
    """Pick the newer between explicit load_ckpt_path and the latest step_* dir under save_ckpt_dir.
    If a path has no 'step_#_ep_#' pattern, treat its step as 0. Returns None if neither exists."""
    best_dir, sd = None, -1
    if save_ckpt_dir:
        try:
            for n in os.listdir(save_ckpt_dir):
                m = re.search(r"step_(\d+)_ep_\d+", n)
                if m:
                    s = int(m.group(1))
                    if s > sd:
                        sd, best_dir = s, os.path.join(save_ckpt_dir, n)
        except OSError:
            pass
    ld = 0
    if load_ckpt_path:
        m = re.search(r"step_(\d+)", load_ckpt_path)
        if m: ld = int(m.group(1))
    return load_ckpt_path if (load_ckpt_path and ld >= sd) else best_dir


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == "gae":
        values = data.batch["values"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch["token_level_rewards"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, eos_mask=response_mask, gamma=gamma, lam=lam
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "grpo":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "reinforce_plus_plus":
        token_level_rewards = data.batch["token_level_rewards"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "remax":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch["reward_baselines"]

        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards, reward_baselines=reward_baselines, eos_mask=response_mask
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: Dict[str, Any]):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)

    return metrics


def _compute_response_info(batch: DataProto):
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True):
    # TODO: add response length
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        if self.config.algorithm.adv_estimator == "gae":
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == "grpo":
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == "reinforce_plus_plus":
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == "remax":
            self.use_critic = False
        else:
            raise NotImplementedError

        # get_collate = GetCollate(tokenizer=tokenizer, llm_version=self.config.worker.llm_version)
        # self.collate_fn = get_collate.collate_fn
        self.collate_fn = collate_fn
        self.global_steps, self.curr_episode = 0, 0

        # retrain from checkpoint automatically
        self.config.trainer.load_checkpoint_path = latest_ckpt_path(
            save_ckpt_dir=self.config.trainer.save_checkpoint_path,
            load_ckpt_path=self.config.trainer.load_checkpoint_path
        )

    def create_dataloader(self, mode: str = "train"):
        train_files = [p.strip() for p in self.config.data.train_files.split("|") if p.strip()]
        train_datasets = []
        for train_file in train_files:
            train_datasets.append(
                RLHFDataset(
                    data_anno_path=train_file,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    llm_version=self.config.worker.llm_version,
                    sam_embed_dir=self.config.data.sam_embed_dir,
                    k_max_objects=1,
                    mode=mode,
                    prompt_key=self.config.data.prompt_key,
                    max_prompt_length=self.config.data.max_prompt_length,
                    truncation="right",
                    system_prompt=self.config.data.system_prompt,
                    min_pixels=self.config.data.min_pixels,
                    max_pixels=self.config.data.max_pixels,
                )
            )
        self.train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            num_workers=8,
            drop_last=True,
            collate_fn=self.collate_fn,
            sampler=sampler,
        )

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes
        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
            rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Rollout], config=self.config.worker, role="rollout"
            )
            self.resource_pool_to_cls[resource_pool]["rollout"] = rollout_cls

            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
            actor_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Actor], config=self.config.worker, role="actor"
            )
            self.resource_pool_to_cls[resource_pool]["actor"] = actor_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg: FSDPWorker = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg: FSDPWorker = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg: FSDPWorker = all_wg["rm"]
            self.rm_wg.init_model()

        if self.hybrid_engine:
            self.actor_rollout_wg: FSDPWorker = all_wg["actor_rollout"]
            self.actor_rollout_wg.init_model()
            self.actor_rollout_wg.set_actor_global_step(self.global_steps)
        else:
            self.rollout_wg: FSDPWorker = all_wg["rollout"]
            self.rollout_wg.init_model()
            self.actor_wg: FSDPWorker = all_wg["actor"]
            self.actor_wg.init_model()
            self.actor_wg.set_actor_global_step(self.global_steps)

    def _save_checkpoint(self, dir_name=None, save_llm_hf=False):
        actor = self.actor_rollout_wg if self.hybrid_engine else self.actor_wg
        # path: {save_checkpoint_path}/step_{global_steps}_ep_{curr_episode}/actor
        dir_name = dir_name if dir_name else f"step_{self.global_steps}_ep_{self.curr_episode}"
        local_global_step_folder = os.path.join(self.config.trainer.save_checkpoint_path, dir_name)
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor.save_checkpoint(
            actor_local_path,
            self.global_steps,
            remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
            save_llm_hf=save_llm_hf,
        )

        local_latest_ckpt_itr = os.path.join(
            self.config.trainer.save_checkpoint_path, "latest_ckpt_itr.txt"
        )
        with open(local_latest_ckpt_itr, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self, load_optim=None):
        actor = self.actor_rollout_wg if self.hybrid_engine else self.actor_wg
        load_path = self.config.trainer.load_checkpoint_path
        if load_path is None:
            return

        print(f"Load from checkpoint: {load_path}")

        base = os.path.basename(os.path.normpath(load_path))
        m = re.search(r"step_(\d+)_ep_(\d+)", base)
        self.global_steps, self.curr_episode = (int(m.group(1)), int(m.group(2))) if m else (0, 0)

        actor_path = os.path.join(load_path, "actor")

        load_optim = load_optim if load_optim else (False if self.global_steps == 0 else True)
        actor.load_checkpoint(actor_path, load_optim=load_optim)

    def _update_latest_checkpoint(self):
        root = self.config.trainer.save_checkpoint_path
        try:
            old_latest_dirs = [e.path for e in os.scandir(root) if e.is_dir() and "latest" in e.name]
        except FileNotFoundError:
            os.makedirs(root, exist_ok=True)
            old_latest_dirs = []

        latest_dir_name = f"latest_step_{self.global_steps}_ep_{self.curr_episode}"
        self._save_checkpoint(dir_name=latest_dir_name)

        new_path = os.path.join(root, latest_dir_name)
        for p in old_latest_dirs:
            if os.path.abspath(p) != os.path.abspath(new_path):
                shutil.rmtree(p, ignore_errors=True)

    def _write_train_metrics(self, metrics, reward_score):
        path = os.path.join(self.config.trainer.save_checkpoint_path, "metrics.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"--------- curr episode {self.curr_episode} --------- global step {self.global_steps} ---------\n")
            f.write("reward score: \n")
            f.write(f"{reward_score}\n")
            f.write("training metrics:\n")
            f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
            f.write("\n\n")

    def _get_write_evaluate_dir(self):
        out_dir = Path(self.config.trainer.eval_checkpoints_dir).parent / "eval_metrics"
        out_dir = out_dir / Path(self.config.data.train_files).name
        path = out_dir / f"step_{self.global_steps}_ep_{self.curr_episode}/"
        path.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        actor = self.actor_rollout_wg if self.hybrid_engine else self.actor_wg
        rollout = self.actor_rollout_wg if self.hybrid_engine else self.rollout_wg

        # load checkpoint before doing anything
        self._load_checkpoint()

        # save config.txt
        save_dir = self.config.trainer.save_checkpoint_path
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2))

        assert self.config.trainer.total_episodes - self.curr_episode > 0, "No episode left to train."
        for _ in range(self.config.trainer.total_episodes - self.curr_episode):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if "pixel_values" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "pixel_values", "image_grid_thw", "raw_prompt_ids", "raw_prompt", "images"
                        ],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "raw_prompt"],
                    )

                with _timer("step", timing_raw):
                    # generate a batch, repeat n times before generation
                    with _timer("gen", timing_raw):  # wg: worker group
                        gen_batch_output = rollout.generate_sequences(gen_batch)

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    batch.meta_info["llm_version"] = self.config.worker.llm_version.lower()
                    batch.meta_info["image_token_id"] = self.tokenizer.convert_tokens_to_ids("<image>")
                    batch.meta_info["replace_token_id"] = self.tokenizer.unk_token_id \
                        if self.tokenizer.unk_token_id is not None else self.tokenizer.eos_token_id

                    # recompute old_log_probs
                    with _timer("old log prob with various forward results", timing_raw):
                        various_forward_info = actor.compute_log_prob(batch)
                        batch = batch.union(various_forward_info)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with _timer("adv", timing_raw):
                        if self.use_reward_model:
                            raise NotImplementedError

                        reward_tensor = self.reward_fn(batch)
                        reward_scalar = reward_tensor.detach().sum(dim=-1).mean().item()
                        batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.worker.actor.use_kl_loss:  # not grpo
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.worker.rollout.n,
                        )

                    with _timer("update_actor", timing_raw):
                        actor_output = actor.update_actor(batch)

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    save_freq, save_llm_freq = self.config.trainer.save_freq, self.config.trainer.save_llm_hf_freq
                    if save_freq > 0 and self.global_steps % save_freq == 0 and self.global_steps > 0:
                        with _timer("save_checkpoint", timing_raw):
                            save_llm = False if (save_llm_freq is None or save_llm_freq < 0) else (
                                not bool(self.global_steps % (save_freq * save_llm_freq)))
                            self._save_checkpoint(save_llm_hf=save_llm)

                    if self.global_steps % 100 == 0 and self.global_steps > 0:  # replace latest checkpoint
                        self._update_latest_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                self._write_train_metrics(actor_output_metrics, reward_scalar)

                self.global_steps += 1
                actor.set_actor_global_step(self.global_steps)

            self.curr_episode += 1

        self._save_checkpoint(save_llm_hf=True)

    def evaluate(self):
        actor = self.actor_rollout_wg if self.hybrid_engine else self.actor_wg
        rollout = self.actor_rollout_wg if self.hybrid_engine else self.rollout_wg
        assert self.config.worker.rollout.n == 1

        # search and sort for further iterations.
        ckpt_root = Path(self.config.trainer.eval_checkpoints_dir)
        step_pat = re.compile(r"step_(\d+)")
        pairs = []
        for p in ckpt_root.iterdir():
            if p.is_dir():
                m = step_pat.search(p.name)
                if m:
                    pairs.append((int(m.group(1)), p))
        st, ed = self.config.trainer.eval_step_start, self.config.trainer.eval_step_end
        st = 0 if st == -1 else st
        ed = max((i for i, _ in pairs), default=0) if ed == -1 else ed
        selected_paths = [p.resolve() for i, p in sorted(pairs, key=lambda t: t[0], reverse=True) if st <= i <= ed]

        with torch.inference_mode():
            for selected_path in selected_paths:
                self.config.trainer.load_checkpoint_path = str(selected_path)
                self._load_checkpoint(load_optim=False)

                # call train_dataloader API
                for batch_dict in self.train_dataloader:
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.meta_info["temperature"] = self.config.worker.rollout.temperature

                    write_eval_dir = Path(os.path.join(ckpt_root, "eval_metrics/", f"step_{self.global_steps}"))
                    write_eval_dir.mkdir(parents=True, exist_ok=True)
                    batch.meta_info["write_eval_dir"] = str(write_eval_dir)

                    # pop those keys for generation
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "pixel_values", "image_grid_thw", "raw_prompt_ids", "raw_prompt", "images"
                        ],
                    )

                    gen_batch_output = rollout.generate_sequences(gen_batch)

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    batch = batch.union(gen_batch_output)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    batch.meta_info["llm_version"] = self.config.worker.llm_version.lower()
                    batch.meta_info["image_token_id"] = self.tokenizer.convert_tokens_to_ids("<image>")
                    batch.meta_info["replace_token_id"] = self.tokenizer.unk_token_id \
                        if self.tokenizer.unk_token_id is not None else self.tokenizer.eos_token_id

                    actor.evaluate_actor(batch)
