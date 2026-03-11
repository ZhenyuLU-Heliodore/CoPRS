######## Devices ########
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

######## bind attention backend with bf16 support ########
SUPP_BF16=true    # <<<<<< A100为true

if [ "$SUPP_BF16" = "true" ]; then export VLLM_ATTENTION_BACKEND=FLASH_ATTN; DTYPE=bfloat16; SUPP_BF16_ARG=true;
else export VLLM_ATTENTION_BACKEND=SDPA; export VLLM_USE_TRITON=0; export XFORMERS_FORCE_DISABLE_TRITON=1; DTYPE=float16; SUPP_BF16_ARG=false; fi

######## load ckpt path for training or evaluate ########
LOAD_CKPT_PATH=""

######## function tools ########
set -euo pipefail
ROOT_DIR_STORAGE=""; RUN_FLAG="default"
usage(){ echo "Usage: $0 [-r|--root DIR] [-c|--ckpt PATH] [-f|--run_flag RUN_FLAG]"; exit 1; }
while [ $# -gt 0 ]; do case "$1" in
  -r|--root|-c|--ckpt|-f|--run_flag)
    [ $# -ge 2 ] || usage; key="$1"; val="$2"; shift 2;
    case "$key" in -r|--root) ROOT_DIR_STORAGE="$val";; -c|--ckpt) LOAD_CKPT_PATH="$val";; -f|--run_flag) RUN_FLAG="$val";; esac ;;
  *) usage ;; esac; done
prefix(){ printf '%s' "${ROOT_DIR_STORAGE:+${ROOT_DIR_STORAGE%/}/}$1"; }

######## PATH NAME ########
DATA_DIR="$(prefix 'data/refcoco/refcoco_validation')" # <<<<<<
MODEL_PATH="$(prefix 'pretrained_models/RSeg-7B')"
RUN_NAME="$(basename "$0" .sh)"
if [[ -z "$LOAD_CKPT_PATH" ]]; then LOAD_PATH=null; else LOAD_PATH="${ROOT_DIR_STORAGE:+${ROOT_DIR_STORAGE%/}/}$LOAD_CKPT_PATH"; fi

######## Disable NCCL InfiniBand/RDMA and wandb ########
export NCCL_IB_DISABLE=1; export WANDB_MODE=offline; export WANDB_DIR="$(prefix 'wandb')"

######## Grouped CLI args ########
ARGS=(
  ### data ###
  trainer.eval_checkpoints_dir="$(prefix 'checkpoints/refcoco_train_8X80G/default/')"
  trainer.eval_step_start=1000
  data.train_files=${DATA_DIR}
  data.sam_embed_dir="$(prefix 'data/refcoco_series_sam_embed')" # <<<<<<
  trainer.load_checkpoint_path=${LOAD_PATH}
  config=training_scripts/initial.yaml

  ### batch size ###   (global_batch_size * n) / nnodes 要被 micro_batch_size_per_device 整除
  worker.rollout.n=1 # <<<<<<
  data.rollout_batch_size=128 # <<<<<<
  worker.actor.global_batch_size=128 # <<<<<<
  worker.actor.micro_batch_size_per_device_for_update=8 # <<<<<<
  worker.actor.micro_batch_size_per_device_for_experience=16 # <<<<<<

  ### worker ###
  worker.supp_bf16=${SUPP_BF16_ARG}
  worker.actor.model.model_path=${MODEL_PATH}
  worker.actor.model.init_sam_ckpt="$(prefix "pretrained_models/sam_vit_h_4b8939.pth")"

  ### rollout ###
  worker.rollout.tensor_parallel_size=4  # <<<<<<
  worker.rollout.max_num_seqs=256 # <<<<<<
  worker.rollout.gpu_memory_utilization=0.55 # <<<<<<
  worker.rollout.dtype=${DTYPE}
  worker.rollout.enable_chunked_prefill=${SUPP_BF16_ARG}

  ### trainer ###
  trainer.n_gpus_per_node=8 # <<<<<<
)

set -x; PY="$(prefix 'v-seg/bin/python')"; [ -x "$PY" ] || PY=python
PYTHONUNBUFFERED=1 "$PY" -u -m verl.trainer.evaluate "${ARGS[@]}" 2>&1 | tee -a "$(prefix 'print_log.txt')"