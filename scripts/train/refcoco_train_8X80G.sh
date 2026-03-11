######## Devices ########
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

######## bind attention backend with bf16 support ########
SUPP_BF16=true    # <<<<<< A100为true

if [ "$SUPP_BF16" = "true" ]; then export VLLM_ATTENTION_BACKEND=FLASH_ATTN; DTYPE=bfloat16; SUPP_BF16_ARG=true;
else export VLLM_ATTENTION_BACKEND=SDPA; export VLLM_USE_TRITON=0; export XFORMERS_FORCE_DISABLE_TRITON=1; DTYPE=float16; SUPP_BF16_ARG=false; fi

######## function tools ########
set -euo pipefail
ROOT_DIR_STORAGE=""; LOAD_CKPT_PATH=""; RUN_FLAG="default"
usage(){ echo "Usage: $0 [-r|--root DIR] [-c|--ckpt PATH] [-f|--run_flag RUN_FLAG]"; exit 1; }
while [ $# -gt 0 ]; do case "$1" in
  -r|--root|-c|--ckpt|-f|--run_flag)
    [ $# -ge 2 ] || usage; key="$1"; val="$2"; shift 2;
    case "$key" in -r|--root) ROOT_DIR_STORAGE="$val";; -c|--ckpt) LOAD_CKPT_PATH="$val";; -f|--run_flag) RUN_FLAG="$val";; esac ;;
  *) usage ;; esac; done
prefix(){ printf '%s' "${ROOT_DIR_STORAGE:+${ROOT_DIR_STORAGE%/}/}$1"; }

######## PATH NAME ########
DATA_DIR="$(prefix 'data/refcoco/refcoco_train')" # <<<<<<
MODEL_PATH="$(prefix 'pretrained_models/RSeg-7B')"
RUN_NAME="$(basename "$0" .sh)"
if [[ -z "$LOAD_CKPT_PATH" ]]; then LOAD_PATH=null; else LOAD_PATH="${ROOT_DIR_STORAGE:+${ROOT_DIR_STORAGE%/}/}$LOAD_CKPT_PATH"; fi

######## Disable NCCL InfiniBand/RDMA and wandb ########
export NCCL_IB_DISABLE=1; export WANDB_MODE=offline; export WANDB_DIR="$(prefix 'wandb')"

######## Grouped CLI args ########
ARGS=(
  ### data ###
  data.train_files=${DATA_DIR}
  data.sam_embed_dir="$(prefix 'data/refcoco_series_sam_embed')" # <<<<<<
  trainer.load_checkpoint_path=${LOAD_PATH}
  config=training_scripts/initial.yaml

  ### batch size ###   (global_batch_size * n) / nnodes 要被 micro_batch_size_per_device 整除
  worker.rollout.n=8 # <<<<<<
  data.rollout_batch_size=16 # <<<<<<
  worker.actor.global_batch_size=16 # <<<<<<
  worker.actor.micro_batch_size_per_device_for_update=2 # <<<<<<
  worker.actor.micro_batch_size_per_device_for_experience=8 # <<<<<<

  ### worker ###
  worker.supp_bf16=${SUPP_BF16_ARG}
  worker.actor.model.model_path=${MODEL_PATH}
  worker.actor.model.init_sam_ckpt="$(prefix "pretrained_models/sam_vit_h_4b8939.pth")"

  ### rollout ###
  worker.rollout.tensor_parallel_size=4  # <<<<<<
  worker.rollout.max_num_seqs=64 # <<<<<<
  worker.rollout.gpu_memory_utilization=0.6 # <<<<<<
  worker.rollout.dtype=${DTYPE}
  worker.rollout.enable_chunked_prefill=${SUPP_BF16_ARG}

  ### trainer ###
  trainer.n_gpus_per_node=8 # <<<<<<
  trainer.total_episodes=10 # <<<<<<
  trainer.save_freq=100
  trainer.save_llm_hf_freq=10
  trainer.save_checkpoint_path="$(prefix "checkpoints/${RUN_NAME}/${RUN_FLAG}")"

  ### loss coef and lr ###
  worker.actor.optim.lr_head_rate=25
  worker.actor.optim.lr_pe_rate=10
  worker.actor.optim.lr_decoder_rate=5
  worker.actor.optim.base_lr=1.6e-6
  worker.actor.optim.lr_final_div_factor=6.67

  worker.actor.seg_loss_coef=0.3
  worker.actor.kl_loss_coef=0.2
  worker.actor.adjust_loss_step=300
  worker.actor.entropy_coef=0.0
  worker.actor.dice_loss_coef_new=2.0
  worker.actor.focal_loss_coef_new=5.0
)

set -x; PY="$(prefix 'coprs/bin/python')"; [ -x "$PY" ] || PY=python
PYTHONUNBUFFERED=1 "$PY" -u -m verl.trainer.main "${ARGS[@]}" 2>&1 | tee -a "$(prefix 'print_log.txt')"