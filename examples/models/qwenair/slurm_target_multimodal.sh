#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Canonical QwenAir multimodal diagnostic: four 8-GPU nodes, one Slurm task
# per GPU. Select the B200 or B300 account and partition on the sbatch command
# line; this script intentionally contains no site-specific scheduler values.

#SBATCH --job-name=qwenair-target-1024
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --exclusive

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_env() {
    local name=$1
    [[ -n "${!name:-}" ]] || die "${name} must be set"
}

require_positive_integer() {
    local name=$1
    local value=${!name:-}
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer; observed ${value:-unset}"
}

verify_git_revision() {
    local label=$1
    local source_dir=$2
    local expected_revision=$3
    local observed_revision
    local dirty_files

    [[ -d "${source_dir}" ]] || die "${label} source directory does not exist: ${source_dir}"
    observed_revision=$(git -C "${source_dir}" rev-parse HEAD) \
        || die "${label} source is not a readable Git checkout: ${source_dir}"
    [[ "${observed_revision}" == "${expected_revision}" ]] \
        || die "${label} revision ${observed_revision} does not match expected ${expected_revision}"

    if [[ "${QWENAIR_ALLOW_DIRTY_SOURCE:-0}" != "1" ]]; then
        dirty_files=$(git -C "${source_dir}" status --short --untracked-files=normal)
        [[ -z "${dirty_files}" ]] \
            || die "${label} source has uncommitted files; set QWENAIR_ALLOW_DIRTY_SOURCE=1 only for development gates"
    fi
}

worker_main() {
    shift

    export RANK="${SLURM_PROCID:?SLURM_PROCID is missing from the srun task}"
    export WORLD_SIZE="${SLURM_NTASKS:?SLURM_NTASKS is missing from the srun task}"
    local slurm_local_rank="${SLURM_LOCALID:?SLURM_LOCALID is missing from the srun task}"

    [[ "${WORLD_SIZE}" == "32" ]] || die "QwenAir EP32 requires WORLD_SIZE=32; observed ${WORLD_SIZE}"
    [[ -f "${QWENAIR_MODEL_CONFIG}" ]] || die "model config is not visible to rank ${RANK}: ${QWENAIR_MODEL_CONFIG}"

    local rank_cache_root="${QWENAIR_CACHE_ROOT}/ranks/rank-${RANK}"
    export TMPDIR="${rank_cache_root}/tmp"
    export TRITON_CACHE_DIR="${rank_cache_root}/triton"
    export TORCHINDUCTOR_CACHE_DIR="${rank_cache_root}/torchinductor"
    export TORCH_EXTENSIONS_DIR="${rank_cache_root}/torch-extensions"
    mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

    local source_path="${QWENAIR_MCORE_ROOT}:${QWENAIR_BRIDGE_ROOT}/src"
    if [[ -n "${QWENAIR_TE_PYTHON_OVERLAY:-}" ]]; then
        source_path="${QWENAIR_TE_PYTHON_OVERLAY}:${source_path}"
    fi
    export PYTHONPATH="${source_path}${PYTHONPATH:+:${PYTHONPATH}}"

    if [[ "${QWENAIR_PYTHON}" == */* ]]; then
        [[ -x "${QWENAIR_PYTHON}" ]] || die "Python executable is not visible or executable: ${QWENAIR_PYTHON}"
    else
        command -v "${QWENAIR_PYTHON}" >/dev/null || die "Python executable is not on PATH: ${QWENAIR_PYTHON}"
    fi

    local visible_device_count
    visible_device_count=$("${QWENAIR_PYTHON}" -c "import torch; print(torch.cuda.device_count())")
    [[ "${visible_device_count}" =~ ^[1-9][0-9]*$ ]] \
        || die "rank ${RANK} cannot see a CUDA device; torch reported ${visible_device_count:-no device count}"
    if [[ "${visible_device_count}" == "1" ]]; then
        # Slurm/Pyxis may map each task's assigned physical GPU to logical cuda:0.
        export LOCAL_RANK=0
    elif ((slurm_local_rank < visible_device_count)); then
        # --gpu-bind=none normally exposes all eight node-local GPUs.
        export LOCAL_RANK=${slurm_local_rank}
    else
        die "SLURM_LOCALID=${slurm_local_rank} is invalid with ${visible_device_count} visible CUDA devices"
    fi
    echo "QwenAir rank mapping: rank=${RANK} slurm_localid=${slurm_local_rank} visible_gpus=${visible_device_count} local_rank=${LOCAL_RANK}"

    local training_args=(
        --model-config "${QWENAIR_MODEL_CONFIG}"
        --dataset-revision "${QWENAIR_DATASET_REVISION}"
        --processor-revision "${QWENAIR_PROCESSOR_REVISION}"
        --dataset-split "${QWENAIR_DATASET_SPLIT}"
        --train-iters "${QWENAIR_TRAIN_ITERS}"
        --global-batch-size "${QWENAIR_GLOBAL_BATCH_SIZE}"
        --seq-length "${QWENAIR_SEQ_LENGTH}"
        --image-size "${QWENAIR_IMAGE_SIZE}"
        --tensorboard-dir "${QWENAIR_TENSORBOARD_DIR}"
    )
    if [[ -n "${QWENAIR_CHECKPOINT_DIR}" ]]; then
        training_args+=(--checkpoint-dir "${QWENAIR_CHECKPOINT_DIR}")
    fi

    cd "${QWENAIR_BRIDGE_ROOT}"
    exec "${QWENAIR_PYTHON}" -u examples/models/qwenair/finetune_qwenair_target_multimodal.py \
        "${training_args[@]}"
}

if [[ "${1:-}" == "--worker" ]]; then
    worker_main "$@"
fi

require_env QWENAIR_BRIDGE_ROOT
require_env QWENAIR_MCORE_ROOT
require_env QWENAIR_TE_SOURCE_ROOT
require_env QWENAIR_MODEL_CONFIG
require_env QWENAIR_RUN_DIR
require_env QWENAIR_CACHE_ROOT
require_env QWENAIR_PYTHON
require_env QWENAIR_EXPECTED_BRIDGE_COMMIT
require_env QWENAIR_EXPECTED_MCORE_COMMIT
require_env QWENAIR_EXPECTED_TE_COMMIT
require_env QWENAIR_EXPECTED_MODEL_CONFIG_SHA256

[[ "$#" -eq 0 ]] || die "configure this launcher with QWENAIR_* environment variables, not positional arguments"

[[ -n "${SLURM_JOB_ID:-}" ]] || die "submit this launcher with sbatch from a Slurm cluster"
[[ "${SLURM_JOB_NUM_NODES:-0}" == "4" ]] \
    || die "canonical launch requires four nodes; observed ${SLURM_JOB_NUM_NODES:-unset}"
[[ "${SLURM_NTASKS:-0}" == "32" ]] \
    || die "canonical launch requires 32 tasks; observed ${SLURM_NTASKS:-unset}"
[[ -f "${QWENAIR_MODEL_CONFIG}" ]] || die "model config does not exist: ${QWENAIR_MODEL_CONFIG}"

verify_git_revision "Megatron Bridge" "${QWENAIR_BRIDGE_ROOT}" "${QWENAIR_EXPECTED_BRIDGE_COMMIT}"
verify_git_revision "Megatron Core" "${QWENAIR_MCORE_ROOT}" "${QWENAIR_EXPECTED_MCORE_COMMIT}"
verify_git_revision "Transformer Engine" "${QWENAIR_TE_SOURCE_ROOT}" "${QWENAIR_EXPECTED_TE_COMMIT}"

if [[ -n "${QWENAIR_TE_PYTHON_OVERLAY:-}" ]]; then
    source_qsa="${QWENAIR_TE_SOURCE_ROOT}/transformer_engine/pytorch/attention/qsa.py"
    overlay_qsa="${QWENAIR_TE_PYTHON_OVERLAY}/transformer_engine/pytorch/attention/qsa.py"
    [[ -f "${source_qsa}" ]] || die "Transformer Engine source QSA file does not exist: ${source_qsa}"
    [[ -f "${overlay_qsa}" ]] || die "Transformer Engine overlay QSA file does not exist: ${overlay_qsa}"
    cmp --silent "${source_qsa}" "${overlay_qsa}" \
        || die "Transformer Engine Python overlay QSA differs from expected source commit"
fi

observed_config_sha256=$(sha256sum "${QWENAIR_MODEL_CONFIG}" | awk '{print $1}')
[[ "${observed_config_sha256}" == "${QWENAIR_EXPECTED_MODEL_CONFIG_SHA256}" ]] \
    || die "model config SHA256 ${observed_config_sha256} does not match expected ${QWENAIR_EXPECTED_MODEL_CONFIG_SHA256}"

export QWENAIR_TENSORBOARD_DIR=${QWENAIR_TENSORBOARD_DIR:-${QWENAIR_RUN_DIR}/tensorboard}
export QWENAIR_DATASET_REVISION=81fc5f3a41274c80f17b0406426d57cac57ce6fb
export QWENAIR_PROCESSOR_REVISION=2fc06364715b967f1860aea9cf38778875588b17
export QWENAIR_DATASET_SPLIT=${QWENAIR_DATASET_SPLIT:-train}
export QWENAIR_TRAIN_ITERS=${QWENAIR_TRAIN_ITERS:-1024}
export QWENAIR_GLOBAL_BATCH_SIZE=${QWENAIR_GLOBAL_BATCH_SIZE:-128}
export QWENAIR_SEQ_LENGTH=${QWENAIR_SEQ_LENGTH:-128}
export QWENAIR_IMAGE_SIZE=${QWENAIR_IMAGE_SIZE:-224}
export QWENAIR_CHECKPOINT_DIR=${QWENAIR_CHECKPOINT_DIR:-}
require_positive_integer QWENAIR_TRAIN_ITERS
require_positive_integer QWENAIR_GLOBAL_BATCH_SIZE
require_positive_integer QWENAIR_SEQ_LENGTH
require_positive_integer QWENAIR_IMAGE_SIZE
[[ -n "${QWENAIR_DATASET_SPLIT}" ]] || die "QWENAIR_DATASET_SPLIT must be non-empty"
((QWENAIR_GLOBAL_BATCH_SIZE % 32 == 0)) \
    || die "QWENAIR_GLOBAL_BATCH_SIZE must be divisible by the 32 dense-data-parallel ranks"
export QWENAIR_GRADIENT_ACCUMULATION_STEPS=$((QWENAIR_GLOBAL_BATCH_SIZE / 32))
export HF_HOME=${QWENAIR_HF_HOME:-${QWENAIR_CACHE_ROOT}/huggingface}
export NEMO_HOME=${QWENAIR_NEMO_HOME:-${QWENAIR_CACHE_ROOT}/nemo}
export XDG_CACHE_HOME=${QWENAIR_XDG_CACHE_HOME:-${QWENAIR_CACHE_ROOT}/xdg}
export UV_CACHE_DIR=${QWENAIR_UV_CACHE_DIR:-${QWENAIR_CACHE_ROOT}/uv}
export TORCH_HOME=${QWENAIR_TORCH_HOME:-${QWENAIR_CACHE_ROOT}/torch}
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HF_HUB_OFFLINE=${QWENAIR_HF_HUB_OFFLINE:-${HF_HUB_OFFLINE:-0}}
export TRANSFORMERS_OFFLINE=${QWENAIR_TRANSFORMERS_OFFLINE:-${TRANSFORMERS_OFFLINE:-0}}
export HF_DATASETS_OFFLINE=${QWENAIR_DATASETS_OFFLINE:-${HF_DATASETS_OFFLINE:-0}}

mkdir -p \
    "${QWENAIR_RUN_DIR}" \
    "${QWENAIR_TENSORBOARD_DIR}" \
    "${HF_HOME}" \
    "${NEMO_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${UV_CACHE_DIR}" \
    "${TORCH_HOME}" \
    "${QWENAIR_CACHE_ROOT}/ranks"

if [[ -z "${MASTER_ADDR:-}" ]]; then
    mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is missing}")
    [[ "${#allocated_nodes[@]}" -eq 4 ]] \
        || die "expected four allocated hosts; scontrol returned ${#allocated_nodes[@]}"
    MASTER_ADDR=${allocated_nodes[0]}
fi
if [[ -z "${MASTER_PORT:-}" ]]; then
    job_id_digits=${SLURM_JOB_ID//[!0-9]/}
    [[ -n "${job_id_digits}" ]] || die "cannot derive MASTER_PORT from SLURM_JOB_ID=${SLURM_JOB_ID}"
    job_id_suffix=${job_id_digits: -4}
    MASTER_PORT=$((15000 + 10#${job_id_suffix}))
fi
export MASTER_ADDR MASTER_PORT

bridge_revision=$(git -C "${QWENAIR_BRIDGE_ROOT}" rev-parse HEAD)
mcore_revision=$(git -C "${QWENAIR_MCORE_ROOT}" rev-parse HEAD)
te_revision=$(git -C "${QWENAIR_TE_SOURCE_ROOT}" rev-parse HEAD)
manifest_path="${QWENAIR_RUN_DIR}/source-manifest-${SLURM_JOB_ID}.txt"
cat >"${manifest_path}" <<EOF
slurm_job_id=${SLURM_JOB_ID}
slurm_nodes=${SLURM_JOB_NUM_NODES}
slurm_tasks=${SLURM_NTASKS}
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
bridge_root=${QWENAIR_BRIDGE_ROOT}
bridge_commit=${bridge_revision}
mcore_root=${QWENAIR_MCORE_ROOT}
mcore_commit=${mcore_revision}
transformer_engine_source_root=${QWENAIR_TE_SOURCE_ROOT}
transformer_engine_commit=${te_revision}
transformer_engine_python_overlay=${QWENAIR_TE_PYTHON_OVERLAY:-}
model_config=${QWENAIR_MODEL_CONFIG}
model_config_sha256=${observed_config_sha256}
python=${QWENAIR_PYTHON}
run_dir=${QWENAIR_RUN_DIR}
cache_root=${QWENAIR_CACHE_ROOT}
tensorboard_dir=${QWENAIR_TENSORBOARD_DIR}
dataset=tsystems/flickr8k
dataset_revision=${QWENAIR_DATASET_REVISION}
dataset_split=${QWENAIR_DATASET_SPLIT}
processor=Qwen/Qwen3.5-0.8B
processor_revision=${QWENAIR_PROCESSOR_REVISION}
hf_hub_offline=${HF_HUB_OFFLINE}
transformers_offline=${TRANSFORMERS_OFFLINE}
datasets_offline=${HF_DATASETS_OFFLINE}
world_size=32
nodes=4
tasks_per_node=8
expert_model_parallel_size=32
global_batch_size=${QWENAIR_GLOBAL_BATCH_SIZE}
micro_batch_size=1
gradient_accumulation_steps=${QWENAIR_GRADIENT_ACCUMULATION_STEPS}
train_iters=${QWENAIR_TRAIN_ITERS}
peak_learning_rate=3e-4
min_learning_rate=3e-5
lr_decay_style=cosine
lr_warmup_iters=$((QWENAIR_TRAIN_ITERS > 64 ? 64 : QWENAIR_TRAIN_ITERS - 1))
lr_decay_iters=${QWENAIR_TRAIN_ITERS}
seq_length=${QWENAIR_SEQ_LENGTH}
image_size=${QWENAIR_IMAGE_SIZE}
checkpoint_dir=${QWENAIR_CHECKPOINT_DIR}
EOF

echo "Launching canonical QwenAir training"
echo "  source manifest: ${manifest_path}"
echo "  training log: ${QWENAIR_RUN_DIR}/train-${SLURM_JOB_ID}.log"
echo "  tensorboard: ${QWENAIR_TENSORBOARD_DIR}"
echo "  rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"

srun_command=(
    srun
    --nodes=4
    --ntasks=32
    --ntasks-per-node=8
    --gpus-per-task=1
    --gpu-bind=none
    --kill-on-bad-exit=1
    --wait=60
    --export=ALL
)

if [[ -n "${QWENAIR_CONTAINER_IMAGE:-}" ]]; then
    require_env QWENAIR_CONTAINER_MOUNTS
    srun_command+=(
        --container-image="${QWENAIR_CONTAINER_IMAGE}"
        --container-mounts="${QWENAIR_CONTAINER_MOUNTS}"
        --no-container-mount-home
    )
fi

launcher_path="${QWENAIR_BRIDGE_ROOT}/examples/models/qwenair/slurm_target_multimodal.sh"
train_log="${QWENAIR_RUN_DIR}/train-${SLURM_JOB_ID}.log"
"${srun_command[@]}" bash "${launcher_path}" --worker 2>&1 | tee "${train_log}"
