#!/usr/bin/env bash

set -x

NODE=$1
JOB_NAME=$2
CONFIG=$3
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

NCCL_IB_DISABLE=1 \
NCCL_DEBUG=VERSION \
NCCL_P2P_DISABLE=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -w ${NODE} -p A100 \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --kill-on-bad-exit=1 \
    python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
