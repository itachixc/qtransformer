#!/usr/bin/env bash
CONFIG=$1
CUDA_VISIBLE_DEVICES=$2

python tools/train.py ${CONFIG_FILE} [ARGS]
