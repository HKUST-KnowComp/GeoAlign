#!/bin/sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

DIMS="10"
MODEL="lorentz"
GPU=-1
DSET_PATH="wordnet/mammal_closure"
DSET_TYPE="csv"
DATE_TIME=$(date +"%Y%m%d_%H%M%S")
CHECKPOINT="models/$DSET_PATH/$DATE_TIME"
[[ -d $CHECKPOINT ]] || mkdir -p $CHECKPOINT
CHECKPOINT="$CHECKPOINT/checkpoint.bin"
echo "Checkpoint: $CHECKPOINT"
while true; do
  case "$1" in
    -d | --dim ) DIMS=$2; shift; shift ;;
    -m | --model ) MODEL=$2; shift; shift ;;
    -g | --gpu ) GPU=$2; shift; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

USAGE="usage: ./train-mammals.sh -d <dim> -m <model>
  -d: dimensions to use
  -m: model to use (can be lorentz or poincare)
  -g: gpu to use (-1 no use)
  Example: ./train-mammals.sh -m lorentz -d 10
"

python3 embed.py \
       -dim "$DIMS" \
       -lr 0.3 \
       -epochs 300 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -manifold "$MODEL" \
       -dset wordnet/mammal_closure.csv \
       -checkpoint $CHECKPOINT \
       -batchsize 10 \
       -eval_each 300 \
       -fresh \
       -sparse \
       -train_threads 16 \
       -gpu "$GPU"
