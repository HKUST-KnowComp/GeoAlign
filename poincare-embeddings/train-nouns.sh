#!/bin/bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Script to reproduct results

DIMS="10"
MODEL="lorentz"
GPU="-1"

while true; do
  case "$1" in
    -d | --dim ) DIMS=$2; shift; shift ;;
    -m | --model ) MODEL=$2; shift; shift ;;
    -g | --gpu ) GPU=$2; shift; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

USAGE="usage: ./train-nouns.sh -d <dim> -m <model>
  -d: dimensions to use
  -m: model to use (can be lorentz or poincare)
  -g: gpu to use (-1 no use)
  Example: ./train-nouns.sh -m lorentz -d 10
"

case "$MODEL" in
  "lorentz" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "poincare" ) EXTRA_ARGS=("-lr" "1.0");;
  * ) echo "$USAGE"; exit 1;;
esac

python3 embed.py \
  -checkpoint nouns.bin \
  -dset wordnet/noun_closure.csv \
  -epochs 1 \
  -negs 50 \
  -burnin 20 \
  -dampening 0.75 \
  -ndproc 4 \
  -eval_each 1 \
  -fresh \
  -sparse \
  -burnin_multiplier 0.01 \
  -neg_multiplier 0.1 \
  -lr_type constant \
  -train_threads 16 \
  -dampening 1.0 \
  -batchsize 50 \
  -manifold "$MODEL" \
  -dim "$DIMS" \
  -gpu "$GPU" \
  "${EXTRA_ARGS[@]}"
