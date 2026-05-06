#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-$(pwd)}
DATA_DIR=${2:-customer_service_finetune/data/customer_service/processed}
NUM_PROCESSES=${NUM_PROCESSES:-4}
NUM_UTTS_PER_PARQUET=${NUM_UTTS_PER_PARQUET:-1000}
PYTHON_BIN=${PYTHON_BIN:-/home/b24zll/software/anaconda3/envs/voice/bin/python}

cd "$ROOT_DIR"

for split in train dev; do
  src_dir="$DATA_DIR/$split"
  parquet_dir="$src_dir/parquet"
  mkdir -p "$parquet_dir"
  "$PYTHON_BIN" tools/make_parquet_list.py \
    --num_utts_per_parquet "$NUM_UTTS_PER_PARQUET" \
    --num_processes "$NUM_PROCESSES" \
    --src_dir "$src_dir" \
    --des_dir "$parquet_dir"
done

cp "$DATA_DIR/train/parquet/data.list" customer_service_finetune/data/customer_service/train.data.list
cp "$DATA_DIR/dev/parquet/data.list" customer_service_finetune/data/customer_service/dev.data.list
