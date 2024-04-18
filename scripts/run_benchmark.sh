#!/bin/bash

set -e

IN="resources"
OUT="output"

[[ ! -d "$OUT" ]] && mkdir -p "$OUT"

# run benchmark
export NXF_VER=23.04.2

nextflow run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  --publish_dir "$OUT" \
  --output_state "state.yaml" \
  --id neurips-2023-data \
  --dataset_info "$IN/neurips-2023-data/dataset_info.yaml" \
  --de_train "$IN/neurips-2023-data/de_train.parquet" \
  --de_test "$IN/neurips-2023-data/de_test.parquet" \
  --id_map "$IN/neurips-2023-data/id_map.csv"

  # Alternatively: could also replace everything starting from '--id' with:
  # -entry auto \
  # --input_states "$IN/**/state.yaml"