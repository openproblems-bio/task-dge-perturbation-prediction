#!/bin/bash

set -e

IN="resources"
OUT="output"

[[ ! -d "$OUT" ]] && mkdir -p "$OUT"

# run benchmark
# 'input_states' looks for state.yaml files corresponding to datasets
export NXF_VER=23.04.2

nextflow run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  --publish_dir "$OUT" \
  --output_state "state.yaml" \
  -entry auto \
  --input_states "$IN/**/state.yaml"