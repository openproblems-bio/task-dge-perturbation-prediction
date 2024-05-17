#!/bin/bash

RUN_ID="run_$(date +%Y-%m-%d_%H-%M-%S)"
publish_dir="s3://openproblems-data/resources/dge_perturbation_prediction/results/${RUN_ID}"

cat > /tmp/params.yaml << HERE
id: dge_perturbation_task
input_states: s3://openproblems-bio/public/neurips-2023-competition/workflow-resources/**/state.yaml
output_state: "state.yaml"
publish_dir: "$publish_dir"
HERE

tw launch https://github.com/openproblems-bio/task-dge-perturbation-prediction.git \
  --revision main_build \
  --pull-latest \
  --main-script target/nextflow/workflows/run_benchmark/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --entry-name auto \
  --config src/common/nextflow_helpers/labels_tw.config
