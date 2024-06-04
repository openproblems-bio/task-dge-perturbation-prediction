#!/bin/bash

RUN_ID="run_$(date +%Y-%m-%d_%H-%M-%S)"
resources_dir="s3://openproblems-bio/public/neurips-2023-competition/workflow-resources"
publish_dir="s3://openproblems-data/resources/dge_perturbation_prediction/results/${RUN_ID}"

cat > /tmp/params.yaml << HERE
param_list:
  - id: neurips-2023-data
    de_train_h5ad: "$resources_dir/neurips-2023-data/de_train.h5ad"
    de_test_h5ad: "$resources_dir/neurips-2023-data/de_test.h5ad"
    id_map: "$resources_dir/neurips-2023-data/id_map.csv"
    layer: clipped_sign_log10_pval
  # - id: neurips-2023-kaggle
  #   de_train_h5ad: "$resources_dir/neurips-2023-kaggle/de_train.h5ad"
  #   de_test_h5ad: "$resources_dir/neurips-2023-kaggle/de_test.h5ad"
  #   id_map: "$resources_dir/neurips-2023-kaggle/id_map.csv"
  #   layer: sign_log10_pval
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
  --config src/common/nextflow_helpers/labels_tw.config
