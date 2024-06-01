#!/bin/bash

RUN_ID="stability_$(date +%Y-%m-%d_%H-%M-%S)"
publish_dir="s3://openproblems-data/resources/dge_perturbation_prediction/results/${RUN_ID}"

cat > /tmp/params.yaml << HERE
id: neurips-2023-data
sc_counts: s3://openproblems-bio/public/neurips-2023-competition/sc_counts_reannotated_with_counts.h5ad
layer: clipped_sign_log10_pval
publish_dir: "$publish_dir"
HERE

tw launch https://github.com/openproblems-bio/task-dge-perturbation-prediction.git \
  --revision main_build \
  --pull-latest \
  --main-script target/nextflow/workflows/run_stability_analysis/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --config src/common/nextflow_helpers/labels_tw.config
