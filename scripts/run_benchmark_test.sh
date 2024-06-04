#!/bin/bash

export NXF_VER=23.04.2

resources_dir="resources"
publish_dir="output/test_run_benchmark"

cat > /tmp/params.yaml << HERE
param_list:
  - id: neurips-2023-data
    de_train_h5ad: "$resources_dir/neurips-2023-data/de_train.h5ad"
    de_test_h5ad: "$resources_dir/neurips-2023-data/de_test.h5ad"
    id_map: "$resources_dir/neurips-2023-data/id_map.csv"
    layer: clipped_sign_log10_pval
  - id: neurips-2023-kaggle
    de_train_h5ad: "$resources_dir/neurips-2023-kaggle/de_train.h5ad"
    de_test_h5ad: "$resources_dir/neurips-2023-kaggle/de_test.h5ad"
    id_map: "$resources_dir/neurips-2023-kaggle/id_map.csv"
    layer: sign_log10_pval
output_state: "state.yaml"
publish_dir: "$publish_dir"
HERE

nextflow run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  -params-file /tmp/params.yaml