#!/bin/bash

export NXF_VER=23.04.2

cat > /tmp/params.yaml << EOF
id: neurips-2023-data
de_train_h5ad: resources/neurips-2023-data/de_train.h5ad
de_test_h5ad: resources/neurips-2023-data/de_test.h5ad
id_map: resources/neurips-2023-data/id_map.csv
method_ids: ['ground_truth', 'sample', 'mean_across_celltypes', 'mean_across_compounds']
layer: t # test a different layer
publish_dir: "output/test_run_benchmark"
output_state: state.yaml
EOF

nextflow run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  -params-file /tmp/params.yaml