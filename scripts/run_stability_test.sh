#!/bin/bash

export NXF_VER=23.04.2

cat > /tmp/params.yaml <<'HERE'
id: neurips-2023-data
sc_counts: resources/neurips-2023-raw/sc_counts_reannotated_with_counts.h5ad
method_ids: ['ground_truth', 'sample', 'mean_across_celltypes', 'mean_across_compounds']
layer: t # test a different layer
publish_dir: "output/test_stability_analysis"
output_state: "state.yaml"
HERE

nextflow run . \
  -main-script target/nextflow/workflows/run_stability_analysis/main.nf \
  -profile docker \
  -resume \
  -params-file /tmp/params.yaml