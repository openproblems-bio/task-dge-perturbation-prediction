#!/bin/bash

echo "This script is not supposed to be run directly."
echo "Please run the script step-by-step."
exit 1

# sync resources
scripts/download_resources.sh

# create a new component
method_id="my_method"
method_lang="python" # change this to "r" if need be

viash run src/common/create_component/config.vsh.yaml -- \
  --language "$method_lang" \
  --name "$method_id"

# TODO: fill in required fields in src/task/methods/foo/config.vsh.yaml
# TODO: edit src/task/methods/foo/script.py/R

# test the component
viash test src/task/methods/$method_id/config.vsh.yaml

# rebuild the container (only if you change something to the docker platform)
# You can reduce the memory and cpu allotted to jobs in _viash.yaml by modifying .platforms[.type == "nextflow"].config.labels
viash run src/task/methods/$method_id/config.vsh.yaml -- \
  ---setup cachedbuild ---verbose

# run the method
viash run src/task/methods/$method_id/config.vsh.yaml -- \
  --de_train "resources/neurips-2023-kaggle/de_train.parquet" \
  --id_map "resources/neurips-2023-kaggle/id_map.csv" \
  --output "output/prediction.parquet"

# run evaluation metric
viash run src/task/metrics/mean_rowwise_rmse/config.vsh.yaml -- \
  --de_test "resources/neurips-2023-kaggle/de_test.parquet" \
  --prediction "output/prediction.parquet" \
  --output "output/score.h5ad"

# print score on kaggle test dataset
python -c 'import anndata; print(anndata.read_h5ad("output/score.h5ad").uns)'