#!/bin/bash

set -e

OUT=resources/neurips-2023-kaggle

[[ ! -d $OUT ]] && mkdir -p $OUT

if [[ ! -f "$OUT/2023-09-12_de_by_cell_type_test.h5ad" ]]; then
  echo ">> Downloading data"
  aws s3 cp s3://openproblems-bio/public/neurips-2023-competition/2023-09-14_kaggle_upload/2023-09-12_de_by_cell_type_test.h5ad --no-sign-request $OUT/2023-09-12_de_by_cell_type_test.h5ad
  aws s3 cp s3://openproblems-bio/public/neurips-2023-competition/2023-09-14_kaggle_upload/2023-09-12_de_by_cell_type_train.h5ad --no-sign-request $OUT/2023-09-12_de_by_cell_type_train.h5ad

  # recompress h5ad files
  python -c \
    "import anndata as ad; ad.read_h5ad('$OUT/2023-09-12_de_by_cell_type_test.h5ad').write_h5ad('$OUT/2023-09-12_de_by_cell_type_test.h5ad', compression='gzip')"
  python -c \
    "import anndata as ad; ad.read_h5ad('$OUT/2023-09-12_de_by_cell_type_train.h5ad').write_h5ad('$OUT/2023-09-12_de_by_cell_type_train.h5ad', compression='gzip')"
fi

viash run src/task/process_dataset/convert_kaggle_h5ad_to_parquet/config.vsh.yaml -- \
  --input_train "$OUT/2023-09-12_de_by_cell_type_train.h5ad" \
  --input_test "$OUT/2023-09-12_de_by_cell_type_test.h5ad" \
  --input_single_cell_h5ad "resources/neurips-2023-raw/sc_counts.h5ad" \
  --output_train_h5ad "$OUT/de_train.h5ad" \
  --output_test_h5ad "$OUT/de_test.h5ad" \
  --output_id_map "$OUT/id_map.csv" \
  --dataset_id neurips-2023-kaggle \
  --dataset_name "NeurIPS2023 scPerturb DGE (Kaggle)" \
  --dataset_summary 'Original Kaggle dataset' \
  --dataset_description 'Original Kaggle dataset' \
  --dataset_url TBD \
  --dataset_reference TBD \
  --dataset_organism homo_sapiens

echo ">> Run method"
viash run src/task/control_methods/mean_across_compounds/config.vsh.yaml -- \
  --de_train_h5ad "$OUT/de_train.h5ad" \
  --de_test_h5ad "$OUT/de_test.h5ad" \
  --id_map "$OUT/id_map.csv" \
  --output "$OUT/prediction.h5ad"

echo ">> Run metric"
viash run src/task/metrics/mean_rowwise_error/config.vsh.yaml -- \
  --prediction "$OUT/prediction.h5ad" \
  --de_test_h5ad "$OUT/de_test.h5ad" \
  --output "$OUT/score.h5ad"

cat > "$OUT/state.yaml" <<'EOF'
id: neurips-2023-kaggle
de_train_h5ad: !file de_train.h5ad
de_test_h5ad: !file de_test.h5ad
id_map: !file id_map.csv
EOF

echo ">> Uploading results to S3"
aws s3 sync --profile op2 \
  --include "*" \
  --exclude "neurips-2023-raw/*" \
  --exclude "neurips-2023-public/*" \
  "resources" \
  "s3://openproblems-bio/public/neurips-2023-competition/workflow-resources/" \
  --delete --dryrun
