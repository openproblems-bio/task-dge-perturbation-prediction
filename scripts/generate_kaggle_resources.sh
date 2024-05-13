#!/bin/bash

set -e

OUT=resources/neurips-2023-kaggle

[[ ! -d $OUT ]] && mkdir -p $OUT

aws s3 cp s3://openproblems-bio/public/neurips-2023-competition/2023-09-14_kaggle_upload/2023-09-12_de_by_cell_type_test.h5ad --no-sign-request $OUT/2023-09-12_de_by_cell_type_test.h5ad
aws s3 cp s3://openproblems-bio/public/neurips-2023-competition/2023-09-14_kaggle_upload/2023-09-12_de_by_cell_type_train.h5ad --no-sign-request $OUT/2023-09-12_de_by_cell_type_train.h5ad

# recompress h5ad files
python -c \
  "import anndata as ad; ad.read_h5ad('$OUT/2023-09-12_de_by_cell_type_test.h5ad').write_h5ad('$OUT/2023-09-12_de_by_cell_type_test.h5ad', compression='gzip')"
python -c \
  "import anndata as ad; ad.read_h5ad('$OUT/2023-09-12_de_by_cell_type_train.h5ad').write_h5ad('$OUT/2023-09-12_de_by_cell_type_train.h5ad', compression='gzip')"

viash run src/task/process_dataset/convert_kaggle_h5ad_to_parquet/config.vsh.yaml -- \
  --input_train "$OUT/2023-09-12_de_by_cell_type_train.h5ad" \
  --input_test "$OUT/2023-09-12_de_by_cell_type_test.h5ad" \
  --output_train "$OUT/de_train.parquet" \
  --output_train_h5ad "$OUT/de_train.h5ad" \
  --output_test "$OUT/de_test.parquet" \
  --output_test_h5ad "$OUT/de_test.h5ad" \
  --output_id_map "$OUT/id_map.csv"

echo ">> Run method"
viash run src/task/control_methods/sample/config.vsh.yaml -- \
  --de_train "$OUT/de_train.parquet" \
  --de_test "$OUT/de_test.parquet" \
  --id_map "$OUT/id_map.csv" \
  --output "$OUT/prediction.parquet"

echo ">> Run metric"
viash run src/task/metrics/mean_rowwise_error/config.vsh.yaml -- \
  --prediction "$OUT/prediction.parquet" \
  --de_test "$OUT/de_test.parquet" \
  --output "$OUT/score.h5ad"

echo ">> Manually create meta files"
cat > "$OUT/dataset_info.yaml" <<'EOF'
dataset_id: neurips-2023-kaggle
dataset_name: NeurIPS2023 scPerturb DGE (Kaggle)
dataset_summary: Differential gene expression sign(logFC) * -log10(p-value) values
  after 24 hours of treatment with 144 compounds in human PBMCs
dataset_description: 'For this competition, we designed and generated a novel single-cell
  perturbational dataset in human peripheral blood mononuclear cells (PBMCs). We selected
  144 compounds from the Library of Integrated Network-Based Cellular Signatures (LINCS)
  Connectivity Map dataset (PMID: 29195078) and measured single-cell gene expression
  profiles after 24 hours of treatment. The experiment was repeated in three healthy
  human donors, and the compounds were selected based on diverse transcriptional signatures
  observed in CD34+ hematopoietic stem cells (data not released). We performed this
  experiment in human PBMCs because the cells are commercially available with pre-obtained
  consent for public release and PBMCs are a primary, disease-relevant tissue that
  contains multiple mature cell types (including T-cells, B-cells, myeloid cells,
  and NK cells) with established markers for annotation of cell types. To supplement
  this dataset, we also measured cells from each donor at baseline with joint scRNA
  and single-cell chromatin accessibility measurements using the 10x Multiome assay.
  We hope that the addition of rich multi-omic data for each donor and cell type at
  baseline will help establish biological priors that explain the susceptibility of
  particular genes to exhibit perturbation responses in difference biological contexts.'
dataset_url: TBD
dataset_reference: TBD
dataset_organism: homo_sapiens
EOF

cat > "$OUT/state.yaml" <<'EOF'
id: neurips-2023-kaggle
de_train: !file de_train.parquet
de_test: !file de_test.parquet
de_train_h5ad: !file de_train.h5ad
de_test_h5ad: !file de_test.h5ad
id_map: !file id_map.csv
dataset_info: !file dataset_info.yaml
EOF


echo ">> Uploading results to S3"
aws s3 sync --profile op2 \
  --include "*" \
  --exclude "neurips-2023-raw/*" \
  --exclude "neurips-2023-public/*" \
  "resources" \
  "s3://openproblems-bio/public/neurips-2023-competition/workflow-resources/" \
  --delete --dryrun
