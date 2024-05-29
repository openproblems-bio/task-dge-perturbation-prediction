#!/bin/bash

nextflow run . \
  -main-script target/nextflow/methods/lgc_ensemble/main.nf \
  -profile docker \
  -resume \
  --de_train_h5ad resources/neurips-2023-data/de_train.h5ad \
  --id_map resources/neurips-2023-data/id_map.csv \
  --layer sign_log10_pval \
  --epochs 2 \
  --kf_n_splits 2 \
  --schemes "initial;light" \
  --models "LSTM;GRU" \
  --publish_dir "output/lgc_ensemble"
