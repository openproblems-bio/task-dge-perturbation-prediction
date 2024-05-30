requireNamespace("anndata", quietly = TRUE)
requireNamespace("arrow", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)
library(tidyr, warn.conflicts = FALSE)
library(purrr, warn.conflicts = FALSE)

## VIASH START
par <- list(
  train_h5ad = "resources/neurips-2023-data/de_train.h5ad",
  test_h5ad = "resources/neurips-2023-data/de_test.h5ad",
  output_train_h5ad = "output/de_train_filtered_*.h5ad",
  output_test_h5ad = "output/de_test_filtered_*.h5ad",
  num_replicates = 10,
  obs_fraction = .95,
  var_fraction = .95
)
## VIASH END

# Load data
train_h5ad <- anndata::read_h5ad(par$train_h5ad)
test_h5ad <- anndata::read_h5ad(par$test_h5ad)

for (i in seq_len(par$num_replicates)) {
  cat("Generating replicate", i, "\n", sep = "")

  # sample indices
  obs_ix <- sample.int(
    nrow(train_h5ad),
    round(nrow(train_h5ad) * par$obs_fraction, 0),
    replace = FALSE
  )
  var_ix <- sample.int(
    ncol(train_h5ad),
    round(ncol(train_h5ad) * par$var_fraction, 0),
    replace = FALSE
  )

  # subset h5ad
  output_train_h5ad <- train_h5ad[obs_ix, var_ix]
  output_test_h5ad <- test_h5ad[, var_ix]

  original_dataset_id <- output_train_h5ad$uns[["dataset_id"]]
  dataset_id <- paste0(original_dataset_id, "_bootstrap", i)
  output_train_h5ad$uns[["dataset_id"]] <- dataset_id
  output_test_h5ad$uns[["dataset_id"]] <- dataset_id
  output_train_h5ad$uns[["original_dataset_id"]] <- original_dataset_id
  output_test_h5ad$uns[["original_dataset_id"]] <- original_dataset_id

  # write output
  output_train_h5ad_path <- gsub("\\*", i, par$output_train_h5ad)
  output_test_h5ad_path <- gsub("\\*", i, par$output_test_h5ad)

  zzz <- output_train_h5ad$write_h5ad(output_train_h5ad_path, compression = "gzip")
  zzz <- output_test_h5ad$write_h5ad(output_test_h5ad_path, compression = "gzip")
}
