requireNamespace("anndata", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)

## VIASH START
par <- list(
  de_train_h5ad = "resources/neurips-2023-data/de_train.h5ad",
  de_test_h5ad = "resources/neurips-2023-data/de_test.h5ad",
  layer = "sign_log10_pval",
  id_map = "resources/neurips-2023-data/id_map.csv",
  output = "resources/neurips-2023-data/output_identity.h5ad"
)
## VIASH END

# read data
de_test_h5ad <- anndata::read_h5ad(par$de_test_h5ad)

# remove unneeded columns
output <- anndata::AnnData(
  layers = list(
    prediction = de_test_h5ad$layers[[par$layer]]
  ),
  obs = de_test_h5ad$obs[, c()],
  var = de_test_h5ad$var[, c()],
  uns = list(
    dataset_id = de_test_h5ad$uns$dataset_id,
    method_id = meta$functionality_name
  )
)

# write output
output$write_h5ad(par$output, compression = "gzip")