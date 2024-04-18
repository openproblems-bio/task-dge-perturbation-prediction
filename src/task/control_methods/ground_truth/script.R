requireNamespace("arrow", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)


## VIASH START
par <- list(
  de_train = "resources/neurips-2023-data/de_train.parquet",
  de_test = "resources/neurips-2023-data/de_test.parquet",
  id_map = "resources/neurips-2023-data/id_map.csv",
  output = "resources/neurips-2023-data/output_identity.parquet"
)
## VIASH END

# read data
de_test <- arrow::read_parquet(par$de_test)

# remove unneeded columns
output <- de_test %>% select(-cell_type, -sm_name, -sm_lincs_id, -SMILES, -split, -control)

# store output
arrow::write_parquet(output, par$output)
