requireNamespace("arrow", quietly = TRUE)


## VIASH START
par <- list(
  de_train = "resources/neurips-2023-data/de_train.parquet",
  de_test = "resources/neurips-2023-data/de_test.parquet",
  id_map = "resources/neurips-2023-data/id_map.csv",
  output = "resources/neurips-2023-data/output_identity.parquet"
)
## VIASH END

# read data
de_train <- arrow::read_parquet(par$de_train)
id_map <- arrow::read_csv_arrow(par$id_map)

# get gene names
gene_names <- setdiff(names(de_train), c("id", "cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control"))

# create output data structure
output <- data.frame(id = id_map$id)

# generate random data
for (gene_name in gene_names) {
  output[[gene_name]] <- sample(de_train[[gene_name]], size = nrow(output), replace = TRUE)
}

# store output
arrow::write_parquet(output, par$output)
