requireNamespace("arrow", quietly = TRUE)
requireNamespace("ranger", quietly = TRUE)
requireNamespace("pbapply", quietly = TRUE)

## VIASH START
par <- list(
  de_train = "resources/neurips-2023-data/de_train.parquet",
  id_map = "resources/neurips-2023-data/id_map.csv",
  output = "resources/neurips-2023-data/output_rf.parquet"
)
## VIASH END

# read data
de_train <- arrow::read_parquet(par$de_train)
id_map <- read.csv(par$id_map)

# determine relevant columns
x_vars <- c("sm_name", "cell_type")
gene_names <- setdiff(colnames(de_train), c("cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control", "index"))

# make predictions
x <- de_train[, x_vars, drop = FALSE]
xpred <- id_map[, x_vars, drop = FALSE]

preds <- pbapply::pblapply(
  gene_names,
  cl = parallel::detectCores(),
  function(gene_name) {
    rf <- ranger::ranger(
      x = x,
      y = de_train[[gene_name]],
      num.threads = 1L,
      num.trees = 100L
    )

    predict(rf, xpred)$predictions
  }
)

output <- data.frame(
  c(
    list(id = id_map$id),
    setNames(preds, gene_names)
  ),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

# store output
arrow::write_parquet(output, par$output)
