library(anndata)

## VIASH START (unchanged)
par <- list(
  de_test_h5ad = "resources/neurips-2023-data/de_test.h5ad",
  de_test_layer = "clipped_sign_log10_pval",
  prediction = "resources/neurips-2023-data/prediction.h5ad",
  prediction_layer = "prediction",
  resolve_genes = "de_test",
  output = "output.h5ad"
)
## VIASH END

cat("Load data\n")
de_test <- read_h5ad(par$de_test_h5ad)
cat("de_test: "); print(de_test)
prediction <- read_h5ad(par$prediction)
cat("prediction: "); print(prediction)

cat("Resolve genes\n")
genes <-
  if (par$resolve_genes == "de_test") {
    de_test$var_names
  } else if (par$resolve_genes == "intersection") {
    intersect(de_test$var_names, prediction$var_names)
  }
de_test <- de_test[, genes]
prediction <- prediction[, genes]

de_test_X <- de_test$layers[[par$de_test_layer]]
prediction_X <- prediction$layers[[par$prediction_layer]]

if (any(is.na(de_test_X))) {
  stop("NA values in de_test_X")
}
if (any(is.na(prediction_X))) {
  warning("NA values in prediction_X")
  prediction_X[is.na(prediction_X)] <- 0
}

cat("Calculate mean rowwise RMSE\n")
rowwise_rmse <- sqrt(rowMeans((de_test_X - prediction_X)^2))
mean_rowwise_rmse <- mean(rowwise_rmse)

cat("Calculate mean rowwise MAE\n")
rowwise_mae <- rowMeans(abs(de_test_X - prediction_X))
mean_rowwise_mae <- mean(rowwise_mae)

cat("Create output\n")
output <- AnnData(
  shape = c(0L, 0L),
  uns = list(
    dataset_id = de_test$uns[["dataset_id"]],
    method_id = prediction$uns[["method_id"]],
    metric_ids = c(
      "mean_rowwise_rmse",
      "mean_rowwise_mae"
    ),
    metric_values = zapsmall(
      c(
        mean_rowwise_rmse,
        mean_rowwise_mae
      ),
      10
    )
  )
)

cat("Write output\n")
write_h5ad(output, par$output, compression = "gzip")
