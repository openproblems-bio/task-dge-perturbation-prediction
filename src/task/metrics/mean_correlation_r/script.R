library(anndata)

## VIASH START
par <- list(
  de_test_h5ad = "resources/neurips-2023-data/de_test.h5ad",
  de_test_layer = "sign_log10_pval",
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

cat("Calculate metrics\n")
out <- cor(t(de_test_X), t(prediction_X), method = "pearson")
pearson <- diag(out)
mean_pearson <- mean(ifelse(is.na(pearson), 0, pearson))

out2 <- cor(t(de_test_X), t(prediction_X), method = "spearman")
spearman <- diag(out2)
mean_spearman <- mean(ifelse(is.na(spearman), 0, spearman))

cat("Create output\n")
output <- AnnData(
  shape = c(0L, 0L),
  uns = list(
    dataset_id = de_test$uns[["dataset_id"]],
    method_id = prediction$uns[["method_id"]],
    metric_ids = c("mean_pearson_r", "mean_spearman_r"),
    metric_values = c(mean_pearson, mean_spearman)
  )
)

cat("Write output\n")
output$write_h5ad(par$output, compression = "gzip")
