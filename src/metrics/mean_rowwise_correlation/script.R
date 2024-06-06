library(anndata)
library(rlang)

## VIASH START
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

cat("Calculate metrics\n")
pearson <- proxyC::simil(de_test_X, prediction_X, method = "correlation", diag = TRUE)
mean_rowwise_pearson <- mean(ifelse(is.finite(pearson@x), pearson@x, 0))

out <- cor(t(de_test_X), t(prediction_X), method = "spearman")
spearman <- diag(out)
mean_rowwise_spearman <- mean(ifelse(is.finite(spearman), spearman, 0))

cosine <- proxyC::simil(de_test_X, prediction_X, method = "cosine", diag = TRUE)
mean_rowwise_cosine <- mean(ifelse(is.finite(cosine@x), cosine@x, 0))

cat("Create output\n")
output <- AnnData(
  shape = c(0L, 0L),
  uns = list(
    dataset_id = de_test$uns[["dataset_id"]],
    method_id = prediction$uns[["method_id"]],
    metric_ids = c("mean_rowwise_pearson", "mean_rowwise_spearman", "mean_rowwise_cosine"),
    metric_values = zapsmall(c(mean_rowwise_pearson, mean_rowwise_spearman, mean_rowwise_cosine), 10)
  )
)

cat("Write output\n")
output$write_h5ad(par$output, compression = "gzip")
