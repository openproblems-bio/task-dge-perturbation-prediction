library(anndata)

## VIASH START (unchanged)
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

cat("Clipping values\n")
threshold_0001 <- -log10(0.0001)
de_test_X_clipped_0001 <- pmax(pmin(de_test_X, threshold_0001), -threshold_0001)
prediction_clipped_0001 <- pmax(pmin(prediction_X, threshold_0001), -threshold_0001)

cat("Calculate mean cosine similarity\n")
cosine <- proxyC::simil(de_test_X, prediction_X, method = "cosine", diag = TRUE)
mean_cosine_similarity <- mean(cosine@x)

cosine_clipped <- proxyC::simil(de_test_X_clipped_0001, prediction_clipped_0001, method = "cosine", diag = TRUE)
mean_cosine_similarity_clipped_0001 <- mean(cosine_clipped@x)

cat("Create output\n")
output <- AnnData(
  shape = c(0L, 0L),
  uns = list(
    dataset_id = de_test$uns[["dataset_id"]],
    method_id = prediction$uns[["method_id"]],
    metric_ids = c("mean_cosine_sim_r", "mean_cosine_sim_clipped_0001_r"),
    metric_values = c(mean_cosine_similarity, mean_cosine_similarity_clipped_0001)
  )
)

cat("Write output\n")
write_h5ad(output, par$output, compression = "gzip")
