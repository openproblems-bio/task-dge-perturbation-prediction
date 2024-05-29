import anndata as ad
import numpy as np

## VIASH START
par = {
    "de_test_h5ad": "resources/neurips-2023-data/de_test.h5ad",
    "de_test_layer": "sign_log10_pval",
    "prediction": "resources/neurips-2023-data/prediction.h5ad",
    "prediction_layer": "prediction",
    "resolve_genes": "de_test",
    "output": "output.h5ad",
}
## VIASH END

print("Load data", flush=True)
de_test = ad.read_h5ad(par["de_test_h5ad"])
print(f"de_test: {de_test}")
prediction = ad.read_h5ad(par["prediction"])
print(f"prediction: {prediction}")

print("Resolve genes", flush=True)
if par["resolve_genes"] == "de_test":
    genes = list(de_test.var_names)
elif par["resolve_genes"] == "intersection":
    genes = list(set(de_test.var_names) & set(prediction.var_names))
de_test = de_test[:, genes]
prediction = prediction[:, genes]

# get data
de_test_X = de_test.layers[par["de_test_layer"]]
prediction_X = prediction.layers[par["prediction_layer"]]

print("Clipping values", flush=True)
threshold_0001 = -np.log10(0.0001)
de_test_X_clipped_0001 = np.clip(de_test_X, -threshold_0001, threshold_0001)
prediction_clipped_0001 = np.clip(prediction_X, -threshold_0001, threshold_0001)

print("Calculate mean cosine similarity", flush=True)
mean_cosine_similarity = np.mean(
    np.sum(de_test_X * prediction_X, axis=1) / (np.linalg.norm(de_test_X, axis=1) * np.linalg.norm(prediction_X, axis=1))
)
mean_cosine_similarity_clipped_0001 = np.mean(
    np.sum(de_test_X_clipped_0001 * prediction_clipped_0001, axis=1) / (np.linalg.norm(de_test_X_clipped_0001, axis=1) * np.linalg.norm(prediction_clipped_0001, axis=1))
)

print("Create output", flush=True)
output = ad.AnnData(
    uns={
        "dataset_id": de_test.uns["dataset_id"],
        "method_id": prediction.uns["method_id"],
        "metric_ids": ["mean_cosine_sim", "mean_cosine_sim_clipped_0001"],
        "metric_values": [mean_cosine_similarity, mean_cosine_similarity_clipped_0001]
    }
)

print("Write output", flush=True)
output.write_h5ad(par["output"], compression="gzip")