import anndata as ad
import numpy as np

## VIASH START
par = {
    "de_test_h5ad": "resources/neurips-2023-data/de_test.h5ad",
    "de_test_layer": "sign_log10_pval",
    "prediction": "resources/neurips-2023-data/prediction.h5ad",
    "prediction_layer": "prediction",
    "resolve_genes": "de_test",
    "output": "output.h5ad"
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

print("Calculate metrics", flush=True)
mean_pearson = np.mean(
    [np.corrcoef(de_test_X[i,], prediction_X[i,])[0, 1] for i in range(de_test_X.shape[0])]
)

# compute ranks
de_test_r = np.argsort(de_test_X, axis=1).argsort(axis=1)
prediction_r = np.argsort(prediction_X, axis=1).argsort(axis=1)
mean_spearman = np.mean(
    [np.corrcoef(de_test_r[i,], prediction_r[i,])[0, 1] for i in range(de_test_X.shape[0])]
)

print("Create output", flush=True)
output = ad.AnnData(
    uns={
        "dataset_id": de_test.uns["dataset_id"],
        "method_id": prediction.uns["method_id"],
        "metric_ids": ["mean_pearson", "mean_spearman"],
        "metric_values": [mean_pearson, mean_spearman]
    }
)

print("Write output", flush=True)
output.write_h5ad(par["output"], compression="gzip")