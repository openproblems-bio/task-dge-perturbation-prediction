import pandas as pd
import anndata as ad
import numpy as np

## VIASH START
par = {
    "de_test_h5ad": "resources/neurips-2023-kaggle/de_test.h5ad",
    "prediction": "resources/neurips-2023-kaggle/prediction.h5ad",
    "output": "resources/neurips-2023-data/score.h5ad",
}
## VIASH END

print("Load data", flush=True)
de_test = ad.read_h5ad(par["de_test_h5ad"])
prediction = ad.read_h5ad(par["prediction"])

print("Select genes", flush=True)
genes = list(de_test.var_names)
de_test_X = de_test.layers[par["layer"]]
prediction_X = prediction.layers["prediction"]

print("Calculate mean pearson", flush=True)
mean_pearson = 0
mean_spearman = 0
for i in range(de_test_X.shape[0]):
    y_i = de_test_X[i,]
    y_hat_i = prediction_X[i,]

    # compute ranks
    r_i = y_i.argsort().argsort()
    r_hat_i = y_hat_i.argsort().argsort()

    pearson = np.corrcoef(y_i, y_hat_i)[0, 1]
    spearman = np.corrcoef(r_i, r_hat_i)[0, 1]

    mean_pearson += pearson
    mean_spearman += spearman

mean_pearson /= de_test_X.shape[0]
mean_spearman /= de_test_X.shape[0]

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