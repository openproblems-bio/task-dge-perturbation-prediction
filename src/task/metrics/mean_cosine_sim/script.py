import pandas as pd
import anndata as ad
import numpy as np

## VIASH START
par = {
    "de_test_h5ad": "resources/neurips-2023-data/de_test.h5ad",
    "prediction": "resources/neurips-2023-data/prediction.parquet",
    "method_id": "foo",
    "output": "resources/neurips-2023-data/score.h5ad",
}
## VIASH END

print("Load data", flush=True)
de_test = ad.read_h5ad(par["de_test_h5ad"])
prediction = pd.read_parquet(par["prediction"]).set_index('id')

print("Select genes", flush=True)
genes = list(de_test.var_names)
de_test_X = de_test.layers["sign_log10_pval"]
prediction = prediction[genes]

print("Calculate mean cosine similarity", flush=True)
mean_cosine_similarity = 0
for i in range(de_test_X.shape[0]):
    y_i = de_test_X[i,]
    y_hat_i = prediction.iloc[i]

    dot_product = np.dot(y_i, y_hat_i)

    norm_y_i = np.linalg.norm(y_i)
    norm_y_hat_i = np.linalg.norm(y_hat_i)

    cosine_similarity = dot_product / (norm_y_i * norm_y_hat_i)

    mean_cosine_similarity += cosine_similarity

mean_cosine_similarity /= de_test_X.shape[0]

print("Create output", flush=True)
output = ad.AnnData(
    uns={
        "dataset_id": de_test.uns["dataset_id"],
        "method_id": par["method_id"],
        "metric_ids": ["mean_cosine_sim"],
        "metric_values": [mean_cosine_similarity]
    }
)

print("Write output", flush=True)
output.write_h5ad(par["output"], compression="gzip")