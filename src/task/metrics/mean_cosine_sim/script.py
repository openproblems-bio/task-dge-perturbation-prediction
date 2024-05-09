import pandas as pd
import anndata as ad
import numpy as np

## VIASH START
par = {
    "de_test": "resources/neurips-2023-data/de_test.parquet",
    "prediction": "resources/neurips-2023-data/output_rf.parquet",
    "output": "resources/neurips-2023-data/score.h5ad",
}
## VIASH END

print("Load data", flush=True)
de_test = pd.read_parquet(par["de_test"]).set_index('id')
prediction = pd.read_parquet(par["prediction"]).set_index('id')

print("Select genes", flush=True)
genes = list(set(de_test.columns) - set(["cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control"]))
de_test = de_test.loc[:, genes]
prediction = prediction[genes]

print("Calculate mean cosine similarity", flush=True)
mean_cosine_similarity = 0
for i in de_test.index:
    y_i = de_test.iloc[i]
    y_hat_i = prediction.iloc[i]

    dot_product = np.dot(y_i, y_hat_i)

    norm_y_i = np.linalg.norm(y_i)
    norm_y_hat_i = np.linalg.norm(y_hat_i)

    cosine_similarity = dot_product / (norm_y_i * norm_y_hat_i)

    mean_cosine_similarity += cosine_similarity

mean_cosine_similarity /= de_test.shape[0]

print("Create output", flush=True)
output = ad.AnnData(
    uns = {
        # this info is not stored in the parquet files
        "dataset_id": "unknown",
        "method_id": "unknown",
        "metric_ids": ["mean_cosine_sim"],
        "metric_values": [mean_cosine_similarity]
    }
)

print("Write output", flush=True)
output.write_h5ad(par["output"], compression="gzip")