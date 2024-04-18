import pandas as pd
import anndata as ad

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

print("Calculate mean rowwise RMSE", flush=True)
mean_rowwise_rmse = 0
for i in de_test.index:
    mean_rowwise_rmse += ((de_test.iloc[i] - prediction.iloc[i])**2).mean()

mean_rowwise_rmse /= de_test.shape[0]

print("Create output", flush=True)
output = ad.AnnData(
    uns = {
        # this info is not stored in the parquet files
        "dataset_id": "unknown",
        "method_id": "unknown",
        "metric_ids": ["mean_rowwise_rmse"],
        "metric_values": [mean_rowwise_rmse]
    }
)

print("Write output", flush=True)
output.write_h5ad(par["output"], compression="gzip")