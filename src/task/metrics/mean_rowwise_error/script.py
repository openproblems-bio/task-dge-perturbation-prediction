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

print("Clipping values", flush=True)
threshold_0001 = -np.log10(0.0001)
de_test_X_clipped_0001 = np.clip(de_test_X, -threshold_0001, threshold_0001)
prediction_clipped_0001 = np.clip(prediction.values, -threshold_0001, threshold_0001)

print("Calculate mean rowwise RMSE", flush=True)
mean_rowwise_rmse = 0
mean_rowwise_rmse_clipped_0001 = 0
mean_rowwise_mae = 0
mean_rowwise_mae_clipped_0001 = 0
for i in range(de_test_X.shape[0]):
    diff = de_test_X[i,] - prediction.iloc[i]
    diff_clipped_0001 = de_test_X_clipped_0001[i,] - prediction_clipped_0001[i]

    mean_rowwise_rmse += np.sqrt((diff**2).mean())
    mean_rowwise_rmse_clipped_0001 += np.sqrt((diff_clipped_0001 ** 2).mean())
    mean_rowwise_mae += np.abs(diff).mean()
    mean_rowwise_mae_clipped_0001 += np.abs(diff_clipped_0001).mean()

mean_rowwise_rmse /= de_test.shape[0]
mean_rowwise_rmse_clipped_0001 /= de_test.shape[0]
mean_rowwise_mae /= de_test.shape[0]
mean_rowwise_mae_clipped_0001 /= de_test.shape[0]

print("Create output", flush=True)
output = ad.AnnData(
    uns={
        "dataset_id": de_test.uns["dataset_id"],
        "method_id": par["method_id"],
        "metric_ids": ["mean_rowwise_rmse", "mean_rowwise_mae",
                          "mean_rowwise_rmse_clipped_0001", "mean_rowwise_mae_clipped_0001"],
        "metric_values": [mean_rowwise_rmse, mean_rowwise_mae,
                          mean_rowwise_rmse_clipped_0001, mean_rowwise_mae_clipped_0001]
    }
)

print("Write output", flush=True)
output.write_h5ad(par["output"], compression="gzip")