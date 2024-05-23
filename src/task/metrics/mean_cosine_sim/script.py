import pandas as pd
import anndata as ad
import numpy as np

## VIASH START
par = {
    "de_test_h5ad": "resources/neurips-2023-kaggle/de_test.h5ad",
    "prediction": "resources/neurips-2023-kaggle/output_mean_compounds.parquet",
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

print("Calculate mean cosine similarity", flush=True)
mean_cosine_similarity = 0
mean_cosine_similarity_clipped_0001 = 0
for i in range(de_test_X.shape[0]):
    y_i = de_test_X[i,]
    y_hat_i = prediction.iloc[i]
    y_i_clipped_0001 = de_test_X_clipped_0001[i,]
    y_hat_i_clipped_0001 = prediction_clipped_0001[i]

    dot_product = np.dot(y_i, y_hat_i)
    dot_product_clipped_0001 = np.dot(y_i_clipped_0001, y_hat_i_clipped_0001)

    norm_y_i = np.linalg.norm(y_i)
    norm_y_i_clipped_0001 = np.linalg.norm(y_i_clipped_0001)
    norm_y_hat_i = np.linalg.norm(y_hat_i)
    norm_y_hat_i_clipped_0001 = np.linalg.norm(y_hat_i_clipped_0001)

    cosine_similarity = dot_product / (norm_y_i * norm_y_hat_i)
    cosine_similarity_clipped_0001 = dot_product_clipped_0001 / (norm_y_i_clipped_0001 * norm_y_hat_i_clipped_0001)

    mean_cosine_similarity += cosine_similarity
    mean_cosine_similarity_clipped_0001 += cosine_similarity_clipped_0001

mean_cosine_similarity /= de_test_X.shape[0]
mean_cosine_similarity_clipped_0001 /= de_test_X.shape[0]

print("Create output", flush=True)
output = ad.AnnData(
    uns={
        "dataset_id": de_test.uns["dataset_id"],
        "method_id": par["method_id"],
        "metric_ids": ["mean_cosine_sim", "mean_cosine_sim_clipped_0001"],
        "metric_values": [mean_cosine_similarity, mean_cosine_similarity_clipped_0001]
    }
)

print("Write output", flush=True)
output.write_h5ad(par["output"], compression="gzip")