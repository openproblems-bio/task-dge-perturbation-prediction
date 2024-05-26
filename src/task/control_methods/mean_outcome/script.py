import pandas as pd
import anndata as ad
import numpy as np
import sys

## VIASH START
par = {
  "de_train_h5ad": "resources/neurips-2023-data/de_train.h5ad",
  "de_test_h5ad": "resources/neurips-2023-data/de_test.h5ad",
  "layer": "sign_log10_pval",
  "id_map": "resources/neurips-2023-data/id_map.csv",
  "output": "resources/neurips-2023-data/output_mean.parquet",
}
## VIASH END

sys.path.append(meta["resources_dir"])
from anndata_to_dataframe import anndata_to_dataframe

de_train_h5ad = ad.read_h5ad(par["de_train_h5ad"])
id_map = pd.read_csv(par["id_map"])
gene_names = list(de_train_h5ad.var_names)
de_train = anndata_to_dataframe(de_train_h5ad, par["layer"])

mean_pred = de_train[gene_names].mean(axis=0)

# write output
output = ad.AnnData(
    layers={
        "prediction": np.vstack([mean_pred.values] * id_map.shape[0])
    },
    obs=pd.DataFrame(index=id_map["id"]),
    var=pd.DataFrame(index=gene_names),
    uns={
      "dataset_id": de_train_h5ad.uns["dataset_id"],
      "method_id": meta["functionality_name"]
    }
)
output.write_h5ad(par["output"], compression="gzip")