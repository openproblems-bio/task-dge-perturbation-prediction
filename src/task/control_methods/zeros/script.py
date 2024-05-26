import pandas as pd
import anndata as ad

## VIASH START
par = {
  "de_train": "resources/neurips-2023-data/de_train.parquet",
  "de_test": "resources/neurips-2023-data/de_test.parquet",
  "id_map": "resources/neurips-2023-data/id_map.csv",
  "output": "resources/neurips-2023-data/output_baseline_zero.parquet",
}
## VIASH END

de_train_h5ad = ad.read_h5ad(par["de_train_h5ad"])
id_map = pd.read_csv(par["id_map"])
gene_names = list(de_train_h5ad.var_names)

# create new pandas dataframe with columns "id" from id_map and gene_names for the rest. "id" fill from id_map and the rest set to zero
output = pd.DataFrame(0, index=id_map["id"], columns=gene_names).reset_index()
output.to_parquet(par["output"])