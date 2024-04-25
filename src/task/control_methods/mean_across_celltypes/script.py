import pandas as pd

## VIASH START
par = {
  "de_train": "resources/neurips-2023-data/de_train.parquet",
  "de_test": "resources/neurips-2023-data/de_test.parquet",
  "id_map": "resources/neurips-2023-data/id_map.csv",
  "output": "resources/neurips-2023-data/output_mean_celltypes.parquet",
}
## VIASH END

de_train = pd.read_parquet(par["de_train"])
id_map = pd.read_csv(par["id_map"])
gene_names = [col for col in de_train.columns if col not in {"cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control", "index"}]
mean_celltype = de_train.groupby("cell_type")[gene_names].mean()
mean_celltype = mean_celltype.loc[id_map.cell_type]
output = pd.DataFrame(mean_celltype.values, index=id_map["id"], columns=gene_names).reset_index()
output.to_parquet(par["output"])