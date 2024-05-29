
def anndata_to_dataframe(adata, layer_name="sign_log10_pval"):
  import pandas as pd

  metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'split', 'control']
  metadata = adata.obs[metadata_cols].copy()

  # turn all category columns to string
  for col in metadata.select_dtypes(include=["category"]).columns:
      metadata[col] = metadata[col].astype(str)

  data = pd.DataFrame(
    adata.layers[layer_name],
    columns=adata.var_names,
    index=adata.obs.index
  )

  return pd.concat([metadata, data], axis=1).reset_index(drop=True)
