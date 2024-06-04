import anndata as ad

## VIASH START
par = {
  'input': 'resources/neurips-2023-raw/pseudobulk_cleaned.h5ad',
  'dataset_id': 'neurips-2023-data',
  'dataset_name': 'NeurIPS2023 scPerturb DGE',
  'dataset_url': 'TBD',
  'dataset_reference': 'TBD',
  'dataset_summary': 'Differential gene expression ...',
  'dataset_description': 'For this competition, we designed ...',
  'dataset_organism': 'homo_sapiens',
  'output': 'resources/neurips-2023-data/pseudobulk_uns.h5ad'
}
## VIASH END

print(">> Load dataset", flush=True)
input = ad.read_h5ad(par["input"])

for key in ["dataset_id", "dataset_name", "dataset_url", "dataset_reference",\
            "dataset_summary", "dataset_description", "dataset_organism"]:
    input.uns[key] = par[key]

print(">> Save filtered bulk dataset", flush=True)
input.write_h5ad(par["output"], compression="gzip")
