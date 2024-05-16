import anndata as ad
import subprocess
import os

## VIASH START
par = {
    "input": "resources/neurips-2023-data/pseudobulk.h5ad",
    "output": "resources/neurips-2023-data/pseudobulk_cleaned.h5ad"
}
meta = {"resources_dir": "src/task/process_dataset/clean_pseudobulk"}
## VIASH END

print(">> Load dataset", flush=True)
bulk_adata = ad.read_h5ad(par["input"])

print(">> Filter out samples with cell_count_by_well_celltype <= 10", flush=True)
bulk_adata = bulk_adata[bulk_adata.obs.cell_count_by_well_celltype > 10]

print(">> Filter molecules", flush=True)
# Alvocidib only T cells in only 2 donors, remove
bulk_adata = bulk_adata[bulk_adata.obs.sm_name != "Alvocidib"]
# BMS-387032 - one donor with only T cells, two other consistent, but only 2 cell types - leave the 2 cell types in, remove donor 2 with only T cells
bulk_adata = bulk_adata[~((bulk_adata.obs.sm_name == "BMS-387032") & (bulk_adata.obs.donor_id == "Donor 2"))]
# BMS-387032 remove myeloid cells and B cells
bulk_adata = bulk_adata[~((bulk_adata.obs.sm_name == "BMS-387032") & (bulk_adata.obs.cell_type.isin(["Myeloid cells", "B cells"])))]
# CGP 60474 has only T cells left, remove
bulk_adata = bulk_adata[bulk_adata.obs.sm_name != "CGP 60474"]
# Canertinib - the variation of Myeloid cell proportions is very large, skip Myeloid
bulk_adata = bulk_adata[~((bulk_adata.obs.sm_name == "Canertinib") & (bulk_adata.obs.cell_type == "Myeloid cells"))]
# Foretinib - large variation in Myeloid cell proportions (some in T cells), skip Myeloid.
bulk_adata = bulk_adata[~((bulk_adata.obs.sm_name == "Foretinib") & (bulk_adata.obs.cell_type == "Myeloid cells"))]
# Ganetespib (STA-9090) - donor 2 has no Myeloid and small NK cells proportions. Skip Myeloid, remove donor 2
bulk_adata = bulk_adata[~((bulk_adata.obs.sm_name == "Ganetespib (STA-9090)") & (bulk_adata.obs.donor_id == "Donor 2"))]
# IN1451 - donor 2 has no NK or B, remove Donor 2
bulk_adata = bulk_adata[~((bulk_adata.obs.sm_name == "IN1451") & (bulk_adata.obs.donor_id == "Donor 2"))]
# Navitoclax - donor 3 doesn't have B cells and has different T and Myeloid proportions, remove donor 3
bulk_adata = bulk_adata[~((bulk_adata.obs.sm_name == "Navitoclax") & (bulk_adata.obs.donor_id == "Donor 3"))]
# PF-04691502 remove Myeloid (only present in donor 3)
bulk_adata = bulk_adata[~((bulk_adata.obs.sm_name == "PF-04691502") & (bulk_adata.obs.cell_type == "Myeloid cells"))]
# Proscillaridin A;Proscillaridin-A remove Myeloid, since the variation is very high (4x)
bulk_adata = bulk_adata[~((bulk_adata.obs.sm_name == "Proscillaridin A;Proscillaridin-A") & (bulk_adata.obs.cell_type == "Myeloid cells"))]
# R428 - skip NK due to high variation (close to 3x)
bulk_adata = bulk_adata[~((bulk_adata.obs.sm_name == "R428") & (bulk_adata.obs.cell_type == "NK cells"))]
# UNII-BXU45ZH6LI - remove due to large variation across all cell types and missing cell types
bulk_adata = bulk_adata[bulk_adata.obs.sm_name != "UNII-BXU45ZH6LI"]

print(">> Save dataset for R script for gene filtering", flush=True)
bulk_adata.write_h5ad(par["input"], compression="gzip")

print(">> Filter out genes", flush=True)
command = ["Rscript", os.path.join(meta["resources_dir"], "filter_genes.R"), par["input"]]

result = subprocess.run(command, capture_output=True, text=True)

# Check for errors
if result.returncode != 0:
    print("Error running R script:", result.stderr, flush=True)
else:
    output = result.stdout
    filtered_genes = output.strip().split(",")

bulk_adata = bulk_adata[:, filtered_genes]

print(">> Save filtered bulk dataset", flush=True)
bulk_adata.write_h5ad(par["output"], compression="gzip")