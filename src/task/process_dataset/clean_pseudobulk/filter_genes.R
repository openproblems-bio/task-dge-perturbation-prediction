requireNamespace("anndata", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)
library(tidyr, warn.conflicts = FALSE)
library(purrr, warn.conflicts = FALSE)
library(edgeR)

# Accept arguments from the command line
args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]

# Load data
adata <- anndata::read_h5ad(input_path)


# Transform function similar to limma's requirements in the Python script
limma_trafo <- function(value) {
  gsub("[^[:alnum:]_]", "_", value)
}

# Filtering data for each cell type
filtered_gene_lists <- list()

# Unique cell types for processing
cell_types <- unique(adata$obs$cell_type)

for (cell_type in cell_types) {
  # Subset data by cell type
  cell_type_adata <- adata[adata$obs$cell_type == cell_type, ]

  # Prepare count data for edgeR analysis
  counts <- Matrix::t(cell_type_adata$X)

  d <- DGEList(counts)
  design <- model.matrix(~ 0 + sm_name + plate_name, data = cell_type_adata$obs %>% mutate_all(limma_trafo))
  keep <- filterByExpr(d, design)
  genes_to_keep <- rownames(d)[keep]

  # Store the filtered gene list
  filtered_gene_lists[[cell_type]] <- genes_to_keep
}

# Calculate the intersection of genes across all cell types
common_genes <- Reduce(intersect, filtered_gene_lists)

# Output common genes to standard output for Python to read
cat(common_genes, sep=",")