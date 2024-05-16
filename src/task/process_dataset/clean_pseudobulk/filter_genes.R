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

# Prepare count data for edgeR analysis
counts <- Matrix::t(adata$X)

d <- DGEList(counts)
design <- model.matrix(~ 0 + sm_cell_type + donor_id, data = adata$obs %>% mutate_all(limma_trafo))
keep <- filterByExpr(d, design)
filtered_genes <- rownames(d)[keep]

cat(filtered_genes, sep=",")
