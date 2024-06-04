requireNamespace("anndata", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)
library(tidyr, warn.conflicts = FALSE)
library(purrr, warn.conflicts = FALSE)
library(edgeR)

## VIASH START
par <- list(
  input = "resources/neurips-2023-data/pseudobulk.h5ad",
  output = "resources/neurips-2023-data/pseudobulk_cleaned.h5ad"
)
## VIASH END

# Load data
input <- anndata::read_h5ad(par$input)

cat("Filtering variables\n")
# Transform function similar to limma's requirements in the Python script
limma_trafo <- function(value) {
  gsub("[^[:alnum:]_]", "_", value)
}

# Unique cell types for processing
genes_to_keep <- lapply(
  unique(input$obs$cell_type),
  function(cell_type) {
    input_ct <- input[input$obs$cell_type == cell_type, ]

    # Prepare count data for edgeR analysis
    counts <- Matrix::t(input_ct$X)

    d <- DGEList(counts)
    design <- model.matrix(~ 0 + sm_name + plate_name, data = input_ct$obs %>% mutate_all(limma_trafo))
    keep <- filterByExpr(d, design)
    
    rownames(d)[keep]
  }
)

# Calculate the intersection of genes across all cell types
gene_filt <- Reduce(intersect, genes_to_keep)

# filter genes
output <- input[, gene_filt]

cat("Output:\n")
print(output)

cat("Writing output to ", par$output, "\n")
zzz <- output$write_h5ad(par$output, compression = "gzip")
