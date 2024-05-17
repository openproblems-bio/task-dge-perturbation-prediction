requireNamespace("anndata", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)
library(tidyr, warn.conflicts = FALSE)
library(purrr, warn.conflicts = FALSE)
library(tibble, warn.conflicts = FALSE)
library(furrr)
library(future)

## VIASH START
par <- list(
  input = "resources/neurips-2023-data/pseudobulk.h5ad",
  de_sig_cutoff = 0.05,
  control_compound = "Dimethyl Sulfoxide",
  # for public data
  output = "resources/neurips-2023-data/de_train.h5ad",
  input_splits = c("train", "control", "public_test"),
  output_splits = c("train", "control", "public_test")
  # # for private data
  # output = "resources/neurips-2023-data/de_test.h5ad",
  # input_splits = c("train", "control", "public_test", "private_test"),
  # output_splits = c("private_test")
)
meta <- list(
  cpus = 10
)
## VIASH END

plan(multicore, workers = meta$cpus)

# load data
adata <- anndata::read_h5ad(par$input)

# select [cell_type, sm_name] pairs which will be used for DE analysis
new_obs <- adata$obs %>%
  select(sm_cell_type, cell_type, sm_name, sm_lincs_id, SMILES, split, control) %>%
  distinct() %>%
  filter(sm_name != par$control_compound)

if (!is.null(par$output_splits)) {
  new_obs <- new_obs %>%
    filter(split %in% par$output_splits)
}

# check which cell_types to run limma for
cell_types <- as.character(unique(new_obs$cell_type))

# helper function for transforming values to limma compatible values
limma_trafo <- function(value) {
  gsub("[^[:alnum:]]", "_", value)
}

obs_filt <- adata$obs$split %in% par$input_splits

start_time <- Sys.time()

d0 <- Matrix::t(adata[obs_filt, ]$X) %>%
    edgeR::DGEList() %>%
    edgeR::calcNormFactors()

design_matrix <- model.matrix(~ 0 + sm_cell_type + donor_id, adata[obs_filt, ]$obs %>% mutate_all(limma_trafo))

# Voom transformation and lmFit
v <- limma::voom(d0, design = design_matrix, plot = FALSE)
fit <- limma::lmFit(v, design_matrix)

# run limma DE for each cell type and compound
de_df <- future_map_dfr(
  seq_len(nrow(new_obs)),
  function(row_i) {
    cat("Computing DE contrasts (", row_i, "/", nrow(new_obs), ")\n", sep = "")
    sm_cell_type <- as.character(new_obs$sm_cell_type[[row_i]])
    cell_type <- as.character(new_obs$cell_type[[row_i]])

    control_name <- paste(par$control_compound, cell_type, sep = "_")
    # run contrast fit
    contrast_formula <- paste0(
      "sm_cell_type", limma_trafo(sm_cell_type),
      " - ",
      "sm_cell_type", limma_trafo(control_name)
    )
    contr <- limma::makeContrasts(
      contrasts = contrast_formula,
      levels = colnames(coef(fit))
    )

    limma::contrasts.fit(fit, contr) %>%
      limma::eBayes(robust=TRUE) %>%
      limma::topTable(n = Inf, sort = "none") %>%
      rownames_to_column("gene") %>%
      mutate(row_i = row_i)
  },
  .options = furrr_options(seed = TRUE)
)

end_time <- Sys.time()
full_duration <- end_time - start_time
full_duration_seconds <- as.numeric(full_duration, units = "secs")
full_minutes <- full_duration_seconds %/% 60
full_seconds <- full_duration_seconds %% 60
cat("Total limma runtime: ", full_minutes, " minutes, ", full_seconds, " seconds\n")

# transform data
de_df2 <- de_df %>%
  mutate(
    # convert gene names to factor
    gene = factor(gene),
    # readjust p-values for multiple testing
    adj.P.Value = p.adjust(P.Value, method = "BH"),
    # compute sign log10 p-values
    sign_log10_pval = sign(logFC) * -log10(ifelse(adj.P.Value == 0, .Machine$double.eps, P.Value)),
    is_de = P.Value < par$de_sig_cutoff,
    is_de_adj = adj.P.Val < par$de_sig_cutoff
  ) %>%
  as_tibble()

rownames(new_obs) <- paste0(new_obs$cell_type, ", ", new_obs$sm_name)
new_var <- data.frame(row.names = levels(de_df2$gene))

# create layers from de_df
layer_names <- c("is_de", "is_de_adj", "logFC", "P.Value", "adj.P.Value", "sign_log10_pval")
layers <- map(setNames(layer_names, layer_names), function(layer_name) {
  de_df2 %>%
    select(gene, row_i, !!layer_name) %>%
    arrange(row_i) %>%
    spread(gene, !!layer_name) %>%
    select(-row_i) %>%
    as.matrix()
})

# create anndata object
output <- anndata::AnnData(
  obs = new_obs,
  var = new_var,
  layers = setNames(layers, layer_names)
)

# write to file
zz <- output$write_h5ad(par$output, compression = "gzip")
