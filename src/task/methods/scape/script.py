import sys, os, fastparquet, anndata, shutil, argparse
print(sys.executable)
print(os.getcwd())
import pandas as pd
import numpy as np
import tensorflow as tf
print(f"tf version:{tf.__version__}")
print(f"Num GPUs Available:{len(tf.config.list_physical_devices('GPU'))}")
print(tf.config.list_physical_devices('GPU'))
import tempfile

import scape
print(f"scape version:{scape.__version__}")

## VIASH START
par = dict(
	de_train = "resources/neurips-2023-data/de_train.parquet",
	de_train_h5ad = "resources/neurips-2023-data/de_train.h5ad",
	id_map = "resources/neurips-2023-data/id_map.csv",
	output = "output/neurips-2023-data/output_rf.parquet",
	output_model = None,
	# cell = "NK cells",
	cell = "lol",
	epochs = 2,
	epochs_enhanced = 2,
	n_genes = 10,
	n_genes_enhanced = 10,
	n_drugs = 5,
	min_n_top_drugs = 0,
)
meta = dict(
	temp_dir = "/tmp"
)
## VIASH END

print(f"par: {par}")

# if output_model is not provided, create a temporary directory
model_dir = par["output_model"] or tempfile.TemporaryDirectory(dir = meta["temp_dir"]).name


# remove temp dir on exit
if not par["output_model"]:
	import atexit
	atexit.register(lambda: shutil.rmtree(model_dir))

# load log pvals
df_de = scape.io.load_slogpvals(par['de_train']).drop(columns=["id", "split"], axis=1, errors="ignore")

# if held-out cell type is not in the data, select a random cell type
if par["cell"] not in df_de.index.get_level_values("cell_type").unique():
	print(f"Input cell type ({par['cell']}) not found in the data.")
	par["cell"] = np.random.choice(df_de.index.get_level_values("cell_type").unique())
	print(f"Randomly selecting a cell type from the data: {par['cell']}.")


# load logfc
adata = anndata.read_h5ad(par["de_train_h5ad"])

df_lfc = pd.concat(
	[
		pd.DataFrame(adata.layers["logFC"], index=adata.obs_names, columns=adata.var_names),
		adata.obs[['cell_type', 'sm_name']],
	],
	axis=1
).set_index(['cell_type', 'sm_name'])
del adata

# Make sure rows/columns are in the same order
df_lfc = df_lfc.loc[df_de.index, df_de.columns]

# We select only a subset of the genes for the model (top most variant genes)
top_genes = scape.util.select_top_variable([df_de], k=par["n_genes"])

drugs = df_de.loc[df_de.index.get_level_values("cell_type") == par["cell"]].index.get_level_values("sm_name").unique().tolist()

if par["n_drugs"]:
	drugs = drugs[:par["n_drugs"]]

df_id_map = pd.read_csv(par["id_map"])
df_sub_ix = df_id_map.set_index(["cell_type", "sm_name"])

# generate base predictions
base_predictions = []
for i, d in enumerate(drugs):
	print(i, d)
	scm = scape.model.create_default_model(par["n_genes"], df_de, df_lfc)
	result = scm.train(
		val_cells=[par["cell"]], 
		val_drugs=[d],
		input_columns=top_genes,
		epochs=par["epochs"],
		output_folder=f"{model_dir}/_models",
		config_file_name="config.pkl",
		model_file_name=f"drug{i}.keras",
		baselines=["zero", "slogpval_drug"],
	)
	# Collect prediction in the OOF data
	df_pred = scm.predict(df_sub_ix)

	# remove df_pred index to save memory
	df_pred.reset_index(drop=True, inplace=True)
	base_predictions.append(df_pred)

df_sub = pd.DataFrame(np.median(base_predictions, axis=0), index=df_sub_ix.index, columns=df_de.columns)

sub_drugs = df_sub_ix.index.get_level_values("sm_name").unique().tolist()

# This time, exclude control drugs for the calculation of the top genes, in order to
# introduce more variability in the model
top_genes = scape.util.select_top_variable([df_de], k=par["n_genes_enhanced"], exclude_controls=True)

df_drug_effects = pd.DataFrame(df_de.T.pow(2).mean().pow(0.5).groupby("sm_name").mean().sort_values(ascending=False), columns=["effect"])
df_drug_effects["effect_norm"] = (df_drug_effects["effect"] / df_drug_effects["effect"].sum())*100

top_sub_drugs = df_drug_effects.loc[sub_drugs].sort_values("effect", ascending=False).head(par["min_n_top_drugs"]).index.tolist()

top_all_drugs = df_drug_effects.head(par["min_n_top_drugs"]).index.tolist()
top_drugs = set(top_all_drugs) | set(top_sub_drugs)

if len(top_drugs) == 0:
	# df_focus is not computed, just return the original submission
	df_submission_data = df_sub_ix.join(df_sub).reset_index(drop=True)
	fastparquet.write(par['output'], df_submission_data)
	sys.exit(0)

df_de_c = df_de[df_de.index.get_level_values("sm_name").isin(top_drugs)]
df_de_c = df_de_c.loc[:, top_genes]

df_lfc_c = df_lfc.loc[df_de_c.index, df_de_c.columns]

enhanced_predictions = []
for i, d in enumerate(top_drugs):
		print(i, d)
		scm = scape.model.create_default_model(par["n_genes_enhanced"], df_de_c, df_lfc_c)
		result = scm.train(
				val_cells=[par["cell"]], 
				val_drugs=[d],
				input_columns=top_genes,
				epochs=par["epochs_enhanced"],
				output_folder=f"{model_dir}/_models",
				config_file_name="enhanced_config.pkl",
				model_file_name=f"enhanced_drug{i}.keras",
				baselines=["zero", "slogpval_drug"],
		)
		# Collect prediction in the OOF data
		df_pred = scm.predict(df_sub_ix)

		# remove df_pred index to save memory
		df_pred.reset_index(drop=True, inplace=True)
		enhanced_predictions.append(df_pred)

df_sub_enhanced = pd.DataFrame(np.median(enhanced_predictions, axis=0), index=df_sub_ix.index, columns=df_de_c.columns)

df_focus = df_sub.copy()
df_focus.update(df_sub_enhanced)

# write output
df_submission = 0.80 * df_focus + 0.20 * df_sub
df_submission_data = df_sub_ix.join(df_submission).reset_index(drop=True)
fastparquet.write(par['output'], df_submission_data)