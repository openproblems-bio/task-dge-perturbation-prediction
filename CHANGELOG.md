# task-dge-perturbation-prediction 0.1.0

Initial release of the DGE Perturbation Prediction task. Initial components:

* `src/task/process_dataset`: Compute the DGE data from the raw single-cell counts using Limma.
* `src/task/control_methods`: Baseline control methods: sample, ground_truth, zeros, mean_across_celltypes, mean_across_compounds, mean_outcome.
* `src/task/methods`: DGE perturbation prediction methods: random_forest.
* `src/task/metrics`: Evaluation metrics: mean_rowwise_rmse.


