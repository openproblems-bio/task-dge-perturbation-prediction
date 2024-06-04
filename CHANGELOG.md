# task_perturbation_prediction 1.0.0

Initial release of the Perturbation Prediction task. Initial components:

* `src/process_dataset`: Compute the DGE data from the raw single-cell counts using Limma.
* `src/control_methods`: Baseline control methods: sample, ground_truth, zeros, mean_across_celltypes, mean_across_compounds, mean_outcome.
* `src/methods`: Perturbation prediction methods: jn_ap_op2, lgc_ensemble, nn_retraining_with_pseudolabels, pyboost, scape, transformer_ensemble.
* `src/metrics`: Evaluation metrics: mean_rowwise_error, mean_rowwise_correlation.


