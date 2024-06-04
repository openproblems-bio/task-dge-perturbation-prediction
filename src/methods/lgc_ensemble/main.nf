
// benchmarking workflow
workflow run_wf {
  take:
  input_ch

  main:
  prepare_ch = input_ch

    | lgc_ensemble_prepare.run(
      fromState: [
        "de_train_h5ad",
        "id_map",
        "layer",
        "epochs",
        "kf_n_splits",
        "models",
        "schemes"
      ],
      toState: ["train_data_aug_dir"]
    )

  model_ch = prepare_ch
    | flatMap{ id, state ->
      // crossing between models, schemes and kf_n_splits
      def cross = state.models.collectMany{ model ->
        state.schemes.collectMany{ scheme ->
          (0..(state.kf_n_splits-1)).collect{ kf_n_splits ->
            
            def new_id = "${id}.${model}_${scheme}_fold${kf_n_splits}"
            def new_state = [
              "orig_id": id,
              "train_data_aug_dir": state.train_data_aug_dir,
              "model": model,
              "scheme": scheme,
              "fold": kf_n_splits
            ]

            [new_id, new_state]
          }
        }
      }
    }
    | lgc_ensemble_train.run(
      fromState: [
        "train_data_aug_dir",
        "model",
        "scheme",
        "fold"
      ],
      toState: ["model_file", "log_file"]
    )
    // group back on id
    | map{ old_id, state ->
      [state.orig_id, state.model_file]
    }
    | groupTuple()

  
  output_ch = prepare_ch.join(model_ch)
    | map{ id, state, model_files ->
      [id, state + ["model_files": model_files]]
    }
    | lgc_ensemble_predict.run(
      fromState: [
        "train_data_aug_dir",
        "id_map",
        "model_files",
      ],
      toState: ["output"]
    )
    | setState(["output"])

  emit:
  output_ch
}
