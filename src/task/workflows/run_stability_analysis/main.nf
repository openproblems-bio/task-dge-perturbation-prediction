
// helper workflow for starting a workflow based on lists of yaml files
workflow auto {
  findStates(params, meta.config)
    | meta.workflow.run(
      auto: [publish: "state"]
    )
}

// benchmarking workflow
workflow run_wf {
  take:
  input_ch

  main:
  output_ch = input_ch

    // flatten bootstraps
    | flatMap { id, state -> 
      return (1..state.bootstrap_num_replicates).collect{ idx ->
        [
          "${id}-bootstrap${idx}",
          state + [replicate: idx]
        ]
      }
    }

    | bootstrap_sc_counts.run(
      fromState: [
        input: "sc_counts",
        obs_fraction: "bootstrap_obs_fraction",
        var_fraction: "bootstrap_var_fraction"
      ],
      toState: [
        sc_counts: "output"
      ]
    )

    | process_dataset.run(
      fromState: {id, state ->
        [
          sc_counts: state.sc_counts,
          dataset_id: id,
          dataset_name: "/",
          dataset_url: "/",
          dataset_reference: "/",
          dataset_summary: "/",
          dataset_description: "/",
          dataset_organism: "/"
        ]
      },
      toState: [
        de_test_h5ad: "de_test_h5ad",
        de_train_h5ad: "de_train_h5ad",
        id_map: "id_map"
      ]
    )

    | run_benchmark.run(
      fromState: [
        de_train_h5ad: "de_train_h5ad",
        de_test_h5ad: "de_test_h5ad",
        id_map: "id_map",
        method_ids: "method_ids",
        metric_ids: "metric_ids",
        layer: "layer"
      ],
      toState: [
        scores: "scores"
      ]
    )

    | setState(["scores"])

  emit:
  output_ch
}
