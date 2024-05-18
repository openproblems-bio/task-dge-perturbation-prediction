workflow auto {
  findStates(params, meta.config)
    | meta.workflow.run(
      auto: [publish: "state"]
    )
}

workflow run_wf {
  take:
  input_ch

  main:

  // construct list of methods
  methods = [
    ground_truth,
    mean_outcome,
    mean_across_celltypes,
    mean_across_compounds,
    sample,
    zeros,
    random_forest,
    first_place,
    third_place,
    jn_ap_op2,
    scape
  ]

  // construct list of metrics
  metrics = [
    mean_rowwise_error,
    mean_cosine_sim
  ]

  /* **************************
   * PREPARE DATASET AND TASK *
   ************************** */
  dataset_ch = input_ch

    // store original id for later use
    | map{ id, state ->
      [id, state + ["_meta": [join_id: id]]]
    }

    // extract the dataset metadata
    | extract_metadata.run(
      key: "dataset_uns",
      fromState: [input: "de_train_h5ad"],
      toState: { id, output, state ->
        state + [
          dataset_info: readYaml(output.output).uns
        ]
      }
    )

  /***************************
   * RUN METHODS AND METRICS *
   ***************************/
  score_ch = dataset_ch

    // run all methods
    | runEach(
      components: methods,

      // only run the method if it is in the list of method_ids
      filter: { id, state, comp ->
        !state.method_ids || state.method_ids.contains(comp.config.functionality.name)
      },

      // define a new 'id' by appending the method name to the dataset id
      id: { id, state, comp ->
        id + "." + comp.config.functionality.name
      },

      // use 'fromState' to fetch the arguments the component requires from the overall state
      fromState: { id, state, comp ->
        def new_args = [
          de_train: state.de_train,
          de_train_h5ad: state.de_train_h5ad,
          id_map: state.id_map,
        ]
        if (comp.config.functionality.info.type == "control_method") {
          new_args.de_test = state.de_test
        }
        new_args
      },

      // use 'toState' to publish that component's outputs to the overall state
      toState: { id, output, state, comp ->
        state + [
          method_id: comp.config.functionality.name,
          method_output: output.output
        ]
      },
      
      auto: [
        publish: "state"
      ]
    )

    // run all metrics
    | runEach(
      components: metrics,
      id: { id, state, comp ->
        id + "." + comp.config.functionality.name
      },
      // use 'fromState' to fetch the arguments the component requires from the overall state
      fromState: [
        de_test_h5ad: "de_test_h5ad",
        method_id: "method_id",
        prediction: "method_output",
      ],
      // use 'toState' to publish that component's outputs to the overall state
      toState: { id, output, state, comp ->
        state + [
          metric_id: comp.config.functionality.name,
          metric_output: output.output
        ]
      }
    )

  /******************************
   * GENERATE OUTPUT YAML FILES *
   ******************************/

  // extract and combine the dataset metadata
  dataset_meta_ch = dataset_ch
    | joinStates { ids, states ->
      // combine the dataset info into one file
      def dataset_uns = states.collect{it.dataset_info}
      def dataset_uns_yaml_blob = toYamlBlob(dataset_uns)
      def dataset_uns_file = tempFile("dataset_uns.yaml")
      dataset_uns_file.write(dataset_uns_yaml_blob)

      ["output", [dataset_uns: dataset_uns_file]]
    }

  output_ch = score_ch

    // extract the scores
    | extract_metadata.run(
      key: "score_uns",
      fromState: [input: "metric_output"],
      toState: { id, output, state ->
        state + [
          score_uns: readYaml(output.output).uns
        ]
      }
    )

    | joinStates { ids, states ->
      // store the method configs in a file
      def method_configs = methods.collect{it.config}
      def method_configs_yaml_blob = toYamlBlob(method_configs)
      def method_configs_file = tempFile("method_configs.yaml")
      method_configs_file.write(method_configs_yaml_blob)

      // store the metric configs in a file
      def metric_configs = metrics.collect{it.config}
      def metric_configs_yaml_blob = toYamlBlob(metric_configs)
      def metric_configs_file = tempFile("metric_configs.yaml")
      metric_configs_file.write(metric_configs_yaml_blob)

      def task_info_file = meta.resources_dir.resolve("task_info.yaml")

      // store the scores in a file
      def score_uns = states.collect{state ->
        state.score_uns + [
          dataset_id: state.dataset_info.dataset_id,
          method_id: state.method_id
        ]
      }
      def score_uns_yaml_blob = toYamlBlob(score_uns)
      def score_uns_file = tempFile("score_uns.yaml")
      score_uns_file.write(score_uns_yaml_blob)

      def new_state = [
        method_configs: method_configs_file,
        metric_configs: metric_configs_file,
        task_info: task_info_file,
        scores: score_uns_file,
        _meta: states[0]._meta
      ]
      
      ["output", new_state]
    }

    // merge all of the output data 
    | mix(dataset_meta_ch)
    | joinStates{ ids, states ->
      def mergedStates = states.inject([:]) { acc, m -> acc + m }
      [ids[0], mergedStates]
    }

  emit:
  output_ch
}