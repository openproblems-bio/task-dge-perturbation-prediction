workflow run_wf {
  take:
  input_ch

  main:

  output_ch = input_ch

    | compute_pseudobulk.run(
      fromState: [input: "sc_counts"],
      toState: [pseudobulk: "output"]
    )

    | clean_pseudobulk.run(
      fromState: [
        input: "pseudobulk",
      ],
      toState: [pseudobulk_filtered: "output"]
    )

    | run_edger.run(
      key: "edger_train",
      fromState: { id, state ->
        [
          input: state.pseudobulk_filtered,
          input_splits: ["train", "control", "public_test"],
          output_splits: ["train", "control", "public_test"]
        ]
      },
      toState: [de_train_h5ad: "output"]
    )

    | run_edger.run(
      key: "edger_test",
      fromState: { id, state ->
        [
          input: state.pseudobulk_filtered,
          input_splits: ["train", "control", "public_test", "private_test"],
          output_splits: ["private_test"]
        ]
      },
      toState: [de_test_h5ad: "output"]
    )

    | convert_h5ad_to_parquet.run(
      fromState: [
        input_train: "de_train_h5ad",
        input_test: "de_test_h5ad"
      ],
      toState: [
        de_train: "output_train",
        de_test: "output_test",
        id_map: "output_id_map"
      ]
    )

    | setState { id, state ->
      def dataset_info = [
        dataset_id: state.dataset_id,
        dataset_name: state.dataset_name,
        dataset_summary: state.dataset_summary,
        dataset_description: state.dataset_description,
        dataset_url: state.dataset_url,
        dataset_reference: state.dataset_reference,
        dataset_organism: state.dataset_organism,
      ]
      def dataset_info_yaml_blob = toYamlBlob(dataset_info)
      def dataset_info_file = tempFile("dataset_info.yaml")
      dataset_info_file.write(dataset_info_yaml_blob)

      [
        de_train: state.de_train,
        de_train_h5ad: state.de_train_h5ad,
        de_test: state.de_test,
        de_test_h5ad: state.de_test_h5ad,
        id_map: state.id_map,
        dataset_info: dataset_info_file
      ]
    }

  emit:
  output_ch
}
