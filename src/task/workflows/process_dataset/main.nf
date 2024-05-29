workflow run_wf {
  take:
  input_ch

  main:
  output_ch = input_ch

    | filter_obs.run(
      fromState: [input: "sc_counts"],
      toState: [filtered_sc_counts: "output"]
    )

    | compute_pseudobulk.run(
      fromState: [input: "filtered_sc_counts"],
      toState: [pseudobulk: "output"]
    )

    | filter_vars.run(
      fromState: [input: "pseudobulk",],
      toState: [pseudobulk_filtered: "output"]
    )

    | add_uns_metadata.run(
      fromState: [
        input: "pseudobulk_filtered",
        dataset_id: "dataset_id",
        dataset_name: "dataset_name",
        dataset_summary: "dataset_summary",
        dataset_description: "dataset_description",
        dataset_url: "dataset_url",
        dataset_reference: "dataset_reference",
        dataset_organism: "dataset_organism"
      ],
      toState: [pseudobulk_filtered_with_uns: "output"]
    )

    | run_limma.run(
      key: "limma_train",
      fromState: { id, state ->
        [
          input: state.pseudobulk_filtered_with_uns,
          input_splits: ["train", "control", "public_test"],
          output_splits: ["train", "control", "public_test"]
        ]
      },
      toState: [de_train_h5ad: "output"]
    )

    | run_limma.run(
      key: "limma_test",
      fromState: { id, state ->
        [
          input: state.pseudobulk_filtered_with_uns,
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

    | setState([
      // "de_train",
      "de_train_h5ad",
      // "de_test",
      "de_test_h5ad",
      "id_map"
    ])

  emit:
  output_ch
}
