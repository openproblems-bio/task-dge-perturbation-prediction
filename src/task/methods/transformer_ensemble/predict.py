import torch
import torch.optim
import copy
import pandas as pd

from utils import load_transformer_model, prepare_augmented_data, load_transformer_model, prepare_augmented_data_mean_only

@torch.no_grad()
def predict_test(par, data, models, n_components_list, d_list, batch_size, device='cpu', outname='traineddata'):
    num_samples = len(data)
    de_train = pd.read_parquet(par["de_train"])
    id_map = pd.read_csv(par["id_map"])
    gene_names = [col for col in de_train.columns if col not in {"cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control", "index"}]

    for i, n_components in enumerate(n_components_list):
        for j, d_model in enumerate(d_list):
            combined_outputs = []
            label_reducer, scaler, transformer_model = models[f'{n_components},{d_model}']
            transformer_model.eval()
            for i in range(0, num_samples, batch_size):
                batch_unseen_data = data[i:i + batch_size]
                transformed_data = transformer_model(batch_unseen_data)
                if scaler:
                    transformed_data = torch.tensor(scaler.inverse_transform(
                        label_reducer.inverse_transform(transformed_data.cpu().detach().numpy()))).to(device)
                # print(transformed_data.shape)
                combined_outputs.append(transformed_data)

            # Stack the combined outputs
            combined_outputs = torch.vstack(combined_outputs)

            submission_df = pd.DataFrame(
                    combined_outputs.cpu().detach().numpy(),
                    index=id_map["id"],
                    columns=gene_names
                    ).reset_index()
            submission_df.to_csv(f"{outname}_output.csv")
            # only one d_model and n_component is run at a time
            return


def predict_main(
    par,
    n_components_list,
    model_dir,
    d_models_list=[128],
    batch_size=32,
    device='cpu',
    mean_std='mean_std',
    uncommon=False,
):
    data_file = par['de_train']
    id_map_file = par['id_map']

    # Prepare augmented data
    if mean_std == "mean_std":
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data(
            data_file=data_file,
            id_map_file=id_map_file,
            uncommon=uncommon
        )
    else:
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data_mean_only(
            data_file=data_file,
            id_map_file=id_map_file
        )
    unseen_data = torch.tensor(one_hot_test, dtype=torch.float32).to(device)  # Replace X_unseen with your new data
    transformer_models = {}
    for n_components in n_components_list:
        for d_model in d_models_list:
            label_reducer, scaler, transformer_model = load_transformer_model(
                n_components,
                input_features=one_hot_encode_features.shape[1],
                num_targets=targets.shape[1],
                d_model=d_model,
                models_folder=f'{model_dir}',
                device=device,
                mean_std=mean_std
            )
            transformer_model.eval()
            transformer_models[f'{n_components},{d_model}'] = (
                copy.deepcopy(label_reducer), copy.deepcopy(scaler), copy.deepcopy(transformer_model))
    predict_test(par, unseen_data, transformer_models, n_components_list, d_models_list, batch_size, device=device, outname = model_dir)

