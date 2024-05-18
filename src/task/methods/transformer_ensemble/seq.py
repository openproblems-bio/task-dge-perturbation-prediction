import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)


def normalize_weights(weights):
    total_weight = sum(weights)
    return [weight / total_weight for weight in weights]


def calculate_weighted_sum(dataframes, weights):
    weighted_dfs = [df * weight for df, weight in zip(dataframes, weights)]
    return sum(weighted_dfs)


def convert_to_consistent_dtype(df):
    df = df.astype(float)
    df['id'] = df['id'].astype(int)
    return df


def set_id_as_index(df):
    df.set_index('id', inplace=True)
    return df


def save_submission_df(submission_df, file_path='weighted_submission.csv'):
    submission_df.to_csv(file_path, index=False)

def seq_main(
    par,
    model_dirs,
    weights
):
    dataframes = [load_data(f"{path}_output.csv") for path in model_dirs]

    normalized_weights = normalize_weights(weights[:-1]) + [weights[-1]]

    df = calculate_weighted_sum(dataframes, normalized_weights)

    df.to_parquet(par['output'])

