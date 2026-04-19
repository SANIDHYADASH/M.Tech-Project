# sequence_builder.py

import numpy as np
from phase2.config import SEQ_LENGTH


def build_sequences(df, feature_cols, target_col=None, seq_length=SEQ_LENGTH):
    """
    Build sequential windows for LSTM training.

    Parameters:
    - df: input dataframe
    - feature_cols: list of feature columns
    - target_col: target column name
    - seq_length: number of rows in each sequence

    Returns:
    - X_seq: sequence array
    - y_seq: target array
    """

    # Sort by issue month if available
    if "issue_month" in df.columns:
        df = df.sort_values("issue_month")

    numeric_df = df[feature_cols].select_dtypes(include=["number"])
    missing_numeric = [col for col in feature_cols if col not in numeric_df.columns]
    if missing_numeric:
        raise ValueError(
            f"The following sequence features are missing or non-numeric: {missing_numeric}"
        )

    numeric_df = numeric_df.fillna(0)
    values = numeric_df.values

    sequences = []
    for i in range(len(values) - seq_length):
        sequences.append(values[i:i + seq_length])

    X_seq = np.array(sequences)
    print(f"Sequence data shape: {X_seq.shape}")

    if target_col is None:
        return X_seq, None

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    targets = df[target_col].values
    labels = []
    for i in range(len(values) - seq_length):
        labels.append(targets[i + seq_length])

    y_seq = np.array(labels)
    print(f"Target shape: {y_seq.shape}")

    return X_seq, y_seq