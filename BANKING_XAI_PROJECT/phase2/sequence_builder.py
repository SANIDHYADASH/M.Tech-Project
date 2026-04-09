# sequence_builder.py

import numpy as np
from phase2.config import SEQ_LENGTH


def build_sequences(df, feature_cols, target_col, seq_length=SEQ_LENGTH):
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

    sequences = []
    labels = []

    # Keep only numeric columns for LSTM input
    numeric_df = df[feature_cols].select_dtypes(include=["number"]).fillna(0)

    values = numeric_df.values
    targets = df[target_col].values

    for i in range(len(values) - seq_length):
        seq = values[i:i + seq_length]
        label = targets[i + seq_length]

        sequences.append(seq)
        labels.append(label)

    X_seq = np.array(sequences)
    y_seq = np.array(labels)

    print(f"Sequence data shape: {X_seq.shape}")
    print(f"Target shape: {y_seq.shape}")

    return X_seq, y_seq