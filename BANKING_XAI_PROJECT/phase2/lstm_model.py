# lstm_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from phase2.config import DROPOUT


def build_lstm_model(input_shape):
    """
    Smaller and more stable LSTM model for
    loan delinquency sequence prediction.
    """

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(
        LSTM(
            64,
            return_sequences=True
        )
    )

    model.add(Dropout(DROPOUT))

    model.add(
        LSTM(
            32
        )
    )

    model.add(Dropout(0.2))

    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["AUC", "Recall", "Precision"]
    )

    model.summary()

    return model