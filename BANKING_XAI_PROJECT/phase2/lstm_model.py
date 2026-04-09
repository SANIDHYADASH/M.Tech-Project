# lstm_model.py

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from phase2.config import LSTM_UNITS, DROPOUT


def build_lstm_model(input_shape):
    """
    Build LSTM model for loan delinquency prediction.
    """

    model = Sequential()

    model.add(
        LSTM(
            LSTM_UNITS,
            return_sequences=True,
            input_shape=input_shape
        )
    )

    model.add(Dropout(DROPOUT))

    model.add(LSTM(64))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["AUC", "Recall", "Precision"]
    )

    model.summary()

    return model