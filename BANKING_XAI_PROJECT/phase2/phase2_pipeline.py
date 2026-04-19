from phase2.sequence_builder import build_sequences
from phase2.train_lstm import train_lstm
from phase2.config import SEQUENCE_FEATURES
from utils.logging_utils import get_logger
from phase1.preprocessing import basic_cleaning


logger = get_logger(__name__, log_file="phase2_pipeline.log")


def run_phase2(df, features, target):
    logger.info("==== Starting Phase-2 Banking XAI Pipeline ====")

    # Clean and encode target before building sequences
    df = basic_cleaning(df)

    # Use only strongest sequence features for better signal quality
    sequence_features = [
        col for col in SEQUENCE_FEATURES
        if col in df.columns
    ]

    logger.info(f"Using sequence features: {sequence_features}")

    logger.info("Building sequential dataset")

    X, y = build_sequences(
        df=df,
        feature_cols=sequence_features,
        target_col=target,
        seq_length=6
    )

    logger.info(f"Sequence dataset created with shape X={X.shape}, y={y.shape}")

    logger.info("Training LSTM model")
    model, history = train_lstm(X, y)

    logger.info("LSTM model training completed")

    logger.info("==== Phase-2 Pipeline completed successfully ====")

    return model


if __name__ == "__main__":
    logger.info("Running standalone Phase-2 pipeline")

    from phase1.data_ingestion import load_raw_data
    from phase1.feature_engineering import add_domain_features
    from phase1.preprocessing import basic_cleaning

    df = load_raw_data()
    df = add_domain_features(df)
    df = basic_cleaning(df)

    feature_cols = [col for col in df.columns if col != "loan_status"]

    run_phase2(df, feature_cols, "loan_status")