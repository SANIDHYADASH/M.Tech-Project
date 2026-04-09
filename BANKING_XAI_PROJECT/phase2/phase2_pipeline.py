from phase2.sequence_builder import build_sequences
from phase2.train_lstm import train_lstm
from utils.logging_utils import get_logger
from phase1.preprocessing import basic_cleaning


logger = get_logger(__name__, log_file="phase2_pipeline.log")


def run_phase2(df, features, target):
    logger.info("==== Starting Phase-2 Banking XAI Pipeline ====")

    # Clean and encode target before building sequences
    df = basic_cleaning(df)

    # Recompute feature list after cleaning
    features = [col for col in df.columns if col != target]
    
    logger.info("Building sequential dataset")
    X, y = build_sequences(df, features, target)

    logger.info(f"Sequence dataset created with shape X={X.shape}, y={y.shape}")

    logger.info("Training LSTM model")
    model, history = train_lstm(X, y)

    logger.info("LSTM model training completed")

    logger.info("==== Phase-2 Pipeline completed successfully ====")

    return model


if __name__ == "__main__":
    logger.info("Running standalone Phase-2 pipeline")

    # Temporary standalone test loading
    import pandas as pd
    from phase1.data_ingestion import load_raw_data
    from phase1.feature_engineering import add_domain_features
    from phase1.preprocessing import basic_cleaning

    df = load_raw_data()
    df = add_domain_features(df)
    df = basic_cleaning(df)

    feature_cols = [col for col in df.columns if col != "loan_status"]

    run_phase2(df, feature_cols, "loan_status")