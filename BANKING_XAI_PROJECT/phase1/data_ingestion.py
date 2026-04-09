"""Data ingestion module: loads raw loan data for the Banking XAI project."""

from pathlib import Path
from typing import Optional
import pandas as pd
from phase1.config import RAW_DATA_FILE
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="data_ingestion.log")


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw loan data from CSV.

    Args:
        path: Optional custom path. Defaults to RAW_DATA_FILE from config.

    Returns:
        pd.DataFrame: loaded dataset.

    Raises:
        FileNotFoundError: if the CSV does not exist.
    """
    csv_path = Path(path or RAW_DATA_FILE)
    if not csv_path.exists():
        logger.error(f"Data file not found at {csv_path}. Please place your loan_data.csv there.")
        raise FileNotFoundError(f"Data file not found at {csv_path}")

    logger.info(f"Loading data from {csv_path}")

    # Treat 'NA', 'NaN', empty strings as missing
    df = pd.read_csv(
        csv_path,
        na_values=["NA", "NaN", ""],
        keep_default_na=True,
    )

    logger.info(f"Loaded dataset with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    return df
