
"""Logging utilities for the Banking XAI project."""

import logging
from pathlib import Path
from typing import Optional
from phase1.config import LOG_DIR

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        file_name = log_file or f"{name}.log"
        fh = logging.FileHandler(LOG_DIR / file_name)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
