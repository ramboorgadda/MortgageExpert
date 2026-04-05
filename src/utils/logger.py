import logging
import os
from datetime import datetime
LOG_DIR="logs"
os.path.exists(LOG_DIR) or os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_filename)

def get_logger(name):
    """Returns a logger instance with the specified name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger