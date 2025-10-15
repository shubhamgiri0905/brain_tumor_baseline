import logging
from datetime import datetime
import os

def setup_logger():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Log file name with timestamp
    log_filename = datetime.now().strftime("logs/training_%Y%m%d_%H%M%S.log")

    # Configure logger
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    return logging.getLogger(__name__)
