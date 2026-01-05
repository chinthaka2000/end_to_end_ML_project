import logging
import os
from datetime import datetime

# 1. Generate a log file name based on the current timestamp
# Example: "01_05_2026_19_30_00.log"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Define the path where logs will be stored (current_directory/logs)
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# 3. Create the directory.
# We split the path to ensure we create the folder, not the file as a folder.
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

LOG_FILE_PATH = logs_path

# 4. Configure the logging system
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")