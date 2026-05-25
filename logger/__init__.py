


import logging 
import os 
from datetime import datetime


log_dir = "log"

os.makedirs(log_dir,exist_ok=True)

log_file_path = os.path.join(log_dir,"file.log")

logging.basicConfig(
    level=logging.INFO,
    filename=log_file_path,
    format="[ %(asctime)s ] %(name)s - %(filename)s - %(levelname)s - %(message)s"
)

logger =logging.getLogger(__name__)

