import os
import zipfile
from walmart_sales_forecasting import logger
from pathlib import Path
from walmart_sales_forecasting.entity.config_entity import DataIngestionConfig
from walmart_sales_forecasting.utils.common import get_file_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info("Please downlaod the appropriate data file!")
        else:
            logger.info(
                f"File already exists with size: {get_file_size(Path(self.config.local_data_file))}")

    def extract_zip(self):
        unzip_path = self.config.unzip_dir

        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
