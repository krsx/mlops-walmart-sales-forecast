import os
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from walmart_sales_forecasting import logger


@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    try:
        with open(file_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {file_path} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        logger.error(f"Error reading yaml file: {file_path}")
        logger.error(e)
        raise e
    except Exception as e:
        logger.error(f"Error reading yaml file: {file_path}")
        logger.error(e)
        raise e


@ensure_annotations
def create_directories(filepath_list: list, verbose=True):
    for path in filepath_list:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Creating directory {path}")
        else:
            if verbose:
                logger.info(f"{path} already exists")


@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f)

    logger.info(f"Data saved to {path}")


@ensure_annotations
def load_json(file_path: Path) -> ConfigBox:
    try:
        with open(file_path) as f:
            content = json.load(f)

        logger.info(f"json file loaded succesfully from: {file_path}")
        return ConfigBox(content)
    except BoxValueError as e:
        logger.error(f"Error reading yaml file: {file_path}")
        logger.error(e)
        raise e
    except Exception as e:
        logger.error(f"Error loading json file from: {file_path}")
        logger.error(e)
        raise e


@ensure_annotations
def get_file_size(file_path: Path) -> int:
    size_in_kb = round(os.path.getsize(file_path)/1024)
    return size_in_kb
