import os
from walmart_sales_forecasting.constants import *
from walmart_sales_forecasting.utils.common import read_yaml, create_directories
from walmart_sales_forecasting.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelTrainingConfig, ModelEvaluationConfig


class ConfigurationManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH, schema_file_path=SCHEMA_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir, config.unzip_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema

        data_file_dirs = {}
        for key, value in config.data_dirs.items():
            data_file_dirs[key] = value

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            schema=schema,
            data_dirs=data_file_dirs
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        data_file_dirs = {}
        for key, value in config.data_dirs.items():
            data_file_dirs[key] = value

        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_dirs=data_file_dirs
        )

        return data_transformation_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        schema = self.schema
        params = self.params.LGBMRegressor

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            pipeline_name=config.pipeline_name,
            model_instance_name=config.model_name,
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            random_state=params.random_state,
            n_jobs=params.n_jobs,
            target_column=schema.target_column.name
        )

        return model_training_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        schema = self.schema
        params = self.params.LGBMRegressor

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            pipeline_path=config.pipeline_path,
            test_data_path=config.test_data_path,
            evaluation_metrics_path=config.evaluation_metrics_path,
            model_params=params,
            target_column=schema.target_column.name,
            mlflow_project_name="walmart-sales-forecasting",
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )

        return model_evaluation_config
