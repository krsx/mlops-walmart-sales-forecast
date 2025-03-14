import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import pickle
from pathlib import Path
from walmart_sales_forecasting.entity.config_entity import ModelEvaluationConfig
from walmart_sales_forecasting import logger
from walmart_sales_forecasting.utils.common import save_json


class ModelEvalation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = root_mean_squared_error(actual, pred)
        r2 = r2_score(actual, pred)
        mae = mean_absolute_error(actual, pred)
        mape = mean_absolute_percentage_error(actual, pred)

        metrics = {
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "mape": mape
        }

        return metrics

    def run_mlflow(self):
        target_col = self.config.target_column
        test_df = pd.read_csv(self.config.test_data_path)
        with open(self.config.pipeline_path, 'rb') as file:
            regressor_pipeline = pickle.load(file)
        with open(self.config.model_path, 'rb') as file:
            model_instance = pickle.load(file)

        test_df.drop(columns=['Date'], inplace=True)
        x_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.mlflow_project_name)

        with mlflow.start_run():
            pred = regressor_pipeline.predict(x_test)
            metrics = self.eval_metrics(y_test, pred)

            save_json(path=Path(self.config.evaluation_metrics_path),
                      data=metrics)
            mlflow.log_artifact(self.config.evaluation_metrics_path)
            logger.info(
                f"Evaluation metrics saved at {self.config.evaluation_metrics_path}")

            mlflow.log_params(self.config.model_params)
            mlflow.log_metric("rmse", metrics['rmse'])
            mlflow.log_metric("r2", metrics['r2'])
            mlflow.log_metric("mae", metrics['mae'])
            mlflow.log_metric("mape", metrics['mape'])
            logger.info("Metrics logged in MLflow")

            signature = mlflow.models.infer_signature(x_test, pred)
            mlflow.sklearn.log_model(
                regressor_pipeline,
                artifact_path=self.config.pipeline_path,
                signature=signature,
                registered_model_name="LGBMRegressorPipeline"
            )

            mlflow.set_tag("model_type", "LGBMRegressor")
            logger.info("Model logged in MLflow")
