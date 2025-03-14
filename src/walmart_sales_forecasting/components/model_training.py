import os
import pandas as pd
import pickle
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from walmart_sales_forecasting import logger
from walmart_sales_forecasting.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def train_model(self):
        target_col = self.config.target_column

        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_data.drop(columns=['Date'], inplace=True)
        test_data.drop(columns=['Date'], inplace=True)

        # extract data based on column types
        categorical_cols = [
            c for c in train_data.columns if train_data[c].dtype in [object]]
        numerical_cols = [c for c in train_data.columns if train_data[c].dtype in [
            float, int] and c != target_col]
        cycle_num_cols = [c for c in train_data.columns if (
            "sin" in str(c)) or ("cos" in str(c))]

        x_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        x_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]

        # training pipeline orchestration
        # numerical columns are scaled using MinMaxScaler
        pipeline = make_pipeline(
            ColumnTransformer([
                ("num", MinMaxScaler(), [
                 c for c in numerical_cols if c not in cycle_num_cols])
            ]),
            LGBMRegressor(
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate
            )
        )
        regressor = pipeline.fit(x_test, y_train)
        model_instance = regressor.named_steps['lgbmregressor']

        # save the model instance
        pickle.dump(regressor, open(os.path.join(
            self.config.root_dir, self.config.pipeline_name), 'wb'))
        logger.info(
            f"Training Pipeline successfully saved at {self.config.root_dir}/{self.config.pipeline_name}")
        pickle.dump(model_instance, open(os.path.join(
            self.config.root_dir, self.config.model_instance_name), 'wb'))
        logger.info(
            f"Model Instance successfully saved at {self.config.root_dir}/{self.config.model_instance_name}")
