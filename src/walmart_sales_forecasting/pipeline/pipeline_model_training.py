from walmart_sales_forecasting.config.configuration import ConfigurationManager
from walmart_sales_forecasting.components.model_training import ModelTraining
from walmart_sales_forecasting import logger

STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def train_model(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(model_training_config)
        model_training.train_model()


def main():
    try:
        logger.info(f">>>>> Starting stage: {STAGE_NAME} <<<<<")
        obj = ModelTrainingPipeline()
        obj.train_model()
        logger.info(f">>>>> Completed stage: {STAGE_NAME} <<<<<")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME}")
        raise e


if __name__ == "__main__":
    main()
