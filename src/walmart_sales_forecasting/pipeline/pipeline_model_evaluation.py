from walmart_sales_forecasting.config.configuration import ConfigurationManager
from walmart_sales_forecasting.components.model_evaluation import ModelEvalation
from walmart_sales_forecasting import logger

STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def evaluate_model(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvalation(config=model_evaluation_config)
        model_evaluation.run_mlflow()


def main():
    try:
        logger.info(f">>>>> Starting stage: {STAGE_NAME} <<<<<")
        obj = ModelEvaluationPipeline()
        obj.evaluate_model()
        logger.info(f">>>>> Completed stage: {STAGE_NAME} <<<<<")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME}")
        raise e


if __name__ == "__main__":
    main()
