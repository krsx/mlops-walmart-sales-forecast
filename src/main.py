from walmart_sales_forecasting import logger
from walmart_sales_forecasting.pipeline.pipeline_data_ingestion import DataIngestionPipeline
from walmart_sales_forecasting.pipeline.pipeline_data_validation import DataValidationPipeline
from walmart_sales_forecasting.pipeline.pipeline_data_transformation import DataTransformationPipeline
from walmart_sales_forecasting.pipeline.pipeline_model_training import ModelTrainingPipeline
from walmart_sales_forecasting.pipeline.pipeline_model_evaluation import ModelEvaluationPipeline


def data_ingestion():
    STAGE_NAME = "Data Ingestion"

    try:
        logger.info(f">>>>> Starting stage: {STAGE_NAME} <<<<<")
        pipeline = DataIngestionPipeline()
        pipeline.ingest_data()
        logger.info(f">>>>> Completed stage: {STAGE_NAME} <<<<<")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME}")
        raise e


def data_validation():
    STAGE_NAME = "Data Validation"

    try:
        logger.info(f">>>>> Starting stage: {STAGE_NAME} <<<<<")
        pipeline = DataValidationPipeline()
        validation_status = pipeline.validate_data()
        if validation_status:
            logger.info("Data validation successful")
        else:
            logger.error("Data validation failed")
        logger.info(f">>>>> Completed stage: {STAGE_NAME} <<<<<")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME}")
        raise e


def data_transformation():
    STAGE_NAME = "Data Transformation"

    try:
        logger.info(f">>>>> Starting stage: {STAGE_NAME} <<<<<")
        pipeline = DataTransformationPipeline()
        pipeline.transform_data()
        logger.info(f">>>>> Completed stage: {STAGE_NAME} <<<<<")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME}")
        raise e


def model_training():
    STAGE_NAME = "Model Training"

    try:
        logger.info(f">>>>> Starting stage: {STAGE_NAME} <<<<<")
        pipeline = ModelTrainingPipeline()
        pipeline.train_model()
        logger.info(f">>>>> Completed stage: {STAGE_NAME} <<<<<")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME}")
        raise e


def model_evaluation():
    STAGE_NAME = "Model Evaluation"

    try:
        logger.info(f">>>>> Starting stage: {STAGE_NAME} <<<<<")
        pipeline = ModelEvaluationPipeline()
        pipeline.evaluate_model()
        logger.info(f">>>>> Completed stage: {STAGE_NAME} <<<<<")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME}")
        raise e


def main():
    data_ingestion()
    data_validation()
    data_transformation()
    model_training()
    model_evaluation()


if __name__ == "__main__":
    main()
