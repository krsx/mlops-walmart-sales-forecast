from walmart_sales_forecasting.config.configuration import ConfigurationManager
from walmart_sales_forecasting.components.data_validation import DataValidation
from walmart_sales_forecasting import logger

STAGE_NAME = "Data Validation"


class DataValidationPipeline:
    def __init__(self):
        pass

    def validate_data(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(data_validation_config)
        validation_status = data_validation.validate_data()
        return validation_status


def main():
    try:
        logger.info(f">>>>> Starting stage: {STAGE_NAME} <<<<<")
        obj = DataValidationPipeline()
        validation_status = obj.validate_data()
        if validation_status:
            logger.info("Data validation successful")
        else:
            logger.error("Data validation failed")
        logger.info(f">>>>> Completed stage: {STAGE_NAME} <<<<<")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME}")
        raise e


if __name__ == "__main__":
    main()
