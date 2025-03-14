from walmart_sales_forecasting.config.configuration import ConfigurationManager
from walmart_sales_forecasting.components.data_ingestion import DataIngestion
from walmart_sales_forecasting import logger

STAGE_NAME = "Data Ingestion"


class DataIngestionPipeline:
    def __init__(self):
        pass

    def ingest_data(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_zip()


def main():
    try:
        logger.info(f">>>>> Starting stage: {STAGE_NAME} <<<<<")
        obj = DataIngestionPipeline()
        obj.ingest_data()
        logger.info(f">>>>> Completed stage: {STAGE_NAME} <<<<<")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME}")
        raise e


if __name__ == "__main__":
    main()
