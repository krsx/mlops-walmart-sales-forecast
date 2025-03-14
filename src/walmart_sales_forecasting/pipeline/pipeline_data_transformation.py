from pathlib import Path
from walmart_sales_forecasting.config.configuration import ConfigurationManager
from walmart_sales_forecasting.components.data_transformation import DataTransformation
from walmart_sales_forecasting import logger

STAGE_NAME = "Data Transformation"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def transform_data(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read()[-4:]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(
                    data_transformation_config)
                data_transformation.clean_features_table_basic()
                data_transformation.clean_features_table_cpi_unemp()
                data_transformation.join_tables()
                data_transformation.add_features()
                data_transformation.cat_encoding()
                data_transformation.split_sim_data()
                data_transformation.split_train_test()
                data_transformation.push_to_s3()
            else:
                logger.error("Data validation failed.")

        except Exception as e:
            logger.error(f"Error in stage: {STAGE_NAME}")
            raise e


def main():
    try:
        logger.info(f">>>>> Starting stage: {STAGE_NAME} <<<<<")
        obj = DataTransformationPipeline()
        obj.transform_data()
        logger.info(f">>>>> Completed stage: {STAGE_NAME} <<<<<")
    except Exception as e:
        logger.error(f"Error in stage: {STAGE_NAME}")
        raise e


if __name__ == "__main__":
    main()
