from walmart_sales_forecasting import logger
from walmart_sales_forecasting.entity.config_entity import DataValidationConfig
import pandas as pd


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_data(self) -> bool:
        try:
            validation_status = None
            full_status = True

            with open(self.config.status_file, 'w') as status_file:
                status_file.write("Initializing validation tests")

            for key, value in self.config.data_dirs.items():
                df = pd.read_csv(value)
                df_cols = list(df.columns)
                df_status = True

                schema_cols = self.config.schema[key]["columns"].keys()
                with open(self.config.status_file, 'a') as status_file:
                    status_file.write(f"\n\nValidating columns for {key}.csv")

                for col in df_cols:
                    if col not in schema_cols:
                        validation_status = False
                        df_status = False
                        full_status = False
                    else:
                        validation_status = True
                    with open(self.config.status_file, 'a') as status_file:
                        status_file.write(
                            f"\nStatus of {value}: {validation_status}")
                with open(self.config.status_file, 'a') as status_file:
                    status_file.write(
                        f"\n{key}.csv final validation status: {df_status}")

            return full_status

        except Exception as e:
            logger.error(f"Error validating data: {e}")
            full_status = False
            raise e
