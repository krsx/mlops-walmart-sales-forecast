import os
from walmart_sales_forecasting import logger
import pandas as pd
import math
from walmart_sales_forecasting.entity.config_entity import DataTransformationConfig
from pathlib import Path
import boto3


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def clean_features_table_basic(self):
        logger.info("Processing features.csv (stage 1/2 starting)...")
        df = pd.read_csv(self.config.data_dirs["features"])

        # fill NA with 0
        df[["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]] = df[[
            "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]].fillna(value=0, axis=0)

        # converts year and month to numerical values
        df["Year"] = df["Date"].map(lambda x: x[2:4])
        df["Month"] = df["Date"].map(lambda x: int(x[5:7]))

        # save processed features data
        df.to_csv(os.path.join(self.config.root_dir,
                  "features_processed.csv"), index=False)
        logger.info(
            "Processing features.csv (stage 1/2 completed). 'Markdown' columns imputed; Year and month column transformed into numerical types")

    def clean_features_table_cpi_unemp(self):
        logger.info("Processing features.csv (stage 2/2 starting)...")
        features_df = pd.read_csv(os.path.join(
            self.config.root_dir, "features_processed.csv"))

        # generate temporary_df to get CPI and Unemployment rate changes
        temp_df = features_df.copy()
        temp_df["next_CPI"] = temp_df["CPI"].shift(periods=-1)
        temp_df["next_unemployement"] = temp_df["Unemployment"].shift(
            periods=-1)
        temp_df["CPI_change"] = (
            (temp_df["next_CPI"] - temp_df["CPI"]) / temp_df["CPI"])
        temp_df["unemployement_change"] = (
            (temp_df["next_unemployement"] - temp_df["Unemployment"]) / temp_df["Unemployment"])

        # impute CPI and Unemployement Rate with same growth as that week of last year
        # generate unique store_week id for 2012
        temp_year_12 = temp_df.loc[temp_df["Year"] == 12].copy()
        store_week = []
        store_init = 1
        week_counter = 0

        for index in range(0, len(temp_year_12)):
            if temp_year_12.iloc[index]["Store"] == store_init:
                week_counter += 1
            else:
                store_init += 1
                week_counter = 1

            store = str(temp_year_12.iloc[index]["Store"])
            store_week.append(store + "_" + str(week_counter))

        temp_year_12["store_week"] = store_week

        # generate unique store_week id for 2013
        features_processed_13 = features_df.loc[features_df["Year"] == "13"].copy(
        )

        store_week = []
        store_init = 1
        week_counter = 0

        for index in range(0, len(features_processed_13)):
            if features_processed_13.iloc[index]["Store"] == store_init:
                week_counter += 1
            else:
                store_init += 1
                week_counter = 1

            store = str(features_processed_13.iloc[index]["Store"])
            store_week.append(store + "_" + str(week_counter))

        features_processed_13["store_week"] = store_week

        # join CPI and Unemployyment growth rates from 2012 per store per week
        # with corresponding store_week id in 2013
        joint_temp = pd.merge(
            left=features_processed_13,
            right=temp_year_12[[
                "Date",
                "store_week",
                "CPI_change",
                "unemployement_change"
            ]],
            on="store_week",
            how="inner"
        )

        # impute CPI and Unemployement rate with growth rates
        joint_temp["CPI"] = joint_temp["CPI"].fillna(0)
        joint_temp["Unemployment"] = joint_temp["Unemployment"].fillna(-100)
        for index, index2 in zip(range(0, len(joint_temp)), joint_temp.index):
            if joint_temp.iloc[index]["CPI"] == 0:
                joint_temp.at[index2, "CPI"] = joint_temp.iloc[index -
                                                               1]["CPI"] * (joint_temp.iloc[index-1]["CPI_change"] + 1)
            if joint_temp.iloc[index]["Unemployment"] == -100:
                joint_temp.at[index2, "Unemployment"] = joint_temp.iloc[index -
                                                                        1]["Unemployment"] * (joint_temp.iloc[index-1]["unemployement_change"] + 1)

        # clean the joined df
        joint_temp.drop(columns=["store_week", "Date_y",
                        "CPI_change", "unemployement_change"], inplace=True)
        joint_temp.rename(columns={"Date_x": "Date"}, inplace=True)

        # append imputed rows for 2013 to original df
        new_features_df = pd.concat(
            [features_df.loc[features_df["Year"] != "13"], joint_temp], axis=0)
        new_features_df.reset_index(inplace=True, drop=True)

        # save processed features data
        new_features_df.to_csv(os.path.join(
            self.config.root_dir, "features_processed.csv"), index=False)
        logger.info(
            "Processing features.csv (stage 2/2 completed). CPI and Unemployment Rates imputed for year 2013")

    def join_tables(self):
        train_df = pd.read_csv(self.config.data_dirs["train"])
        stores_df = pd.read_csv(self.config.data_dirs["stores"])
        features_df = pd.read_csv(os.path.join(
            self.config.root_dir, "features_processed.csv"))

        # join all tables and drop duplicate columns
        all_join_df = train_df.merge(stores_df, on="Store", how="inner")
        all_join_df = all_join_df.merge(
            features_df, on=["Store", "Date"], how="inner")
        all_join_df.drop(columns=["IsHoliday_y"], inplace=True)
        all_join_df.rename(columns={"IsHoliday_x": "IsHoliday"}, inplace=True)

        # save the joined data
        all_join_df.to_csv(os.path.join(self.config.root_dir,
                                        "processed_data.csv"), index=False)
        logger.info("Joining all tables completed.")

    def add_features(self):
        df = pd.read_csv(os.path.join(
            self.config.root_dir, "processed_data.csv"))
        df['Date'] = pd.to_datetime(df['Date'])

        # add sin and cos features for Month and Week
        df["sin_Month"] = df["Month"].apply(
            lambda x: math.sin((2 * math.pi * x) / 12))
        df["cos_Month"] = df["Month"].apply(
            lambda x: math.cos((2 * math.pi * x) / 12))

        df['Week'] = df['Date'].dt.isocalendar().week
        df["sin_Week"] = df["Week"].apply(
            lambda x: math.sin((2 * math.pi * x) / 52))
        df["cos_Week"] = df["Week"].apply(
            lambda x: math.cos((2 * math.pi * x) / 52))

        # grouped df
        grouped_df = df.groupby(by=["Store", "Dept"])

        # add lagged features for weekly sales
        # in previous week, 2 weeks and 4 weeks
        # enhance understanding of previous week sales
        df["Sales_Lag_1W"] = grouped_df["Weekly_Sales"].shift(periods=1)
        df["Sales_Lag_2W"] = grouped_df["Weekly_Sales"].shift(periods=2)
        df["Sales_Lag_4W"] = grouped_df["Weekly_Sales"].shift(periods=4)

        # add rolling statistics features
        # calculates mean and std of sales for previous 4 weeks
        # to better capture trends or volatility in sales
        df["Sales_Rolling_Mean_4W"] = grouped_df["Weekly_Sales"].transform(
            lambda x: x.rolling(window=4).mean())
        df["Sales_Rolling_Std_4W"] = grouped_df["Weekly_Sales"].transform(
            lambda x: x.rolling(window=4).std())

        # save the processed data
        df.to_csv(os.path.join(self.config.root_dir,
                               "features_processed.csv"), index=False)
        logger.info("Temporal, lagged and rolling statistic features added")

    def cat_encoding(self):
        df = pd.read_csv(os.path.join(
            self.config.root_dir, "features_processed.csv"))

        type_encoded = pd.get_dummies(df["Type"], dtype=int, prefix="Type")
        df = pd.concat([df, type_encoded], axis=1)
        df.drop(columns=["Type"], inplace=True)

        df.to_csv(os.path.join(self.config.root_dir,
                               "features_processed.csv"), index=False)
        logger.info("Categorical features encoded")

    def split_sim_data(self):
        df = pd.read_csv(os.path.join(
            self.config.root_dir, "features_processed.csv"))
        df_sim = df.loc[df["Date"] >= "2012-07-10"]
        df_train = df.drop(index=df_sim.index)

        df_train.to_csv(os.path.join(self.config.root_dir,
                        "use_for_train_data.csv"), index=False)
        df_sim.to_csv(os.path.join(self.config.root_dir,
                                   "use_for_sim_data.csv"), index=False)
        logger.info(
            f"Data split for training and simulation completed. Using {df_train.shape[0]} samples (till 2012-07-10) for building model. Using {df_sim.shape[0]} samples (from 2012-07-11) for simulation")

    def split_train_test(self):
        df = pd.read_csv(os.path.join(
            self.config.root_dir, "use_for_train_data.csv"))
        train_df = df.loc[df["Date"] < "2012-04-01"]
        test_df = df.drop(index=train_df.index)

        train_df.to_csv(os.path.join(self.config.root_dir,
                        "final_train_data.csv"), index=False)
        test_df.to_csv(os.path.join(self.config.root_dir,
                                    "final_test_data.csv"), index=False)
        logger.info(
            f"Further split of model building data for training ({train_df.shape[0]} samples) and testing ({test_df.shape[0]} samples) completed")

    def push_to_s3(self):
        s3 = boto3.client('s3')
        bucket_name = "mlops-walmart-sales-forecast"
        object_path = Path(
            "artifacts/data_transformation/final_train_data.csv")
        key = "train_data.csv"

        try:
            s3.upload_file(object_path, bucket_name, key)
            logger.info(
                f"final_train_data.csv uploaded successfully to s3 bucket '{bucket_name}' as '{key}'")
        except Exception as e:
            print(f"Error uploading file: {e}")
            raise e

        object_path = Path(
            "artifacts/data_transformation/final_test_data.csv")
        key = "test_data.csv"

        try:
            s3.upload_file(object_path, bucket_name, key)
            logger.info(
                f"final_test_data.csv uploaded successfully to s3 bucket '{bucket_name}' as '{key}'")
        except Exception as e:
            print(f"Error uploading file: {e}")
            raise e
