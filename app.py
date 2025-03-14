from walmart_sales_forecasting import logger
from walmart_sales_forecasting.utils.common import create_directories
import warnings
import base64
import io
import boto3
import time
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from pathlib import Path
import pickle
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

st.title("Walmart Sales Forecasting")

if st.button("Run Simulation"):
    st.write("Starting simulation...")

    plot_dates = []
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []

    create_directories(["artifacts/simulation"])
    create_directories(["artifacts/simulation/temp"])

    s3 = boto3.client("s3")
    bucket_name = "mlops-walmart-sales-forecast"
    train_data_key = "train_data.csv"
    test_data_key = "test_data.csv"
    sim_train_data_path = "artifacts/simulation/temp/orig_train_data.csv"
    sim_test_data_path = "artifacts/simulation/temp/orig_test_data.csv"
    full_data_key = "updated_data/full_data.csv"
    full_data_path = "artifacts/simulation/temp/full_data.csv"

    try:
        logger.info("Downloading test and train data from S3")
        s3.download_file(bucket_name, train_data_key, sim_train_data_path)
        s3.download_file(bucket_name, test_data_key, sim_test_data_path)

        train_data = pd.read_csv(sim_train_data_path)
        test_data = pd.read_csv(sim_test_data_path)

        logger.info("Test and Train data downloaded successfully")
    except Exception as e:
        logger.error("Error downloading data from S3")
        logger.error(e)
        raise e

    dates = np.sort(test_data["Date"].unique()).tolist()

    with open(Path('artifacts/model_training/lgbmr_regressor_pipeline.pkl'), 'rb') as file:
        model = pickle.load(file)

    r2_plot_placeholder = st.empty()
    rmse_plot_placeholder = st.empty()
    mae_plot_placeholder = st.empty()
    r2_plot_placeholder = st.empty()

    for date in dates:
        logger.info(f"Running prediction and data updates for {date} batch.")
        batch_data = test_data[test_data["Date"] == date].copy()
        train_data = pd.concat([train_data, batch_data], axis=0)
        train_data.to_csv(full_data_path, index=False)

        try:
            logger.info("Uploading updated full data to S3")
            s3.upload_file(full_data_path, bucket_name, full_data_key)
            logger.info(
                f"Full data updated and uploaded to S3 for {date} batch.")
        except Exception as e:
            logger.error("Error uploading full data to S3")
            logger.error(e)
            raise e
        logger.info(f"Full data updated and uploaded to S3 for {date} batch.")

        chunk_path = f"artifacts/simulation/temp/chunk_{date}.csv"
        batch_data.to_csv(chunk_path, index=False)
        chunk_key = f"updated_data/chunks/chunk_{date}.csv"

        try:
            logger.info("Uploading chunk data to S3")
            s3.upload_file(chunk_path, bucket_name, chunk_key)
            logger.info(f"Chunk data uploaded to S3 for {date} batch.")
        except Exception as e:
            logger.error("Error uploading chunk data to S3")
            logger.error(e)
            raise e

        targets = batch_data["Weekly_Sales"]
        batch_data.drop(columns=["Date", "Weekly_Sales"], inplace=True)

        try:
            logger.info(f"Running prediction for {date} batch")
            predictions = model.predict(batch_data)
            logger.info(f"Prediction completed {date} batch")
        except Exception as e:
            logger.error("Error in prediction")
            logger.error(e)
            raise e

        r2 = round(r2_score(targets, predictions), 4)
        rmse = round(root_mean_squared_error(targets, predictions), 4)
        mae = round(mean_absolute_error(targets, predictions), 4)
        mape = round(mean_absolute_percentage_error(targets, predictions), 4)
        chunk_metrics = {
            "Samples": len(batch_data),
            "r2_score": r2,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
        }

        try:
            logger.info(f"Uploading metrics to S3 for {date} batch")
            s3.put_object(Body=json.dumps(chunk_metrics),
                          Bucket=bucket_name, Key=f"metrics/{date}/metrics.json")
            logger.info(f"Metrics uploaded to S3 for {date} batch")
        except Exception as e:
            logger.error("Error uploading metrics to S3")
            logger.error(e)
            raise e
        logger.info(f"Prediction completed. Metrics logged for {date} batch")

        plot_dates.append(str(date)[5:])
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)

    metrics_df = pd.DataFrame({
        "Date": plot_dates,
        "r2": r2_scores,
        "rmse": rmse_scores,
        "mae": mae_scores,
        "mape": mape_scores
    })

    # Display the DataFrame as a table
    st.header("Aggregated Metrics by Date")
    st.dataframe(metrics_df)

    st.header("Metrics Trend Over Time")
    st.subheader("R2 Score")
    st.line_chart(metrics_df.set_index("Date")["r2"])

    st.subheader("RMSE")
    st.line_chart(metrics_df.set_index("Date")["rmse"])

    st.subheader("MAE")
    st.line_chart(metrics_df.set_index("Date")["mae"])

    st.subheader("MAPE")
    st.line_chart(metrics_df.set_index("Date")["mape"])
