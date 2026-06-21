import os
import sys
import numpy as np
import pandas as pd

from CustomerChurnPrediction.utils.logger import logger
from CustomerChurnPrediction.utils.common import create_directories,save_data
from CustomerChurnPrediction.entity.config_entity import DataTransformationConfig

class DataTransformation:
    """
    Handles transformation of raw customer churn data into a model-ready
    format: encoding categorical features, collapsing redundant dummy
    columns, and checking for multicollinearity via VIF.
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Args:
            config (DataTransformationConfig): Configuration containing
                input/output paths for the transformation stage.
        """
        self.config = config

    def get_input_data(self) -> pd.DataFrame:
        """Load the raw input CSV specified in the config."""
        input_data_path = self.config.input_data_path
        logger.info(f"Loading input data from: {input_data_path}")

        return pd.read_csv(input_data_path)


    def binary_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode simple Yes/No and Male/Female columns as 1/0.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with binary columns encoded.
        """
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
        df[binary_cols] = df[binary_cols].replace(
            {
                'Yes': 1,
                'No': 0,
                'Male': 1,
                'Female': 0
            }
        )
        df[binary_cols] = df[binary_cols].astype(int)

        return df

    def one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode multi-category columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with multi-category columns one-hot encoded.
        """
        multi_cat_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod'
        ]
        return pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    def bool_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert any boolean columns (from one-hot encoding) to 0/1 integers."""
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(int)
        return df

    def collapse_redundant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse redundant 'No internet service' / 'No phone service' dummy
        columns into single combined flags, since these are perfectly
        correlated duplicates created by one-hot encoding.

        Args:
            df (pd.DataFrame): One-hot encoded dataframe.

        Returns:
            pd.DataFrame: Dataframe with redundant dummy columns collapsed.
        """
            # Combine all '*_No internet service' columns into one flag
        no_internet_cols = [
            'OnlineSecurity_No internet service',
            'OnlineBackup_No internet service',
            'DeviceProtection_No internet service',
            'TechSupport_No internet service',
            'StreamingTV_No internet service',
            'StreamingMovies_No internet service',
        ]
        existing_no_internet_cols = [c for c in no_internet_cols if c in df.columns]
        if existing_no_internet_cols:
            df['No_internet_service'] = df[existing_no_internet_cols].any(axis=1).astype(int)
            df = df.drop(columns=existing_no_internet_cols)
        # Handle PhoneService redundancy
        if 'MultipleLines_No phone service' in df.columns:
            df['No_phone_service'] = df['MultipleLines_No phone service'].astype(int)
            df = df.drop(columns=['MultipleLines_No phone service'])
        return df


    def get_transform_data(self) -> pd.DataFrame:
        """
        Run the full transformation pipeline: load data, drop ID column,
        binary-encode, one-hot encode, collapse redundant dummies, and
        convert booleans to ints.

        Returns:
            pd.DataFrame: Fully transformed, model-ready dataframe.
        """
        df = self.get_input_data()
        df = df.drop(columns='customerID')
        df = self.binary_encoding(df)
        df = self.one_hot_encoding(df)
        df = self.collapse_redundant_columns(df)
        df = self.bool_to_int(df)
        logger.info(f"Data transformation complete: {df.shape[0]} rows, {df.shape[1]} columns")

        save_data(df,self.config.transformed_data_path)
        