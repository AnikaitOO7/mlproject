import sys  # Provides access to system-specific parameters and functions (e.g., handling exceptions or exiting the script).
from dataclasses import dataclass  # Allows creating lightweight classes to store data without boilerplate code.

import numpy as np  # Provides support for numerical operations and handling arrays.
import pandas as pd  # Useful for data manipulation, loading datasets, and working with tabular data.
import os
from sklearn.compose import ColumnTransformer  # Helps in applying different preprocessing steps to specific columns of data.
from sklearn.impute import SimpleImputer  # Handles missing values by applying strategies like mean, median, or constant replacement.
from sklearn.pipeline import Pipeline  # Enables chaining multiple data processing steps and models into a single workflow.
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
# OneHotEncoder: Converts categorical variables into a binary matrix.
# StandardScaler: Standardizes features by removing the mean and scaling to unit variance.

from src.exception import CustomException  # Custom class to handle and log specific exceptions for debugging.
from src.logger import logging  # Handles logging of events for tracking the code's execution and errors.
from src.utils import save_object

@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        #This function is responcible for data transformation

        try:
            numerical_columns = ["writing score","reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns:{numerical_columns}")
            logging.info(f"Categorical columns:{categorical_columns}")

            preprocessor= ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipline,categorical_columns)
                ]
            
            )
            return preprocessor

        except Exception as e :
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

