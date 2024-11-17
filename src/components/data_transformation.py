import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    """
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Handles the data transformation process, including preprocessing pipelines
    for numerical and categorical features.
    """
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_as_transformation(self):
        """
        Creates and returns a ColumnTransformer object for preprocessing.
        """
        # Define numerical and categorical columns
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                          'Loan_Amount_Term', 'Credit_History']
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                            'Self_Employed', 'Property_Area']
        
        # Pipeline for numerical features
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        # Pipeline for categorical features
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehotencoder", OneHotEncoder())
            ]
        )

        # Combine pipelines into a preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, numerical_cols),
                ("cat_pipeline", cat_pipeline, categorical_cols)
            ]
        )
        return preprocessor

    def initiate_data_transformation(self, train_data, test_data):
        """
        Transforms train and test data using the preprocessor and saves the preprocessor object.

        Args:
            train_data (str | pd.DataFrame): Path to train data CSV or DataFrame.
            test_data (str | pd.DataFrame): Path to test data CSV or DataFrame.

        Returns:
            Tuple: Processed train array, test array, and preprocessor object file path.
        """
        try:
            # Load train data
            if isinstance(train_data, str):
                assert os.path.exists(train_data), f"Train file not found: {train_data}"
                logging.info(f"Reading train data from: {train_data}")
                train_df = pd.read_csv(train_data)
            elif isinstance(train_data, pd.DataFrame):
                train_df = train_data
            else:
                raise ValueError("train_data must be a file path or a DataFrame")

            # Load test data
            if isinstance(test_data, str):
                assert os.path.exists(test_data), f"Test file not found: {test_data}"
                logging.info(f"Reading test data from: {test_data}")
                test_df = pd.read_csv(test_data)
            elif isinstance(test_data, pd.DataFrame):
                test_df = test_data
            else:
                raise ValueError("test_data must be a file path or a DataFrame")

            logging.info("Creating preprocessor object")
            preprocessor_obj = self.get_data_as_transformation()

            # Target column
            target_col = "Loan_Status"

            logging.info("Separating input features and target column")
            X_train = train_df.drop(columns=target_col)
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=target_col)
            y_test = test_df[target_col]

            logging.info("Applying preprocessor to train and test data")
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)

            logging.info("Combining transformed data with target")
            train_array = np.c_[X_train_transformed, np.array(y_train)]
            test_array = np.c_[X_test_transformed, np.array(y_test)]

            logging.info("Saving the preprocessor object")
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            return train_array, test_array, self.transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)
