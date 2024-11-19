import sys
import os
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from src.logger import logging
import pandas as pd
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Define paths for model and preprocessor
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            # Check if model and preprocessor files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found. Please train and save the model first.")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError("Preprocessor file not found. Please save the preprocessor pipeline.")

            logging.info("Loading the model and preprocessor for prediction")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Log the loaded objects for debugging
            logging.info(f"Model type: {type(model)}")
            logging.info(f"Preprocessor type: {type(preprocessor)}")

            # Apply preprocessor to the input features
            logging.info("Transforming features using the preprocessor")
            data_scaled = preprocessor.transform(features)

            # Log the shape of transformed data
            logging.info(f"Transformed data shape: {data_scaled.shape}")

            # Predict the output
            logging.info("Making predictions using the model")
            prediction = model.predict(data_scaled)

            # Log the prediction
            logging.info(f"Prediction: {prediction}")

            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, 
                 Gender, Married, Dependents, Education, Self_Employed, Property_Area):
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.Property_Area = Property_Area

    def get_data_dataframe(self):
        try:
            logging.info("Creating a DataFrame from the input data")
            input_data = {
                "ApplicantIncome": [self.ApplicantIncome],
                "CoapplicantIncome": [self.CoapplicantIncome],
                "LoanAmount": [self.LoanAmount],
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Credit_History": [self.Credit_History],
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "Property_Area": [self.Property_Area]
            }
            return pd.DataFrame(input_data)
        except Exception as e:
            raise CustomException(e, sys)
