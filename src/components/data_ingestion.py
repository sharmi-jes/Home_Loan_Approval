import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_training import ModelTrainer
from src.components.model_training import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts/train.csv")
    test_data_path:str=os.path.join("artifacts/test.csv")
    raw_data_path:str=os.path.join("artifacts/raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:

         logging.info("read the data")
         df=pd.read_csv(r"D:\RESUME ML PROJECTS\Home Loan Approval\notebooks\clean.csv")

         logging.info("create a directory")
         os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
         df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
         
         logging.info("split the train and test data")
         train_set,test_set=train_test_split(df,test_size=0.2)

         logging.info("passed the train data to the particular train path")
         train_set.to_csv(self.ingestion_config.train_data_path)
          
         logging.info("passed the test data to the particular test path")
         test_set.to_csv(self.ingestion_config.test_data_path)

         logging.info("return the train and test data for the next preocess")

         return(
            train_set,
            test_set
         )
        except Exception as e:
           raise CustomException(e,sys)
        

if __name__=="__main__":
   obj=DataIngestion()
   train_path, test_path=obj.initiate_data_ingestion()

   transformation=DataTransformation()
   train_arr,test_arr,_=transformation.initiate_data_transformation(train_path, test_path)

   trainer=ModelTrainer()
   trainer.initiate_model_trainer(train_arr,test_arr)
