from src.logger import logging
from src.exception import CustomException
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
from src.component.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join('artifact','raw_data.csv')
    train_data_path=os.path.join('artifact','train_data.csv')
    test_data_path=os.path.join('artifact','test_data.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Enter the DataIngestion")  

            df=pd.read_csv('notebook/data/creditcard.csv')
            

            os.makedirs(os.path.dirname(self.data_ingestion.train_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion.raw_data_path,index=False,header=True)

            logging.info("Train Test Split")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.data_ingestion.train_data_path,index=False,header=True)

            test_set.to_csv(self.data_ingestion.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is completed")

            return (self.data_ingestion.train_data_path ,self.data_ingestion.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train,test=obj.initiate_data_ingestion()

    obj2=DataTransformation()
    train_arr,test_arr,_=obj2.initiate_data_transformation(train,test)

