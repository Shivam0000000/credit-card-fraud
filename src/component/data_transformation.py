import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_objects

@dataclass
class DataTransformationConfig:
    StandardScaler_path=os.path.join('artifact',"scaler.pkl")

class DataTransformation:
    logging.info("Enter in the Data Transformation Class")
    def __init__(self):
        self.data_transformation=DataTransformationConfig()

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_set=pd.read_csv(train_path)
            test_set=pd.read_csv(test_path)
            logging.info("Read the train and test set Completed")

            X_train=train_set.drop('Class',axis=1)
            X_test=test_set.drop('Class',axis=1)

            y_train=train_set['Class']
            y_test=test_set['Class']
            logging.info("Seperate independent and dependent feature from train and test set Complete")
            
            scaler=StandardScaler()
            X_train_scaled=scaler.fit_transform(X_train)
            X_test_scaled=scaler.transform(X_test)
            logging.info("Scaled X train and X test Completed")

            save_objects(file_path=self.data_transformation.StandardScaler_path,obj=scaler)
            logging.info("save scaler object Complete")

            train_arr=np.c_[X_train_scaled,np.array(y_train)]
            test_arr=np.c_[X_test_scaled,np.array(y_test)]

            logging.info("Data Transformation is Complete")

            return train_arr,test_arr,self.data_transformation.StandardScaler_path

        except Exception as e:
            raise CustomException(e,sys)    
            

