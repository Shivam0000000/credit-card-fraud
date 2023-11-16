from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_objects
import os
import sys

@dataclass
class ModelTrainerConfig:
    model_obj_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Entered in the ModelTrainer class")
            X_train,X_test,y_train,y_test=(train_arr[:,:-1],
                                           test_arr[:,:-1],
                                           train_arr[:,-1],
                                           test_arr[:,-1]
                                           )
            logging.info("Seperate independent and dependent feature from train and test set Complete")
             
            model=SVC()
            model.fit(X_train,y_train)
            logging.info("Model Training is Complete")

            y_pred=model.predict(X_test)
            logging.info("Prediction of model is Complete")
            
            save_objects(self.model_trainer.model_obj_path,model)

            logging.info("Model object saved in artifact")

            model_score=accuracy_score(y_test,y_pred)
            logging.info("Model Training is Complete")

            return model_score
        
        except Exception as e:
            raise CustomException(e,sys)



            
            


