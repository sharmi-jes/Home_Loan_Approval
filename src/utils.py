import sys
import os
from src.exception import CustomException
import dill
from src.logger import logging
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def evaluate(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        
        for model_name, model in models.items():
            # Retrieve the parameter grid for the current model
            # param_grid = params.get(model_name)
            
            # # Initialize GridSearchCV with the model and its parameters
            # gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            # gs.fit(x_train, y_train)
            model.fit(x_train,y_train)
            
            # Predictions for train and test sets
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            # Compute accuracy scores
            training_score = accuracy_score(y_train, y_train_pred)
            testing_score = accuracy_score(y_test, y_test_pred)
            
            # Log scores for debugging
            logging.info(f"{model_name}: Training Score = {training_score}, Testing Score = {testing_score}")
            
            # Save the best score (test set) for the model
            report[model_name] = testing_score
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
