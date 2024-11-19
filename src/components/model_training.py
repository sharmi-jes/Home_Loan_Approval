import sys
import os
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from src.utils import save_object,evaluate

@dataclass
class ModelTrainerConfig:
    model_file_path:str=os.path.join("artifacts/model.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:

        # Splitting the data into features (X) and target (y)
         x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


        # Define models
         models = {
            "Logistic Regression": LogisticRegression(solver='saga', max_iter=2000, C=0.01),
            "Random Forest": RandomForestClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVC": SVC()
        }

        # Define parameter grids
        #  params = {
        #     "Logistic Regression": [
        #         {
        #             "C": [0.1, 1, 10, 100],
        #             "penalty": ["l1", "l2"],
        #             "solver": ["liblinear"]
        #         },
        #         {
        #             "C": [0.1, 1, 10, 100],
        #             "penalty": ["l2"],
        #             "solver": ["newton-cg", "lbfgs", "saga"]
        #         },
        #         {
        #             "C": [0.1, 1, 10, 100],
        #             "penalty": ["elasticnet"],
        #             "solver": ["saga"],
        #             "l1_ratio": [0.5]
        #         }
        #     ],
        #     "Random Forest": {
        #         "n_estimators": [50, 100, 200],
        #         "max_depth": [None, 10, 20, 30],
        #         "min_samples_split": [2, 5, 10],
        #         "min_samples_leaf": [1, 2, 4]
        #     },
        #     "AdaBoost": {
        #         "n_estimators": [50, 100, 200],
        #         "learning_rate": [0.01, 0.1, 0.5, 1.0]
        #     },
        #     "Gradient Boosting": {
        #         "n_estimators": [50, 100, 200],
        #         "learning_rate": [0.01, 0.1, 0.2],
        #         "max_depth": [3, 5, 7],
        #         "min_samples_split": [2, 5, 10]
        #     },
        #     "SVC": {
        #         "C": [0.1, 1, 10, 100],
        #         "kernel": ["linear", "rbf", "poly", "sigmoid"],
        #         "gamma": ["scale", "auto"],
        #         "degree": [2, 3, 4]
        #     }
        # }

        # Evaluate models with parameters
         model_report: dict = evaluate(x_train, y_train, x_test, y_test, models)

        # Log and print the best score
         logging.info("Get the best score")
         best_score = max(sorted(model_report.values()))
         print(f"Best Score: {best_score}")

        # Log and get the best model name
         logging.info("Get the best model name")
         best_name = list(model_report.keys())[
            list(model_report.values()).index(best_score)
        ]

        # Fetch the best model
         best_model = models[best_name]
         print(f"Best Model: {best_model}")

        # Check if the best score is satisfactory
         logging.info("Check the relation")
         if best_score < 0.6:
            raise CustomException("No suitable model found with good performance.")

         logging.info("A good model has been identified for both train and test data.")

        # Save the best model
         save_object(
            file_path=self.trainer_config.model_file_path,
            obj=best_model
        )

        # Predict and evaluate accuracy on test data
         logging.info("Model predictions on test data")
         x_prediction = best_model.predict(x_test)
         prediction = accuracy_score(y_test, x_prediction)
         print(f"Test Accuracy: {prediction}")

         return prediction
        except Exception as e:
             raise CustomException(e, sys)

