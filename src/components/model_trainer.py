import os
import sys

import pandas as pd
import numpy as np
from dataclasses import dataclass

# from sklearn.preprocessing import StandardScaler

from src.utils import save_object, evaluate_model
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
    )
# from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        """
        Intialize configuration of model trainer
        """
        self.model_trainer_config = ModelTrainerConfig()

    def Initiate_model_trainer(self, train_array, test_array):
        try:
            #splitting feature train and test as input and target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Splitted data into X_train, X_test, y_train, y_test")

            models = {
                "Linear Regression" : LinearRegression() ,
                "Random Forest" : RandomForestRegressor() ,
                "Decision Tree" : DecisionTreeRegressor() ,
                "Gradient Boosting" : GradientBoostingRegressor() ,
                "K-Neighbours " : KNeighborsRegressor() ,
                # "XGBClassifier" : XGBRegressor() ,
                "Cat Boosting Classifier" : AdaBoostRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info("Model evaluation completed")
            logging.info(f"evaluation score as : \n{model_report}")

            best_model_name, best_model_score = max(model_report.items(), key=lambda x:x[1])
            best_model = models[best_model_name]

            if best_model_score < 0.60:
                logging.info("No best model found")
                raise CustomException("No best model found")
            logging.info(f"Best model is {best_model} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r_square = r2_score(y_test, predicted)
            return r_square

        except Exception as e:
            raise CustomException(e, sys)