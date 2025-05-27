import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split training and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear regression": LinearRegression(),
                "random forest": RandomForestRegressor(),
                "gradient forest": GradientBoostingRegressor(),
                "ada boost": AdaBoostRegressor(),
                "k neighbours": KNeighborsRegressor(),
                "decision tree": DecisionTreeRegressor(),
                "xgbooost": XGBRegressor()
            }
            param = {
                "Linear regression": {},
                "random forest": {
                    'n_estimators':[8,16,32,64,128],
                    'max_features':['sqrt','log2',None]
                },
                "gradient forest": {
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128]
                },
                "ada boost": {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128]
                },
                "k neighbours": {
                    'n_neighbors':[5,7,9,11]
                },
                "decision tree": {
                    'max_features':['sqrt','log2'],
                    'criterion' :['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "xgbooost": {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128]
                }
            }

            model_report: dict=evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,models=models,param=param)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info("best model found on training and testonf data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(x_test)
            r2_score1 = r2_score(y_test, predicted)
            return r2_score1, best_model

        except Exception as e:
            raise CustomException(e,sys)