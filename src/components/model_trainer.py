import os, sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arrey, test_arrey):
        try:
            logging.info("Splitting training and test input data...")

            X_train,y_train,X_test,y_test = (
                train_arrey[:,:-1],
                train_arrey[:,-1],

                test_arrey[:,:-1],
                test_arrey[:,-1]
            )
            logging.info("Splitting training and test input data is completed.")

            models = {
                "RandomForestRegressor" :RandomForestRegressor(),
                "AdaBoostRegressor" :AdaBoostRegressor(),
                "GradientBoostingRegressor" :GradientBoostingRegressor(),
                "DecisionTreeRegressor" :DecisionTreeRegressor(),
                "KNeighborsRegressor" :KNeighborsRegressor(),
                "LinearRegression" :LinearRegression(),
                "CatBoostRegressor" :CatBoostRegressor(verbose=False),
                "XGBRegressor" :XGBRegressor(),
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models= models)

            ## To get best model score from dict

            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if(best_model_score<0.6):
                raise CustomException("No best model found..")
            

            logging.info("Best found model on both training and testing dataset.")

            # preprossing_obj=  ## load prepossessor pickle file if needed

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                )
            
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2


        except Exception as e:
            raise CustomException(e,sys)
    

    
        
