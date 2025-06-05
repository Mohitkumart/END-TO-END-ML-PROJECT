from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


model_params={
    "Decision Tree": {
        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        # 'splitter':['best','random'],
        # 'max_features':['sqrt','log2'],
    },
    "Random Forest":{
        # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        
        # 'max_features':['sqrt','log2',None],
        'n_estimators': [8,16,32,64,128,256]
    },
    "Gradient Boosting":{
        # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
        'learning_rate':[.1,.01,.05,.001],
        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
        # 'criterion':['squared_error', 'friedman_mse'],
        # 'max_features':['auto','sqrt','log2'],
        'n_estimators': [8,16,32,64,128,256]
    },
    "Linear Regression":{},
    "XGBRegressor":{
        'learning_rate':[.1,.01,.05,.001],
        'n_estimators': [8,16,32,64,128,256]
    },
    "CatBoosting Regressor":{
        'depth': [6,8,10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    "AdaBoost Regressor":{
        'learning_rate':[.1,.01,0.5,.001],
        # 'loss':['linear','square','exponential'],
        'n_estimators': [8,16,32,64,128,256]
    }
    
}

# model_params = {
#     "RandomForestRegressor": {
#         "model": RandomForestRegressor(),
#         "params": {
#             'n_estimators': [100, 200],
#             'max_depth': [None, 10, 20],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2],
#             'max_features': ['auto', 'sqrt']
#         }
#     },
#     "AdaBoostRegressor": {
#         "model": AdaBoostRegressor(),
#         "params": {
#             'n_estimators': [50, 100],
#             'learning_rate': [0.01, 0.1, 1.0],
#             'loss': ['linear', 'square']
#         }
#     },
#     "GradientBoostingRegressor": {
#         "model": GradientBoostingRegressor(),
#         "params": {
#             'n_estimators': [100, 200],
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'subsample': [0.8, 1.0]
#         }
#     },
#     "DecisionTreeRegressor": {
#         "model": DecisionTreeRegressor(),
#         "params": {
#             'max_depth': [None, 10],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         }
#     },
#     "KNeighborsRegressor": {
#         "model": KNeighborsRegressor(),
#         "params": {
#             'n_neighbors': [3, 5, 7],
#             'weights': ['uniform', 'distance'],
#             'algorithm': ['auto', 'kd_tree']
#         }
#     },
#     "LinearRegression": {
#         "model": LinearRegression(),
#         "params": {
#             'fit_intercept': [True, False]
#         }
#     },
#     "CatBoostRegressor": {
#         "model": CatBoostRegressor(verbose=False),
#         "params": {
#             'iterations': [100, 200],
#             'learning_rate': [0.01, 0.1],
#             'depth': [6, 8],
#             'l2_leaf_reg': [1, 3]
#         }
#     },
#     "XGBRegressor": {
#         "model": XGBRegressor(),
#         "params": {
#             'n_estimators': [100, 200],
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'subsample': [0.8, 1.0],
#             'colsample_bytree': [0.8, 1.0]
#         }
#     }
# }
