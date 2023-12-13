import numpy as np
import pandas as pd
from time import time
# Feature Scaling
from sklearn.preprocessing import StandardScaler
# Regression model
from sklearn.ensemble import GradientBoostingRegressor
# Measurements
from sklearn.metrics import mean_squared_error

# Constant variables & Utils function
from constants import CROSSOVER_RATE, \
                      RMSE_METRIC, \
                      YEARS, \
                      PLAYER_NAME_FEATURE, \
                      YEAR_FEATURE, \
                      MVP_SHARE_FEATURE, \
                      MVP_RANKING_FEATURE, \
                      ACCURACY_METRIC, \
                      LEARNING_RATE, \
                      N_ESTIMATORS

from utils import generate_random_number

class MVPPredictionModel:
    def __init__(self, nba_stats, objectives, hyperparameter=None) -> None:
        self.objectives = objectives
        self.feature_selection = []
        if hyperparameter:
            self.hyperparameter = hyperparameter
        else:
            self.hyperparameter = {
                'n_estimators': generate_random_number(N_ESTIMATORS, is_int=True),
                'learning_rate': generate_random_number(LEARNING_RATE)
            }
        self.model_measurements = pd.DataFrame()
        self.mvp_predicted_df = pd.DataFrame()
        self.nba_stats = nba_stats

        start_time = time()
        self.cross_validation()
        self.execution_time = round(time() - start_time, 2)

    def __gt__(self, other):
        # Dominance
        s_fitness = self.get_fitness()
        o_fitness = other.get_fitness()
        result = True
        for i in range(len(s_fitness)):
            result = result and (s_fitness[i] < o_fitness[i])
        return result

    def __eq__(self, other):
        # Equal
        s_fitness = self.get_fitness()
        o_fitness = other.get_fitness()
        result = True
        for i in range(len(s_fitness)):
            result = result and (s_fitness[i] == o_fitness[i])
        return result

    def __ge__(self, other):
        # Weakly dominance
        s_fitness = self.get_fitness()
        o_fitness = other.get_fitness()
        result = True
        for i in range(len(s_fitness)):
            result = result and (s_fitness[i] <= o_fitness[i])
        return result

    def get_fitness(self):
        # fitness = np.array([self.rmse, 1 - self.r2, self.time])
        fitness = np.array([self.mvp_share_rmse, self.execution_time, self.mvp_ranking_rmse])
        return fitness[self.objectives]

    def train(self, X_train, y_train):
        # Model create
        model = GradientBoostingRegressor(**self.hyperparameter)
        # Fitting model
        model.fit(X_train, y_train)
        return model

    def cross_validation(self):
        """Cross-validation train and test through every NBA seasons"""
        for year in YEARS:
            data_test = self.nba_stats[self.nba_stats[YEAR_FEATURE] == year]
            data_train = self.nba_stats[self.nba_stats[YEAR_FEATURE] != year]

            filter_data = [PLAYER_NAME_FEATURE, MVP_SHARE_FEATURE, MVP_RANKING_FEATURE, YEAR_FEATURE]
            # Split Train and test set
            X_train = data_train.drop(filter_data, axis=1)
            y_train = data_train[MVP_SHARE_FEATURE]
            X_test = data_test.drop(filter_data, axis=1)
            y_test = data_test[MVP_SHARE_FEATURE]

            # Feature Scaling: Standardization
            scaler = StandardScaler()
            scaled_X_train = scaler.fit_transform(X_train)
            scaled_X_test = scaler.transform(X_test)

            model = self.train(scaled_X_train, y_train)
            y_pred, mvp_share_rmse = self.predict(model, scaled_X_test, y_test)

            # Copy test data
            mvp_predicted_by_year_df = data_test[filter_data].copy()
            mvp_predicted_by_year_df[f'Predicted {MVP_SHARE_FEATURE}'] = pd.Series(y_pred).values
            mvp_predicted_by_year_df[f'Predicted {MVP_RANKING_FEATURE}'] = mvp_predicted_by_year_df[f'Predicted {MVP_SHARE_FEATURE}'].rank(ascending=False)
            mvp_ranking_rmse = np.sqrt(mean_squared_error(mvp_predicted_by_year_df[MVP_RANKING_FEATURE], mvp_predicted_by_year_df[f'Predicted {MVP_RANKING_FEATURE}']))

            self.model_measurements = pd.concat([
                self.model_measurements,
                pd.DataFrame({
                    YEAR_FEATURE: [year],
                    RMSE_METRIC: [mvp_share_rmse],
                    ACCURACY_METRIC: [mvp_ranking_rmse]
                })
            ], ignore_index=True)

            self.mvp_predicted_df = pd.concat([self.mvp_predicted_df, mvp_predicted_by_year_df], ignore_index=True)
        
        error_mean = self.model_measurements.mean()
        self.mvp_share_rmse = error_mean[RMSE_METRIC]
        self.mvp_ranking_rmse = error_mean[ACCURACY_METRIC]

    def predict(self, model, X, y):
        """Predict MVP Share of players from current season"""
        y_pred = model.predict(X)
        mse = np.sqrt(mean_squared_error(y, y_pred)) # RMSE
        return y_pred, mse

    def crossover(self, other):
        """Choose one point to crossover hyperparameter"""
        hyperparameter_1 = self.hyperparameter.copy()
        hyperparameter_2 = other.hyperparameter.copy()
        if np.random.rand() < CROSSOVER_RATE:
            point = 1
            count = 0
            for key in self.hyperparameter.keys():
                if count <= point:
                    hyperparameter_1[key] = self.hyperparameter[key]
                    hyperparameter_2[key] = other.hyperparameter[key]
                else:
                    hyperparameter_1[key] = other.hyperparameter[key]
                    hyperparameter_2[key] = self.hyperparameter[key]
        return [
            hyperparameter_1,
            hyperparameter_2
        ]

