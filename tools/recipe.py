from typing import Any

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from optuna.trial import Trial


class Recipe:
    def __init__(self, feature_name, random_state, model_name):
        self.feature_name = feature_name
        self.random_state = random_state
        self.model_name = model_name

    def __repr__(self):
        return f"Recipe({self.model_name}, {self.feature_name})"

    def suggest_predictor(self, trial: Trial) -> Any:
        pass


class RandomForestClassifierRecipe(Recipe):
    def __init__(self, feature_name, random_state):
        super().__init__(feature_name, random_state, "RandomForestClassifier")

    def suggest_predictor(self, trial: Trial) -> Any:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100, 10),
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        return RandomForestClassifier(**params, random_state=self.random_state)

    @staticmethod
    def prepare_model_with_params(params):
        return RandomForestClassifier(**params)


class CatboostClassifierRecipe(Recipe):
    def __init__(self, feature_name, random_state):
        super().__init__(feature_name, random_state, "CatBoostClassifier")

    def suggest_predictor(self, trial: Trial) -> Any:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
        }
        return CatBoostClassifier(**params, random_seed=self.random_state)

    @staticmethod
    def prepare_model_with_params(params):
        return CatBoostClassifier(**params)
