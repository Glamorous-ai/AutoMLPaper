from typing import Callable, Dict, Any
import random
from uuid import uuid4

from optuna.trial import Trial
from optuna.study import Study
import pandas as pd

from tools.data_cache import DataCache
from tools.recipe import Recipe


def build_uniform_sampler(values):
    while True:
        shuffled_list = random.sample(values, len(values))  # Shuffle the list
        for item in shuffled_list:
            yield item


class OptunaObjective:
    def __init__(
        self, model_recipe: Recipe, score_func: Callable, data_cache: DataCache
    ):
        self.model_recipe = model_recipe
        self.score_func = score_func
        self.data_cache = data_cache
        self.split_names = build_uniform_sampler(data_cache.get_crossval_split_names())

    def __call__(self, trial: Trial) -> float:
        split_name = next(self.split_names)
        trial.set_user_attr(
            "split_name", split_name
        )  # store so can associate split with model

        X_train, y_train = self.data_cache.load_split(
            self.model_recipe.feature_name, split_name, "train"
        )
        X_val, y_val = self.data_cache.load_split(
            self.model_recipe.feature_name, split_name, "validation"
        )

        model = self.model_recipe.suggest_predictor(trial)
        model.fit(X_train, y_train)
        return self.score_func(y_val, model.predict_proba(X_val))

    def mint_identifier(self) -> str:
        return f"{self.model_recipe.model_name}_{self.model_recipe.feature_name}_{uuid4().hex}"


def _extract_best_performance_per_split(study: Study) -> pd.DataFrame:
    metric_name = f"value_{study.metric_names[0]}"
    maximise_metric = study.direction == "maximize"

    df = study.trials_dataframe()
    df = df.query('state == "COMPLETE"')  # should all be complete but just incase here
    best_split_rows = df.groupby("user_attrs_split_name")[metric_name]
    best_split_rows_idx = (
        best_split_rows.idxmax() if maximise_metric else best_split_rows.idxmin()
    )
    df = df.loc[best_split_rows_idx]
    return df, metric_name


def _parse_optuna_df_to_dict(
    df: pd.DataFrame, metric_name: str
) -> Dict[str, Dict[str, Any]]:
    return {
        row["user_attrs_split_name"]: {
            "parameters": {
                param.replace("params_", ""): row[param]
                for param in row.index
                if "params_" in param
            },
            "performance": row[metric_name],
        }
        for _, row in df.iterrows()
    }


def extract_best_split_info(study: Study) -> Dict[str, Dict[str, Any]]:
    df, metric_name = _extract_best_performance_per_split(study)
    return _parse_optuna_df_to_dict(df, metric_name)
