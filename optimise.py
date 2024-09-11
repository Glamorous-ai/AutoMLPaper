from typing import Tuple
import json

import optuna
from optuna.study import Study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from tools.data_cache import DataCache
from tools.recipe import RandomForestClassifierRecipe, CatboostClassifierRecipe, Recipe
from tools.optimisation import OptunaObjective, extract_best_split_info


# --- Globals -----------------------------------------------------------------
# Non-editable globals
RANDOM_STATE = 42
STORAGE = RDBStorage("sqlite:///optimisation.db")

DATA_CACHE = DataCache(
    split_data_path="store/splits.json",
    target_path="store/target.csv",
    features_path={"maccs": "store/maccs.csv", "ecfp2": "store/ecfp2.csv"},
)


# Editable globals
MAX_TRIALS = 10
PRUNER_KWARGS = {}
SAMPLER_KWARGS = {}


# --- Functions ---------------------------------------------------------------


def run_optuna_study(
    model_recipe, score_func, n_trials: int, maximize: bool = True
) -> Tuple[Recipe, Study]:
    objective_function = OptunaObjective(
        model_recipe=model_recipe, score_func=score_func, data_cache=DATA_CACHE
    )

    study = optuna.create_study(
        study_name=objective_function.mint_identifier(),
        pruner=MedianPruner(**PRUNER_KWARGS),
        sampler=TPESampler(**SAMPLER_KWARGS),
        storage=STORAGE,
        direction=["minimize", "maximize"][int(maximize)],
        load_if_exists=True,
    )

    study.set_metric_names([score_func.__name__])

    study.optimize(func=objective_function, n_trials=n_trials, n_jobs=1)
    return model_recipe, study


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    recipe_pool = [
        recipe(feature_name=feature, random_state=RANDOM_STATE)
        for recipe in [RandomForestClassifierRecipe, CatboostClassifierRecipe]
        for feature in ["maccs", "ecfp2"]
    ]

    delayed_func = delayed(run_optuna_study)
    study_results = Parallel(n_jobs=-1)(
        delayed_func(
            model_recipe=recipe,
            score_func=roc_auc_score,
            n_trials=MAX_TRIALS,
            maximize=True,
        )
        for recipe in recipe_pool
    )

    best_recipe_results = {}

    for recipe, study in study_results:
        if recipe.model_name not in best_recipe_results:
            best_recipe_results[recipe.model_name] = {}

        best_recipe_results[recipe.model_name][
            recipe.feature_name
        ] = extract_best_split_info(study)

    with open("optimisation_results.json", "w") as f:
        json.dump(best_recipe_results, f, indent=4)
