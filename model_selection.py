import json
from joblib import Parallel, delayed
import pickle
from typing import Any


import numpy as np
from sklearn.metrics import accuracy_score

from optimise import DATA_CACHE
from tools.recipe import RandomForestClassifierRecipe, CatboostClassifierRecipe


def create_model(model_name: str, parameters: dict[str, Any]):
    if model_name == "CatBoostClassifier":
        return CatboostClassifierRecipe.prepare_model_with_params(parameters)
    elif model_name == "RandomForestClassifier":
        return RandomForestClassifierRecipe.prepare_model_with_params(parameters)
    else:
        raise ValueError(f"Model {model_name} not supported.")


# Function to train on other folds and test on the excluded fold
def train_on_other_folds(
    models_dict: dict[str, Any], model_name: str, featurization: str, exclude_fold: str
) -> dict[str, Any]:
    model_data = models_dict[model_name][featurization]

    performances = {}

    # Train models using parameters of each fold except the excluded one
    for fold_name, fold_data in model_data.items():
        if fold_name != exclude_fold or fold_name == exclude_fold:
            parameters = fold_data["parameters"]
            model = create_model(model_name, parameters)
            X_train, y_train = DATA_CACHE.load_split(featurization, fold_name, "train")
            X_val, y_val = DATA_CACHE.load_split(featurization, fold_name, "validation")
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            performance = accuracy_score(y_val, predictions)
            performances[fold_name] = performance
            print(
                f"Performance using {fold_name} parameters on {exclude_fold} data: {performance}"
            )
    return performances


def train_on_all_variations_parallel(models_dict: dict[str, Any], n_jobs: int = 1):
    def process_combination(model_name, featurization, exclude_fold):
        print(
            f"Processing: Model={model_name}, Featurization={featurization}, Exclude Fold={exclude_fold}"
        )
        performances = train_on_other_folds(
            models_dict, model_name, featurization, exclude_fold
        )
        return (model_name, featurization, exclude_fold, performances)

    tasks = []

    for model_name, featurizations in models_dict.items():
        for featurization, folds in featurizations.items():
            for exclude_fold in folds:
                tasks.append((model_name, featurization, exclude_fold))

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_combination)(model_name, featurization, exclude_fold)
        for model_name, featurization, exclude_fold in tasks
    )

    return {
        (model_name, featurization, exclude_fold): performances
        for model_name, featurization, exclude_fold, performances in results
    }


def get_best_model_parameters(models_dict, all_results):
    # Dictionary to store performance and parameter stats per (model_name, featurization)
    performance_stats = {}

    # Gather performances and parameters for each fold
    for (model_name, featurization, exclude_fold), performances in all_results.items():
        key = (model_name, featurization)
        if key not in performance_stats:
            performance_stats[key] = {"fold_performances": {}, "fold_parameters": {}}

        performance_stats[key]["fold_performances"][exclude_fold] = list(
            performances.values()
        )
        performance_stats[key]["fold_parameters"][exclude_fold] = models_dict[
            model_name
        ][featurization][exclude_fold]["parameters"]

    # Track the best result
    best_result = None

    # Find the best performing fold for each model and featurization
    for key, data in performance_stats.items():
        model_name, featurization = key
        fold_performances = data["fold_performances"]
        fold_parameters = data["fold_parameters"]

        for fold, performances in fold_performances.items():
            fold_stats = {
                "median_performance": np.median(performances),
                "mean_performance": np.mean(performances),
                "std_performance": np.std(performances),
                "fold": fold,
                "parameters": fold_parameters[fold],
            }

            if best_result is None or (
                fold_stats["median_performance"],
                fold_stats["mean_performance"],
                -fold_stats["std_performance"],
            ) > (
                best_result["best_median_performance"],
                best_result["best_mean_performance"],
                -best_result["best_std_performance"],
            ):
                best_result = {
                    "model_name": model_name,
                    "featurization": featurization,
                    "best_median_performance": fold_stats["median_performance"],
                    "best_mean_performance": fold_stats["mean_performance"],
                    "best_std_performance": fold_stats["std_performance"],
                    "best_fold": fold_stats["fold"],
                    "best_parameters": fold_stats["parameters"],
                }

    return best_result


def train_full_model(best_results):
    model = create_model(best_results["model_name"], best_results["best_parameters"])
    X_train, Y_train = DATA_CACHE.load_split(
        best_results["featurization"], "random_split", "train"
    )
    X_val, Y_val = DATA_CACHE.load_split(
        best_results["featurization"], "random_split", "validation"
    )
    X_test, Y_test = DATA_CACHE.load_split(
        best_results["featurization"], "random_split", "test"
    )

    X_total_train = np.concatenate([X_train, X_val])
    Y_total_train = np.concatenate([Y_train, Y_val])

    model.fit(X_total_train, Y_total_train)

    print(f"Performance on test set: {accuracy_score(Y_test, model.predict(X_test))}")

    with open(
        f'best_model_{best_results["model_name"]}_{best_results["featurization"]}.pkl',
        "wb",
    ) as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    with open("optimisation_results.json", "r") as f:
        optimisation_results = json.load(f)

    all_results = train_on_all_variations_parallel(
        models_dict=optimisation_results, n_jobs=-1
    )

    ranked_results = get_best_model_parameters(optimisation_results, all_results)

    print(f"Best model found: {ranked_results['model_name']} ")
    print(f"Featurization: {ranked_results['featurization']}")
    print(f"Best median performance: {ranked_results['best_median_performance']}")
    print(f"Best mean performance: {ranked_results['best_mean_performance']}")
    print(f"Best std performance: {ranked_results['best_std_performance']}")
    print(f"Best parameters: {ranked_results['best_parameters']}")

    print("Training full model with best parameters...")

    train_full_model(ranked_results)
