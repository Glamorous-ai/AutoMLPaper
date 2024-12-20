import time
import pandas as pd
import optuna
from skopt import Optimizer
from skopt.space import Integer, Real
import random
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
from tdc.single_pred import ADME

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

NUMBER_TRIALS = 50
NUMBER_RUNS = 5  # Number of optimization runs to perform
SEARCH_SPACE = {
    "depth": (2, 10),  # max_depth range
    "learning_rate": (0.1, 0.8),  # learning_rate range
    "iterations": (500, 8000),  # num_trees range
    "l2_leaf_reg": (0.1, 1.0),  # l2_leaf_reg range
}

def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    """Convert a SMILES string to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,), dtype=np.int8)
    ConvertToNumpyArray(fp, arr)
    return arr

def time_function(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        print(f"Timing function: {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    return wrapper

def objective(params):
    """Objective function for hyperparameter optimization."""
    model = CatBoostClassifier(
        depth=int(params["depth"]),
        learning_rate=params["learning_rate"],
        iterations=int(params["iterations"]),
        random_seed=42,
        l2_leaf_reg=params["l2_leaf_reg"],
        verbose=0,
        thread_count=-1,
    )
    score = -np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc"))
    print(f"Score for parameters {params}: {score}")
    return score

@time_function
def run_optuna_tpe_no_pruner():
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(optuna_objective, n_trials=NUMBER_TRIALS, show_progress_bar=True)
    return study.best_params

@time_function
def run_optuna_tpe_pruner():
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(optuna_objective, n_trials=NUMBER_TRIALS, show_progress_bar=True)
    return study.best_params


def optuna_objective(trial):
    params = {
        "depth": trial.suggest_int("depth", *SEARCH_SPACE["depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"]),
        "iterations": trial.suggest_int("iterations", *SEARCH_SPACE["iterations"]),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *SEARCH_SPACE["l2_leaf_reg"]),
    }
    score = objective(params)
    trial.report(score, step=0)
    print(f"Trial {trial.number}: Parameters={params}, Score={score}")
    return score


@time_function
def run_bayesian_optimization(n_trials=NUMBER_TRIALS):
    """Run Bayesian Optimization using Scikit-Optimize."""
    skopt_SEARCH_SPACE = [
        Integer(*SEARCH_SPACE["depth"]),
        Real(*SEARCH_SPACE["learning_rate"]),
        Integer(*SEARCH_SPACE["iterations"]),
        Real(*SEARCH_SPACE["l2_leaf_reg"]),
    ]

    optimizer = Optimizer(skopt_SEARCH_SPACE)
    best_score = float("inf")
    best_params = {}

    for i in range(n_trials):
        suggestion = optimizer.ask()
        params = {
            "depth": suggestion[0],
            "learning_rate": suggestion[1],
            "iterations": suggestion[2],
            "l2_leaf_reg": suggestion[3],
        }
        print(f"Bayesian Optimization Trial {i + 1}/{n_trials}: Evaluating parameters {params}")
        score = objective(params)
        print(f"Score for Trial {i + 1}: {score}")
        optimizer.tell(suggestion, score)
        if score < best_score:
            best_score = score
            best_params = params

    return best_params

@time_function
def run_random_search(n_trials=NUMBER_TRIALS):
    """Run Random Search."""
    best_score = float("inf")
    best_params = {}
    for i in range(n_trials):
        params = {
            "depth": random.randint(*SEARCH_SPACE["depth"]),
            "learning_rate": random.uniform(*SEARCH_SPACE["learning_rate"]),
            "iterations": random.randint(*SEARCH_SPACE["iterations"]),
            "l2_leaf_reg": random.uniform(*SEARCH_SPACE["l2_leaf_reg"]),
        }
        print(f"Random Search Trial {i + 1}/{n_trials}: Evaluating parameters {params}")
        score = objective(params)
        print(f"Score for Trial {i + 1}: {score}")
        if score < best_score:
            best_score = score
            best_params = params
    return best_params


def run_multiple_times(func, n_runs=NUMBER_RUNS, *args, **kwargs):
    """Run a function multiple times and collect results and timing."""
    best_params_all_runs = []
    total_time = []
    for i in range(n_runs):
        print(f"Run {i + 1}/{n_runs}")
        best_params, current_time = func(*args, **kwargs)
        print(f"Time for Run {i + 1}: {current_time:.2f} seconds")
        best_params_all_runs.append(best_params)
        total_time.append(current_time)
    return best_params_all_runs, np.mean(total_time), np.std(total_time)

def statistical_significance_test(test_scores):
    methods = list(test_scores.keys())
    scores = list(test_scores.values())

    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            try:
                stat, p_val = wilcoxon(scores[i], scores[j])
                result = f"{methods[i]} vs {methods[j]}: stat={stat:.3f}, p-value={p_val:.3f}\n"
                print(result)
                with open("optimization_results.txt", "a") as file:
                    file.write(result)
            except ValueError:
                result = f"{methods[i]} vs {methods[j]}: Unable to calculate (no variance)\n"
                print(result)
                with open("optimization_results.txt", "a") as file:
                    file.write(result)

if __name__ == "__main__":
    split = ADME(name="Bioavailability_Ma").get_split(method="scaffold", frac=[0.8, 0, 0.2])
    # use cv later with catboost so only need train and test here
    
    train_data = split["train"]
    test_data = split["test"]

    X_train_smiles = train_data["Drug"]
    y_train = train_data["Y"]
    X_test_smiles = test_data["Drug"]
    y_test = test_data["Y"]

    X_train = np.array([smiles_to_morgan(smiles) for smiles in X_train_smiles])
    X_test = np.array([smiles_to_morgan(smiles) for smiles in X_test_smiles])

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    train_pool = Pool(X_train, y_train)

    best_params_dict = {}

    best_params_dict["Random_Search"], random_search_time, random_search_time_std = run_multiple_times(run_random_search)
    best_params_dict["Bayesian_Optimization"], bayesian_time, bayesian_time_std = run_multiple_times(run_bayesian_optimization)
    best_params_dict["TPE"], tpe_time, tpe_time_std = run_multiple_times(run_optuna_tpe_no_pruner)
    best_params_dict["TPE_Median_Pruning"], tpe_pruner_time, tpe_pruner_time_std = run_multiple_times(run_optuna_tpe_pruner)

    # After optimization, train 5 models and evaluate on the test set
    test_scores = {method: [] for method in best_params_dict}
    
    for method, best_params_all_runs in best_params_dict.items():
        for best_params in best_params_all_runs:
            final_model = CatBoostClassifier(
                depth=best_params["depth"],
                learning_rate=best_params["learning_rate"],
                iterations=best_params["iterations"],
                l2_leaf_reg=best_params["l2_leaf_reg"],
                random_seed=42,
                verbose=0,
                thread_count=-1,
            )

            final_model.fit(X_train, y_train)
            
            y_pred_prob = final_model.predict_proba(X_test)[:, 1]
            
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            
            test_scores[method].append(roc_auc)
            print(f"ROC AUC score for {method}: {roc_auc}")

        print(f"Average ROC AUC score for {method}: {np.mean(test_scores[method])}, std: {np.std(test_scores[method])}")


    statistical_significance_test(test_scores)

    with open("optimization_results.txt", "a") as file:
                file.write("\nTest scores from 5 runs:\n")
                for method, scores in test_scores.items():
                    file.write(f"{method}: {scores}\n")
                file.write("\nAverage test scores, Standard Deviations, and time taken for each method:\n")
                file.write(f"Random_Search: Avg Test Score = {np.mean(test_scores['Random_Search']):.4f}, "
                        f"Std = {np.std(test_scores['Random_Search']):.4f}, Time = {random_search_time} sec {random_search_time_std} \n")
                file.write(f"Bayesian_Optimization: Avg Test Score = {np.mean(test_scores['Bayesian_Optimization']):.4f}, "
                        f"Std = {np.std(test_scores['Bayesian_Optimization']):.4f}, Time = {bayesian_time} sec {bayesian_time_std} \n")
                file.write(f"TPE: Avg Test Score = {np.mean(test_scores['TPE']):.4f}, "
                        f"Std = {np.std(test_scores['TPE']):.4f}, Time = {tpe_time} sec, Time Std ={tpe_time_std} \n")
                file.write(f"TPE_Median_Pruning: Avg Test Score = {np.mean(test_scores['TPE_Median_Pruning']):.4f}, "
                        f"Std = {np.std(test_scores['TPE_Median_Pruning']):.4f}, Time = {tpe_pruner_time}, Time Std = {tpe_pruner_time_std} sec\n")

