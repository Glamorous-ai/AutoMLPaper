import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tdc.single_pred import ADME
from tdc.chem_utils import MolConvert

TRAIN_SIZE = 0.8
RANDOM_STATE = 42
N_FOLDS = 5


# ensure the store directory exists
if not os.path.exists("store"):
    os.makedirs("store")


# Source chemical dataset from TDC
df = ADME(name="Bioavailability_Ma").get_data()[["Drug", "Y"]].drop_duplicates(["Drug"])
df.columns = ["smiles", "target"]

df[["smiles"]].to_csv("store/smiles.csv", index=False)
df[["target"]].to_csv("store/target.csv", index=False)


# Generate and store relevant chemical transformations
smiles = df["smiles"].tolist()
maccs = MolConvert(src="SMILES", dst="MACCS")(smiles)
ecfp2 = MolConvert(src="SMILES", dst="ECFP2")(smiles)

pd.DataFrame(maccs).to_csv("store/maccs.csv", index=False)
pd.DataFrame(ecfp2).to_csv("store/ecfp2.csv", index=False)


# Generate split mappings and write to disk for later usage
indices = np.arange(len(df))
train_indices, test_indices = train_test_split(
    indices, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
)
train_indices_full, validation_indices_full = train_test_split(
    train_indices, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
)
k_fold_splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cross_val_indices = {
    f"cross_val_{int(idx)}": {
        "train": train.tolist(),
        "validation": validation.tolist(),
    }
    for idx, (train, validation) in enumerate(k_fold_splitter.split(train_indices))
}

all_splits = {
    "random_split": {
        "train": train_indices_full.tolist(),
        "validation": validation_indices_full.tolist(),
        "test": test_indices.tolist(),
    },
    **cross_val_indices,
}

with open("store/splits.json", "w") as f:
    json.dump(all_splits, f, indent=4)
