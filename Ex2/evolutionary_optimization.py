from typing import List, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Ex2.hyperp_config import HyperpConfig


def breast_cancer_preprocessing():
    # Load the data
    breast_data = pd.read_csv("../datasets/BreastCancer/breast-cancer-diagnostic.shuf.lrn.csv")
    # Drop ID
    breast_data.pop("ID")
    # Extract the label
    X = breast_data.copy(deep=True)
    Y = X.pop("class")

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), X.columns)
        ]
    )

    return X, Y, preprocessor


def evolutionary_optimization(X, Y, preprocessor, pool_size: int):
    # https://en.wikipedia.org/wiki/Hyperparameter_optimization
    # https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)

    # create pool_size many mlp with randomly initialized hyperparameters
    mlps: List[MLPClassifier] = []
    for i in range(pool_size):
        hyper_p_config = HyperpConfig()
        hyper_p_config.random_init()
        mlps.append(hyper_p_config.create_initialized_mlp())

    mean_score_dict: Dict[float, MLPClassifier] = {}

    for mlp in mlps:
        pipe = Pipeline(steps=[
            ('pre', preprocessor),
            ('mlpc', mlp)
        ])

        scores = cross_val_score(pipe, X, Y, cv=5, scoring="f1_macro")
        mean = scores.mean()
        mean_score_dict[mean] = mlp

    print(list(mean_score_dict.keys()))


if __name__ == "__main__":
    bc_X, bc_Y, bc_preprocessor = breast_cancer_preprocessing()
    evolutionary_optimization(bc_X, bc_Y, bc_preprocessor, 100)
