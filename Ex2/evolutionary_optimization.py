from typing import List

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import time
import pandas as pd

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

    mlps: List[MLPClassifier] = []
    for i in range(pool_size):
        # create pool_size many mlp with randomly initialized hyperparameters
        hyper_p_config = HyperpConfig()
        hyper_p_config.random_init()
        mlps.append(hyper_p_config.create_initialized_mlp())

    mean_score_list: List[float] = []

    for mlp in mlps:
        pipe = Pipeline(steps=[
            ('pre', preprocessor),
            ('mlpc', mlp)
        ])

        scores = cross_val_score(pipe, X, Y, cv=5, scoring="f1_macro")
        mean = scores.mean()
        mean_score_list.append(mean)

    print(mean_score_list)


if __name__ == "__main__":
    bc_X, bc_Y, bc_preprocessor = breast_cancer_preprocessing()
    evolutionary_optimization(bc_X, bc_Y, bc_preprocessor, 100)