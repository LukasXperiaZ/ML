from typing import List

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import time
import pandas as pd
from scipy.io import arff
from hyperp_config import HyperpConfig


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

def loan_preprocessing():
    # load data
    loan = pd.read_csv("../datasets/Loan/loan-10k.lrn.csv", engine='python')

    # preprocessing
    X = loan.drop(['ID', 'grade'], axis = 1)
    Y = loan.grade

    numeric_features = X.drop(
        ['term', 'emp_length', 'home_ownership', 'verification_status', 'loan_status',
         'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status', 'application_type',
         'hardship_flag', 'debt_settlement_flag', 'disbursement_method'],
        axis=1
    ).columns

    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numeric_features),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    return X, Y, preprocessor

def satimages_preprocessing():
    satimage_arff = arff.loadarff('../datasets/Satimage/dataset_186_satimage.arff')
    satimage = pd.DataFrame(satimage_arff[0])

    X = satimage.drop(['class'], axis = 1)
    Y = satimage['class'].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), X.columns)
        ]
    )

    return X, Y, preprocessor


def evolutionary_optimization(X, Y, preprocessor, pool_size: int):
    # https://en.wikipedia.org/wiki/Hyperparameter_optimization
    # https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)

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
    bc_X, bc_Y, bc_preprocessor = satimages_preprocessing()
    evolutionary_optimization(bc_X, bc_Y, bc_preprocessor, 3)