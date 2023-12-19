from random import seed
from typing import List, Dict

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

    # random initialization
    seed(time.time())

    # 1. Create initial population of random solutions (pool_size many mlp with randomly initialized hyperparameters)
    mlps: List[(HyperpConfig, MLPClassifier)] = []
    for i in range(pool_size):
        hyper_p_config = HyperpConfig()
        hyper_p_config.random_init()
        mlps.append((hyper_p_config, hyper_p_config.create_initialized_mlp()))

    mean_score_dict: Dict[float, (HyperpConfig, MLPClassifier)] = {}

    # --- Loop ---
    best_mean_prev: float = 0
    start_time = time.time()
    max_time = 600  # max time in seconds (10 min)
    while time.time() - start_time < max_time:
        # 2. Evaluate the hyperparameter tuples and acquire their fitness function (5-fold cross-validation f1 value)
        for hyper_p_config, mlp in mlps:
            pipe = Pipeline(steps=[
                ('pre', preprocessor),
                ('mlpc', mlp)
            ])

            scores = cross_val_score(pipe, X, Y, cv=5, scoring="f1_macro")
            # Sometimes two configurations have the same mean
            #   -> They differ by something that does not matter
            #   -> Since we use a dict we only keep one MLP with the same value
            mean = scores.mean()
            mean_score_dict[mean] = (hyper_p_config, mlp)

        # 3. Rank the hyperparameter tuples by their relative fitness
        mean_score_sorted_dict = dict(sorted(mean_score_dict.items()))

        # 4. Replace the worst-performing hyperparameter tuples with new hyperparameter tuples generated through
        #   - crossover and
        #   - mutation
        delete_percentage = 0.3
        n_delete = len(mean_score_sorted_dict) * delete_percentage
        only_best_mean_score_dict: Dict[float, (HyperpConfig, MLPClassifier)] = {}
        i = 0
        # Replace the delete_percentage worst-performing configs
        for mean in mean_score_sorted_dict.keys():
            if i > n_delete:
                only_best_mean_score_dict[mean] = mean_score_sorted_dict[mean]
            i += 1

        # MLPs for the next round: start with best (HyperpConfig, MLPClassifier) tuples:
        mlps: List[(HyperpConfig, MLPClassifier)] = list(only_best_mean_score_dict.values())

        # Generate pool_size - len(only_best_mean_score_dict) new configs with crossover and mutation
        for i in range(0, pool_size - len(only_best_mean_score_dict)):
            pass
            # TODO
            # put generated (HyperpConfig, MLPClassifier) tuples in mlps list
            # mlps.append((..., ...))

        # 5. Repeat steps 2-4 until satisfactory algorithm performance is reached or algorithm is no longer improving.
        #       We look at the best mean score of this run (without generated configs)
        final_sorted_mean_dict = dict(sorted(only_best_mean_score_dict.items()))
        best_mean = list(final_sorted_mean_dict.keys())[len(final_sorted_mean_dict)-1]
        if best_mean > 0.99 or best_mean - best_mean_prev < 0.01:
            # finished
            h_p_conf, b_mlp = final_sorted_mean_dict[best_mean]
            return h_p_conf, b_mlp, best_mean

        best_mean_prev = best_mean

    print("Ran out of time, quitting.")
    h_conf, mlp_ = mean_score_dict[best_mean_prev]
    return h_conf, mlp_, best_mean_prev


if __name__ == "__main__":
    bc_X, bc_Y, bc_preprocessor = breast_cancer_preprocessing()
    hyper_p_config, mlp, best_mean = evolutionary_optimization(bc_X, bc_Y, bc_preprocessor, 10)
    print("Best config is:")
    print(hyper_p_config, "\n",
          "With mean f1 score: ", best_mean)
