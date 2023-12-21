import random
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
import matplotlib.pyplot as plt
from hyperp_config import HyperpConfig, Activation, Solver, LearningRate


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


def crossover(parent_1: HyperpConfig, parent_2: HyperpConfig) -> HyperpConfig:
    child_config = HyperpConfig()
    for attr in ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'learning_rate', 'max_iter']:
        if random.random() < 0.5:
            setattr(child_config, attr, getattr(parent_1, attr))
        else:
            setattr(child_config, attr, getattr(parent_2, attr))
    return child_config


def mutate(config: HyperpConfig) -> HyperpConfig:
    if random.random() < 0.05:

        # mutate hidden_layer_sizes
        num_layers = len(config.hidden_layer_sizes)
        if random.random() < 0.5:
            num_layers += 1
            config.hidden_layer_sizes.append(random.randint(1, 100))
        else:
            num_layers -= 1
            config.hidden_layer_sizes.pop()

        # mutate each layer
        for i in range(num_layers):
            if random.random() < 0.05:
                config.hidden_layer_sizes[i] += random.choice([-1, 1])
                if config.hidden_layer_sizes[i] < 1:
                    config.hidden_layer_sizes[i] = 1

    if random.random() < 0.05:
        # mutate activation
        config.activation = random.choice(list(Activation))

    if random.random() < 0.05:
        # mutate solver
        config.solver = random.choice(list(Solver))

    if random.random() < 0.05:
        # mutate alpha
        config.alpha += random.choice([-0.00001, 0.00001])
        if config.alpha < 0:
            config.alpha = 0

    if random.random() < 0.05:
        # mutate learning_rate
        config.learning_rate = random.choice(list(LearningRate))

    if random.random() < 0.05:
        # mutate max_iter
        config.max_iter += random.choice([-50, 50])

    return config


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
    max_time = 60 * 60 + 5  # max time in seconds (65 min)
    # key: iteration, value: (Config, mean)
    performance_dict: Dict[int, (float, HyperpConfig, float)] = {}
    p_d_i = 0
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
        #   crossover and mutation
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
            parent_1 = random.choice(list(only_best_mean_score_dict.values()))[0]
            parent_2 = random.choice(list(only_best_mean_score_dict.values()))[0]
            child_config = crossover(parent_1, parent_2)
            child_config = mutate(child_config)

            mlps.append((child_config, child_config.create_initialized_mlp()))

        # 5. Repeat steps 2-4 until satisfactory algorithm performance is reached or algorithm is no longer improving.
        #       We look at the best mean score of this run (without generated configs)
        # final_sorted_mean_dict = dict(sorted(only_best_mean_score_dict.items()))
        best_mean = sorted(list(only_best_mean_score_dict.keys()))[-1]
        h_p_conf, b_mlp = only_best_mean_score_dict[best_mean]
        performance_dict[p_d_i] = (round(time.time() - start_time, 2), h_p_conf, round(best_mean, 2))
        p_d_i += 1

        if best_mean > 0.98 or (best_mean - best_mean_prev < 0.001 and best_mean != best_mean_prev):
            # finished
            #h_p_conf, b_mlp = final_sorted_mean_dict[best_mean]
            #return h_p_conf, b_mlp, best_mean
            print(f'Break triggered with new best of {best_mean} and previous best {best_mean_prev}')
            break
        best_mean_prev = best_mean

    #print("Ran out of time, quitting.")
    #h_conf, mlp_ = mean_score_dict[best_mean_prev]
    #return h_conf, mlp_, best_mean_prev

    print_10 = True
    print_30 = True
    print_60 = True
    for p_d_i in performance_dict.keys():
        elapsed_time, h_p_conf, mean = performance_dict[p_d_i]
        if elapsed_time > 10*60 and print_10:
            print("After", elapsed_time, "seconds after iteration ", p_d_i, ", we get a mean of:", mean,
                  " with the hyperparameter config:\n", h_p_conf)
            print_10 = False
        if elapsed_time > 30*60 and print_30:
            print("After", elapsed_time, "seconds after iteration ", p_d_i, ", we get a mean of:", mean,
                  " with the hyperparameter config:\n", h_p_conf)
            print_30 = False
        if elapsed_time > 60*60 and print_60:
            print("After", elapsed_time, "seconds after iteration ", p_d_i, ", we get a mean of:", mean,
                  " with the hyperparameter config:\n", h_p_conf)
            print_60 = False

    return performance_dict


if __name__ == "__main__":
    X, Y, bc_preprocessor = satimages_preprocessing()
    performance_dict = evolutionary_optimization(X, Y, bc_preprocessor, 10)
    n_iterations, best_config = list(performance_dict.items())[-1]
    print(f"After {best_config[0]} seconds in the {n_iterations}. iteration, we get a mean of {best_config[2]}"
          f" with the following hyperparameter config:\n", best_config[1])
    #print(performance_dict)

    plt.plot(list(performance_dict.keys()),
             [v[2] for v in performance_dict.values()],
             '-o', color='blue')
    plt.title('Learning curve of the algorithm')
    plt.xlabel('Iterations')
    plt.ylabel('Performance')
    plt.savefig('Learning_Curve_Satimage.png')

