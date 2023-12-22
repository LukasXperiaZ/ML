import time
from itertools import permutations
from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class GridSearchMLP:
    def __init__(self, params: dict):
        self.params: dict = params
        self.best_score: float = 0.0
        self.best_params: dict = {}
        self.best_model: Pipeline = None

    def find_params(self, X, y, X_test, y_test):
        start_time = time.time()
        time_mean: Dict[float, float] = {}

        for nr_hidden_layers in self.params["nr_hidden_layers"]:
            for hidden_layer_sizes in permutations(self.params["nr_neurons"], r=nr_hidden_layers):
                for activation in self.params["activation"]:
                    for solver in self.params["solver"]:
                        for alpha in self.params["alpha"]:
                            print(f"Trying parameters: {hidden_layer_sizes}, {activation}, {solver}, {alpha}")
                            preprocessor = ColumnTransformer(
                                transformers=[
                                    ("scaler", StandardScaler(), X.select_dtypes(include="number").columns),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore"),
                                     X.select_dtypes(include="object").columns)
                                ]
                            )
                            mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                activation=activation,
                                                solver=solver,
                                                alpha=alpha)
                            pipe = Pipeline(steps=[
                                ('pre', preprocessor),
                                ('mlpc', mlp)
                            ])
                            pipe.fit(X, y)
                            y_pred = pipe.predict(X_test)
                            score = f1_score(y_test, y_pred, average="macro")
                            # scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro")
                            if score > self.best_score:
                                self.best_score = score
                                self.best_params = {
                                    "hidden_layer_sizes": hidden_layer_sizes,
                                    "activation": activation,
                                    "solver": solver,
                                    "alpha": alpha
                                }
                                self.best_model = pipe

                                print(
                                    f"Found new best parameters: {hidden_layer_sizes}, {activation}, {solver}, {alpha}")
                                time_mean[time.time() - start_time] = score

        plt.plot(list(time_mean.keys()),
                 list(time_mean.values()))
        plt.title("Grid search new best config per time")
        plt.xlabel("Time")
        plt.ylabel("F1 score")
        plt.savefig("Grid_Search_Leraning_Curve_Loan.png")
