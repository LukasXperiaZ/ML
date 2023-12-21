import time
from itertools import permutations

import pandas as pd
from sklearn.model_selection import cross_val_score
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

    
    def find_params(self, X, y):
        start_time = time.time()
        for nr_hidden_layers in self.params["nr_hidden_layers"]:
            for hidden_layer_sizes in permutations(self.params["nr_neurons"], r=nr_hidden_layers):
                for activation in self.params["activation"]:
                    for solver in self.params["solver"]:
                        for alpha in self.params["alpha"]:
                            print(f"Trying parameters: {hidden_layer_sizes}, {activation}, {solver}, {alpha}")
                            preprocessor = ColumnTransformer(
                                transformers=[
                                    ("scaler", StandardScaler(), X.select_dtypes(exclude="object").columns),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes(include="object").columns)
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
                            scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro")
                            if scores.mean() > self.best_score:
                                self.best_score = scores.mean()
                                self.best_params = {
                                    "hidden_layer_sizes": hidden_layer_sizes,
                                    "activation": activation,
                                    "solver": solver
                                }
                                self.best_model = pipe
                            
                                print(f"Found new best parameters: {hidden_layer_sizes}, {activation}, {solver}")

        end_time = time.time()
        print("Total time: ", end_time - start_time)