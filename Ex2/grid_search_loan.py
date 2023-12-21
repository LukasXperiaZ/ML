import pandas as pd
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def main():
    loan_data = pd.read_csv("../datasets/Loan/loan-10k.lrn.csv")
    loan_data.pop("ID")
    X = loan_data.copy(deep=True)
    y = X.pop("grade")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1337)

    params = {
        "nr_hidden_layers": list(range(1, 6)),
        "nr_neurons": [1, 2, 3, 4, 5, 10],
        "activation": ["tanh", "relu", "logistic", "identity"],
        "solver": ["adam", "lbfgs", "sgd"],
        "alpha": [0.00001, 0.0001, 0.001]
    }

    from grid_search import GridSearchMLP
    gs = GridSearchMLP(params=params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.time()
        gs.find_params(X=X_train, y=y_train, X_test=X_test, y_test=y_test)
        end = time.time()

    print(f"Best params are {gs.best_params}")
    print(f"Best f1 is {gs.best_score}")
    print(f"Time elapsed: {end - start} seconds")
    scores = cross_val_score(gs.best_model, X, y, cv=5, scoring="f1_macro")
    print(f"Cross val score of the best classifier: {scores.mean()}")


if __name__ == "__main__":
    main()


