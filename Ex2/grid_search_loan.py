import pandas as pd
import warnings
import time

def main():
    loan_data = pd.read_csv("../datasets/Loan/loan-10k.lrn.csv")
    loan_data.pop("ID")
    X = loan_data.copy(deep=True)
    y = X.pop("grade")

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
        gs.find_params(X, y)
        end = time.time()

    print(f"Best params are {gs.best_params}")
    print(f"Best f1 is {gs.best_score}")
    print(f"Time elapsed: {end - start} seconds")

if __name__ == "__main__":
    main()


