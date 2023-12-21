# %%
import pandas as pd
import warnings

# %%
#Load the data
breast_data = pd.read_csv("../datasets/BreastCancer/breast-cancer-diagnostic.shuf.lrn.csv")
#Drop ID
breast_data.pop("ID")
#Extract the label
X = breast_data.copy(deep=True)
y = X.pop("class")

# # %%
# loan_data = pd.read_csv("../datasets/Loan/loan-10k.lrn.csv")
# loan_data.pop("ID")
# X = loan_data.copy(deep=True)
# y = X.pop("grade")

# %%
params = {
    "nr_hidden_layers": list(range(1, 11)),
    "nr_neurons": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "activation": ["tanh", "relu", "logistic", "identity"],
    "solver": ["adam", "lbfgs", "sgd"],
    "alpha": [0.00001, 0.0001, 0.001]
}

# %%
from grid_search import GridSearchMLP
gs = GridSearchMLP(params=params)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gs.find_params(X, y)

# %%
print(gs.best_params)


