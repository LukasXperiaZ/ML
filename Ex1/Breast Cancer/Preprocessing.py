import pandas as pd

# Binary classification under class collumn
# ID not a feature
# No missing values
# All features numerical
def preprocess():
    breast_learn = pd.read_csv("../../datasets/BreastCancer/breast-cancer-diagnostic.shuf.lrn.csv")
    print(breast_learn.isnull().values.any())


if __name__ == "__main__":
    preprocess()