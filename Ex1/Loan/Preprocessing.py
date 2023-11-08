# Load libraries
import pandas as pd
from sklearn import preprocessing


# Observations:
# * no missing values
# * id not a feature
# * grade is the classification
def preprocess():
    loan_lrn = pd.read_csv("../../datasets/Loan/loan-10k.lrn.csv")
    print(loan_lrn.head())


if __name__ == '__main__':
    preprocess()
