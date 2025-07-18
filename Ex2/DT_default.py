import time

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from scipy.io import arff

#Satimage
sat_data = pd.DataFrame(arff.loadarff('../datasets/Satimage/dataset_186_satimage.arff')[0])
print(sat_data.shape)
X_sat = sat_data.copy(deep=True)
y_sat = X_sat.pop("class").astype(str)

#Breast Cancer
breast_data = pd.read_csv("../datasets/BreastCancer/breast-cancer-diagnostic.shuf.lrn.csv")
print(breast_data.shape)
breast_data.pop("ID")
X_breast = breast_data.copy(deep=True)
y_breast = X_breast.pop("class")

#Loan
loan_data = pd.read_csv("../datasets/Loan/loan-10k.lrn.csv")
print(loan_data.shape)
loan_data.pop("ID")
X_loan = loan_data.copy(deep=True)
y_loan = X_loan.pop("grade")

#Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"),
         X_loan.select_dtypes(include="object").columns)
    ]
)

#Classifiers
clf_sat = DecisionTreeClassifier()
clf_breast = DecisionTreeClassifier()
clf_loan = Pipeline(steps=[
    ('pre', preprocessor),
    ('dtc', DecisionTreeClassifier())
])

#Cross Validation
start_sat = time.time()
scores_sat = cross_val_score(clf_sat, X_sat, y_sat, cv=5, scoring="f1_macro")
end_sat = time.time()
start_breast = time.time()
scores_breast = cross_val_score(clf_breast, X_breast, y_breast, cv=5, scoring="f1_macro")
end_breast = time.time()
start_loan = time.time()
scores_loan = cross_val_score(clf_loan, X_loan, y_loan, cv=5, scoring="f1_macro")
end_loan = time.time()

#Print Results
print("Satimage")
print(scores_sat.mean())
print(end_sat - start_sat)
print("Breast Cancer")
print(scores_breast.mean())
print(end_breast - start_breast)
print("Loan")
print(scores_loan.mean())
print(end_loan - start_loan)
