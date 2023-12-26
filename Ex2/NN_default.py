import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.io import arff

#Satimage
sat_data = pd.DataFrame(arff.loadarff('../datasets/Satimage/dataset_186_satimage.arff')[0])
X_sat = sat_data.copy(deep=True)
y_sat = X_sat.pop("class").astype(str)

#Breast Cancer
breast_data = pd.read_csv("../datasets/BreastCancer/breast-cancer-diagnostic.shuf.lrn.csv")
breast_data.pop("ID")
X_breast = breast_data.copy(deep=True)
y_breast = X_breast.pop("class")

#Loan
loan_data = pd.read_csv("../datasets/Loan/loan-10k.lrn.csv")
loan_data.pop("ID")
X_loan = loan_data.copy(deep=True)
y_loan = X_loan.pop("grade")

def perform_cross_validation(X, y):
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"),
             X.select_dtypes(include="object").columns),
            ("scale", StandardScaler(),
             X.select_dtypes(include="number").columns)
        ]
    )
    clf = MLPClassifier()
    pipe = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', clf)
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro")
    return scores

#Cross Validation
scores_sat = perform_cross_validation(X_sat, y_sat)
scores_breast = perform_cross_validation(X_breast, y_breast)
scores_loan = perform_cross_validation(X_loan, y_loan)

#Print Results
print("Satimage")
print(scores_sat.mean())
print("Breast Cancer")
print(scores_breast.mean())
print("Loan")
print(scores_loan.mean())
