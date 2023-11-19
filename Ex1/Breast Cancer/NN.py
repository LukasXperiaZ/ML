
#Imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import time
import pandas as pd

#Load the data
breast_data = pd.read_csv("../../datasets/BreastCancer/breast-cancer-diagnostic.shuf.lrn.csv")
#Drop ID
breast_data.pop("ID")
#Extract the label
X = breast_data.copy(deep=True)
y = X.pop("class")

#Split into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), X.columns)
    ]
)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64),
                    max_iter=100,
                    activation="tanh",
                    solver="lbfgs")
pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('mlpc', mlp)
])

start = time.time()
pipe.fit(X_train, y_train)
elapsed = time.time() - start

#Predicting and testing
y_pred = pipe.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
print(f"F1 score: {metrics.f1_score(y_test, y_pred, average='weighted')}")
print(f"Time to fit: {elapsed}")
cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
#disp.plot().figure_.savefig("Cancer_NN_CM.png")

# Cross-Validation
start = time.time()
scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro")
elapsed = time.time() - start
print(f"Average f1 score: {scores.mean()}")
print(f"Time for CV: {elapsed}")


