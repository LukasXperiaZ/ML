#Imports
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import time

breast_data = pd.read_csv("../../datasets/BreastCancer/breast-cancer-diagnostic.shuf.lrn.csv")
#Drop ID
breast_data.pop("ID")
#Extract the label
X = breast_data.copy(deep=True)
y = X.pop("class")

#Define preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("scaler", StandardScaler(), X.columns)
])

#Split into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Building the classifier
clf = DecisionTreeClassifier(criterion="gini", max_depth=None)

#Building pipeline
pipe = Pipeline(steps=[
#    ("pre", preprocessor),
    ("clf", clf)
])

#Training
start = time.time()
pipe.fit(X_train, y_train)
elapsed = time.time() - start

#Predict
y_pred = pipe.predict(X_test)

# Model Accuracy
print(metrics.classification_report(y_test, y_pred))
print(f"Time for fitting: {elapsed}")
cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig("Cancer_DT_CM.png")

start = time.time()
scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro")
elapsed = time.time() - start
print(f"F1 avg from CV: {scores.mean()}")
print(f"Time needed: {elapsed}")


