#Imports
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import time
from scipy.io import arff

breast_data = pd.DataFrame(arff.loadarff('../../datasets/Satimage/dataset_186_satimage.arff')[0])
#Extract the label
X = breast_data.copy(deep=True)
y = X.pop("class").astype(str)

#Split into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = GaussianNB()

#Training
start = time.time()
clf = clf.fit(X_train, y_train)
elapsed = time.time() - start

#Prediction
y_pred = clf.predict(X_test)

# Model metric
print(metrics.classification_report(y_test, y_pred))
print(f"Time for fitting: {elapsed}")
cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig("Satimage_NB_CM.png")

#Cross-Validation
start = time.time()
scores = cross_val_score(clf, X, y, cv=5, scoring="f1_macro")
elapsed = time.time() - start
print(f"F1 avg from CV: {scores.mean()}")
print(f"Time needed: {elapsed}")


