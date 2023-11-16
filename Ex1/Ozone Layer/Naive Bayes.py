import time
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

df = pd.read_pickle('../../datasets/Ozone/data_preprocessed.pkl')

X = df.drop(['Date','Class'], axis = 1)
y = df.Class

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Naive Bayes classifier object
model = GaussianNB(priors=[0.995,0.005])

start = time.time()
# Train Naive Bayes Classifier
model = model.fit(X_train,y_train)
end = time.time()
duration = end - start

# Predict response for training and test dataset
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print(f"Duration: {round(duration,4)}s")
print(metrics.classification_report(y_test, pred_test))
cm = metrics.confusion_matrix(y_test, pred_test)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig('Confusion_matrix_Ozone_NB.png')

# Cross Validation
start = time.time()
scores = cross_val_score(model, X, y, cv=10)
end = time.time()
elapsed_time = end - start
print("Cross validation yielded %0.2f accuracy with a standard deviation of %0.2f in time %0.2f s" % (
    scores.mean(),
    scores.std(),
    elapsed_time))