import time
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

df = pd.read_pickle('../../datasets/Ozone/data_preprocessed.pkl')

X = df.drop(['Date','Class'], axis = 1)
y = df.Class

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Scale Data. Important for NN.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create Neural Network classifier object
mlp = MLPClassifier(hidden_layer_sizes= (70, 100, 70),
                    activation='tanh',
                    solver='adam')

start = time.time()
# Train Neural Network
mlp = mlp.fit(X_train,y_train)
end = time.time()
duration = end - start

# Predict response for test dataset
pred_test = mlp.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print(f"Duration: {round(duration,2)}s")
print(metrics.classification_report(y_test, pred_test))
cm = metrics.confusion_matrix(y_test, pred_test)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig('Confusion_matrix_Ozone_NN.png')

# Cross Validation
start = time.time()
scores = cross_val_score(mlp, X, y, cv=5)
end = time.time()
elapsed_time = end - start
print("Cross validation yielded %0.2f accuracy with a standard deviation of %0.2f in time %0.2f s" % (
    scores.mean(),
    scores.std(),
    elapsed_time))

# Since the class '1' is obviously underpresented we are trying in the following to countersteer
# by adding more observations of this class using the SMOTE method.
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=22)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

scaler = StandardScaler()
scaler.fit(X_train_res)
X_train_res = scaler.transform(X_train_res)
X_test = scaler.transform(X_test)

# Create Neural Network classifier object
mlp = MLPClassifier(hidden_layer_sizes= (70, 100, 70),
                    activation='tanh',
                    solver='adam')

start = time.time()
# Train Neural Network
mlp = mlp.fit(X_train_res,y_train_res)
end = time.time()
duration = end - start

# Predict response for test dataset
pred_test = mlp.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("---- Using SMOTE ----")
print(f"Duration: {round(duration,2)}s")
print(metrics.classification_report(y_test, pred_test))
cm = metrics.confusion_matrix(y_test, pred_test)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig('Confusion_matrix_Ozone_NN_resampled.png')

# Cross Validation
start = time.time()
scores = cross_val_score(mlp, X, y, cv=5)
end = time.time()
elapsed_time = end - start
print("Cross validation yielded %0.2f accuracy with a standard deviation of %0.2f in time %0.2f s" % (
    scores.mean(),
    scores.std(),
    elapsed_time))