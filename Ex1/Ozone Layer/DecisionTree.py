import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
# for plotting the DT
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

df = pd.read_pickle('../../datasets/Ozone/data_preprocessed.pkl')

X = df.drop(['Date','Class'], axis = 1)
y = df.Class

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier(criterion = 'gini')

start = time.time()
# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)
end = time.time()
duration = end - start

# Predict response for training and test dataset
pred_test = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print(f"Duration: {round(duration,2)}s")
print(metrics.classification_report(y_test, pred_test))
cm = metrics.confusion_matrix(y_test, pred_test)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig('Confusion_matrix_Ozone_DT.png')

#Cross Validation
start = time.time()
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
elapsed_time = end - start
print("Cross validation yielded %0.2f accuracy with a standard deviation of %0.2f in time %0.2f s" % (
    scores.mean(),
    scores.std(),
    elapsed_time))

# Plot decision tree
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = X.columns.to_list(),class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('DecisionTree_Ozone.png')
Image(graph.create_png())

# Since the class '1' is obviously underpresented we are trying in the following to countersteer
# by adding more observations of this class using the SMOTE method.
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=22)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

start = time.time()
# Train Decision Tree Classifier
clf = clf.fit(X_train_res,y_train_res)
end = time.time()
duration = end - start

# Predict response for training and test dataset
pred_test = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print(f"Duration: {round(duration,2)}s")
print(metrics.classification_report(y_test, pred_test))
cm = metrics.confusion_matrix(y_test, pred_test)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig('Confusion_matrix_Ozone_DT_resampled.png')

# Cross Validation
start = time.time()
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
elapsed_time = end - start
print("Cross validation yielded %0.2f accuracy with a standard deviation of %0.2f in time %0.2f s" % (
    scores.mean(),
    scores.std(),
    elapsed_time))
