import time
from scipy.io import arff
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

satimage_arff = arff.loadarff('../../datasets/Satimage/dataset_186_satimage.arff')
satimage = pd.DataFrame(satimage_arff[0])

'''
# data exploration
print(satimage.info())
print(satimage.head())
'''
# preprocessing
X = satimage.drop(['class'], axis = 1)
y_with_bytes = satimage['class']
y = y_with_bytes.astype(str)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Neural Network classifier object
mlp = MLPClassifier(hidden_layer_sizes= (100, 100, 10),
                    max_iter=300,
                    activation='tanh',
                    solver='lbfgs')

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
disp.plot().figure_.savefig('Confusion_matrix_Satimage_NN.png')

# Cross Validation
start = time.time()
scores = cross_val_score(mlp, X, y, cv=5)
end = time.time()
elapsed_time = end - start
print("Cross validation yielded %0.2f accuracy with a standard deviation of %0.2f in time %0.2f s" % (
    scores.mean(),
    scores.std(),
    elapsed_time))
