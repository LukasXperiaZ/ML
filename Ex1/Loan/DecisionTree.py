# Load libraries
import time

import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split, cross_val_score  # Import train_test_split function
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)  # Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus


# Note: Decision Trees do not require scaling/normalization
def classify():
    # load train data
    loan_lrn = pd.read_csv("../../datasets/Loan/loan-10k.lrn.csv", engine='python')

    # data exploration
    loan_lrn.info()

    # preprocessing
    features = loan_lrn.columns.to_list()
    del features[91]
    del features[0]
    x = loan_lrn[features]
    # do one-hot encoding for all categorical variables (sklearn cannot handle categorical variables),
    x = pd.get_dummies(x, drop_first=True)  # it would require some defined order for every categorical variable!
    features = x.columns.to_list()
    y = loan_lrn.grade

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)  # 70% train, 30% test

    # Create Decision Tree classifier Object
    # criterion: "gini" (gini index), "log_loss", "entropy" (information gain)
    clf = DecisionTreeClassifier(criterion="log_loss", max_depth=None)

    start = time.time()
    # Train Decision Tree Classifier
    clf = clf.fit(x_train, y_train)

    end = time.time()
    elapsed_time = end - start

    # Predict the response for the test dataset
    y_pred = clf.predict(x_test)

    # Model Accuracy, how often is the classifier correct?
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_pred, y_test, average="weighted")
    print("\nAccuracy: ", accuracy)
    print("Time: ", elapsed_time)
    print("F1 Score: ", f1)

    # plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig('DT_confusion_matrix.png')

    # Visualize the decision tree
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, feature_names=features, class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    if graph is not None:
        graph.write_png('./plots/loan_tree.png')
        Image(graph.create_png())

    # Cross validation scores
    start = time.time()
    scores = cross_val_score(clf, x, y, cv=5)
    end = time.time()
    elapsed_time = end - start
    print("Cross validation yielded %0.2f accuracy with a standard deviation of %0.2f in time %0.2f s" % (
            scores.mean(),
            scores.std(),
            elapsed_time))


if __name__ == '__main__':
    classify()
