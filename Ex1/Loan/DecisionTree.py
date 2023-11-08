# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function (not needed)
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation


# Note: Decision Trees do not require scaling/normalization
def classify():
    # load train data
    loan_lrn = pd.read_csv("../../datasets/Loan/loan-10k.lrn.csv")
    print(loan_lrn.head())
    column_names = loan_lrn.columns.to_list()

    loan_lrn = pd.read_csv("../../datasets/Loan/loan-10k.lrn.csv", skiprows=1, header=None,
                           names=column_names, engine='python')

    features = column_names.copy()
    del features[91]
    del features[0]
    x = loan_lrn[features]
    # do one-hot encoding for all categorical variables (sklearn cannot handle categorical variables),
    x = pd.get_dummies(x, drop_first=True)  # it would require some defined order for every categorical variable!

    y = loan_lrn.grade

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)  # 70% train, 30% test

    # Create Decision Tree classifier Object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifier
    clf = clf.fit(x_train, y_train)

    # Predict the response for the test dataset
    y_pred = clf.predict(x_test)

    # Model Accuracy, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(accuracy)


if __name__ == '__main__':
    classify()
