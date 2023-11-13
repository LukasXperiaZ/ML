import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


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

    # Create Naive Bayes classifier
    model = GaussianNB(var_smoothing=1e-12)

    start = time.time()

    # Train the model
    model.fit(x_train, y_train)

    end = time.time()
    elapsed_time = end - start

    y_pred = model.predict(x_test)

    # Model accuracy
    accuracy = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="weighted")

    print("\nAccuracy: ", accuracy)
    print("Time: ", elapsed_time)
    print("F1 Score: ", f1)

    # plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig('NB_confusion_matrix.png')


if __name__ == '__main__':
    classify()
