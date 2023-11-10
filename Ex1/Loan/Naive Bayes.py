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
    y = loan_lrn.grade

    numeric_features = x.drop(
        ['term', 'emp_length', 'home_ownership', 'verification_status', 'loan_status',
         'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status', 'application_type',
         'hardship_flag', 'debt_settlement_flag', 'disbursement_method'],
        axis=1
    ).columns.to_list()

    categorical_features = ['term', 'emp_length', 'home_ownership', 'verification_status', 'loan_status',
                            'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status', 'application_type',
                            'hardship_flag', 'debt_settlement_flag', 'disbursement_method']

    # Label encode the categorical features (a label is a number)
    x_cat = x[categorical_features]
    le = LabelEncoder()
    x_cat_encoded = x_cat.apply(le.fit_transform)
    x_numeric_features = x[numeric_features]

    # We have to reset the indices to be able to concatenate
    x_cat_encoded.reset_index(drop=True, inplace=True)
    x_numeric_features.reset_index(drop=True, inplace=True)
    x = pd.concat([x_numeric_features, x_cat_encoded], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)  # 70% train, 30% test

    # Based on this: https://stackoverflow.com/questions/14254203/mixing-categorial-and-continuous-data-in-naive-bayes-classifier-using-scikit-lea
    # Build a Gaussian Classifier for continuous data
    model_cont = GaussianNB()
    # get the subset of the training set containing only continuous/numeric values
    x_train_cont = x_train[numeric_features]

    # Build a Categorical Classifier for categorical data
    model_cat = MultinomialNB()
    # get the subset of the training set containing only categorical values
    x_train_cat = x_train[categorical_features]

    start = time.time()
    # Model training continuous
    model_cont.fit(x_train_cont, y_train)
    # Model training categorical
    model_cat.fit(x_train_cat, y_train)

    end = time.time()
    elapsed_time = end - start

    # Predict Output
    # Split input data
    x_test_cont = x_test[numeric_features]
    x_test_cat = x_test[categorical_features]

    # Predict the log probabilities for both models
    prob_cont = model_cont.predict_log_proba(x_test_cont)
    prob_cat = model_cat.predict_log_proba(x_test_cat)

    new_features = np.hstack((prob_cat, prob_cont))

    new_model = GaussianNB()
    new_model.fit(new_features)

    y_pred = 1

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
