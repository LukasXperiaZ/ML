import time
import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)  # Import scikit-learn metrics module for accuracy calculation


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

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    numeric_features = x.drop(
        ['term', 'emp_length', 'home_ownership', 'verification_status', 'loan_status',
         'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status', 'application_type',
         'hardship_flag', 'debt_settlement_flag', 'disbursement_method'],
        axis=1
    ).columns

    # --- This is just to check whether the scaling worked
    scaled_check = ColumnTransformer(
        transformers=[
            ('num ', numeric_transformer, numeric_features)
        ]
    )

    scaled_columns = scaled_check.fit_transform(x)
    sample_scaled = scaled_columns[0:5]
    # --- End of the check

    # Do One-Hot-Encoding to handle categorical data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    categorical_features = x.select_dtypes(include=['object']).columns

    # --- This is just to check whether the encoding worked
    encoded_check = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    encoded_columns = encoded_check.fit_transform(x)
    sample_encoded = encoded_columns[0:5]
    # --- End of check

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)  # 70% train, 30% test

    # Make a transformer that does the preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    start = time.time()
    # train the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10),
                        max_iter=300,
                        activation='tanh',
                        solver='adam')

    pipe = Pipeline(steps=[
        ('pre', preprocessor),
        ('mlpc', mlp)
    ])

    pipe.fit(x_train, y_train)
    end = time.time()
    elapsed_time = end - start

    # test the MLP
    y_pred = pipe.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)   # compute the accuracy
    f1 = f1_score(y_pred, y_test, average="weighted")
    print("\nAccuracy: " + str(accuracy))
    print("Time: " + str(elapsed_time))
    print("F1 Score: ", f1)

    # plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig('NN-MLP_confusion_matrix.png')


if __name__ == '__main__':
    classify()
