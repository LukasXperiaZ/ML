import time
import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def classify():
    # load train data
    loan_lrn = pd.read_csv("../../datasets/Loan/loan-10k.lrn.csv", engine='python')

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

    encoded_check = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    encoded_columns = encoded_check.fit_transform(x)
    sample_encoded = encoded_columns[0:5]

    test = 1

    # TODO: Resume at "The final part is separating the ..."


if __name__ == '__main__':
    classify()
