import pandas as pd
import numpy as np

# Get column names from onehr.names
names = pd.read_csv('../../datasets/Ozone/edited_onehr.names',
                    engine='python',
                    names=['name'])
names['name'] = names['name'].str.split(':').str[0]
list_of_names = names['name'].tolist()

# Load Data
df = pd.read_csv('../../datasets/Ozone/onehr.data',
                 engine='python',
                 names=list_of_names,
                 parse_dates=['Date'])

# Recognized some missing values shown as '?'. Analyse frequency of '?' to decide how to deal with it.
missing_values_df = pd.DataFrame(columns=['Column_Name', 'Missing_inTotal', 'Missing_in%'])
length = len(df)
df_str = df.astype(str)

for i, column in enumerate(df_str.columns):
    count = (df_str[column] == '?').sum()
    missing_values_df.loc[i] = [column, count, round((count/length)*100, 1)]


# Based on the data structure (numerical hourly weather data) we first thought
# that linear interpolation would be a good choice to handle missing data.
# However, since there are specific dates with roughly 200 consecutive datapoints missing
# For that linear interpolation might not be the best option. Therefore, we decided to drop these datapoints.
df = df.replace('?', np.NaN)
print(f"Size before: {df.shape}")
print(f"Missing values before: {df.isna().sum().sum()}")

# Change datatypes of columns respectively
for col in df.columns:
    if col != 'Date':
        df[col] = pd.to_numeric(df[col])
df['Class'] = df['Class'].astype(bool)

# Try interpolation for as many as possible, maximum of 10 consecutive missing values
df = df.interpolate(method = "linear", limit = 10)
print(f"Missing values after interpolation: {df.isna().sum().sum()}")
df = df.dropna()
print(f"Size after dropping the rest: {df.shape}")

# Check if there are no remaining missing values
assert ((df == '?').sum().sum() == 0)
assert (df.isna().sum().sum() == 0)

df.to_pickle('../../datasets/Ozone/data_preprocessed.pkl')
