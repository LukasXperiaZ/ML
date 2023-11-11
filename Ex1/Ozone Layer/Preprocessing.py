import pandas as pd
import numpy as np

# Get column names from onehr.names in order to name columns respectively.
names = pd.read_csv("onehr.names",
                    names=['name'])
names['name'] = names['name'].str.split(':').str[0]
list_of_names = names['name'].tolist()

# Load Data
df = pd.read_csv('onehr.data',
                 names=list_of_names,
                 parse_dates=['Date'])

# Recognized some missing values in terms of '?'. Analyse the frequency of '?' to decide how to deal with it.
missing_values_df = pd.DataFrame(columns=['Column_Name', 'Missing_inTotal', 'Missing_in%'])
length = len(df)
df_str = df.astype(str)

for i, column in enumerate(df_str.columns):
    count = (df_str[column] == '?').sum()
    missing_values_df.loc[i] = [column, count, round((count/length)*100, 1)]

# Based on the data structure (numerical hourly weather data) we first thought
# that linear interpolation would be a good choice to handle missing data.
# However, since there are roughly 200 consecutive datapoints missing
# linear interpolation as well as ffill makes no sense. Therefore, we decided to drop the missing data.
df = df.replace('?', np.NaN)
df = df.dropna()

# Change datatypes of columns respectively
for col in df.columns:
    if col != 'Date':
        df[col] = pd.to_numeric(df[col])

df['Class'] = df['Class'].astype(bool)

# Check if there are no remaining missing values
assert ((df == '?').sum().sum() == 0)
assert (df.isna().sum().sum() == 0)

df.to_pickle('data_preprocessed.pkl')
