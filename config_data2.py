###This code is used to convert non-numeric values into numeric and replace NaN values###
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

df = pd.read_csv('updated_dataset.csv')

columns_with_missing_values = df.columns[df.isnull().any()].tolist()

# Create a SimpleImputer instance with the desired strategy ('mean', 'median', 'most_frequent', or 'constant')
imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy if needed

# Select columns with missing values and transform the data
df[columns_with_missing_values] = imputer.fit_transform(df[columns_with_missing_values])


##New dataset saved in modified_dataset.csv
df.to_csv("final_dataset.csv", index=False)