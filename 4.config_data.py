###This code is used to convert non-numeric values into numeric and replace NaN values###
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

folder_path = "Dataset Modification/"

df = pd.read_csv(folder_path + 'modified_dataset.csv')

columns_with_missing_values = df.columns[df.isnull().any()].tolist()

# Create a SimpleImputer instance with the desired strategy ('mean', 'median', 'most_frequent', or 'constant')
imputer = SimpleImputer(strategy='mean')  

# Select columns with missing values and transform the data
df[columns_with_missing_values] = imputer.fit_transform(df[columns_with_missing_values])

columns_to_delete = ['src', 'dst']  

# Drop the specified columns
df.drop(columns=columns_to_delete, inplace=True)


##New dataset saved in final_dataset
df.to_csv(folder_path + "final_dataset.csv", index=False)