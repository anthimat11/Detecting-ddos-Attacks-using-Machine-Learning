####This code was used to balance the dataset by oversampling###

import pandas as pd
from imblearn.over_sampling import RandomOverSampler

folder_path = "Datasets/"

df = pd.read_csv( folder_path + 'LRHR_dataset.csv')

# Separate features and target variable
X = df.drop('Label', axis=1)  # Features
y = df['Label']  # Target variable (class labels)

# Initialize RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Fit and transform the data
X_resampled, y_resampled = ros.fit_resample(X, y)

# X_resampled and y_resampled now contain the oversampled data

# Convert oversampled data back to a DataFrame
df_oversampled = pd.DataFrame(data=X_resampled, columns=X.columns)
df_oversampled['label'] = y_resampled  # Add the target column back

#New balanced dataset saved as oversampled_datase
df_oversampled.to_csv( folder_path + 'balanced_dataset.csv', index=False)



