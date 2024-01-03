import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd

# Load your dataset from a CSV file
data = pd.read_csv("final_dataset.csv")

X = data.drop('label', axis=1)  # Features
y = data['label']  # Target variable 

# Create an SVM model with an RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale')  # 'scale' is a common choice for gamma, but you can experiment with other values

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print(f"Mean Accuracy: {np.mean(cv_scores)}")

# Fit the model on the entire dataset
model.fit(X, y)

# Make predictions using cross_val_predict for better prediction visualization
from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(model, X, y, cv=5)

# Print the classification report
report = classification_report(y, predictions)
print("Classification Report:")
print(report)



