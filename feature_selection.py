import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("final_modified_dataset.csv")

X = data.drop('label', axis=1)  # Features
y = data['label']  # Target variable (class labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose the number of top features to select (k)
k = 10

# Initialize SelectKBest with the f_classif test
selector = SelectKBest(score_func=f_classif, k=k)

# Fit and transform the training data
X_train_selected = selector.fit_transform(X_train, y_train)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_features = X.columns[selected_feature_indices]

# Print the names of the selected features
print("Selected Features:", selected_features)

# Transform the test data based on the selected features
X_test_selected = selector.transform(X_test)

# Create a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model using the selected features
rf_model.fit(X_train_selected, y_train)

# Make predictions on the test data
predictions = rf_model.predict(X_test_selected)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
