##Train using Decision Tree model using Cross-Validation WITHOUT Scaler##

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time

start_time = time.time()

#folder_path = "Dataset Modification/"
#data = pd.read_csv(folder_path + "final_dataset.csv")

folder_path = "Datasets/"
data = pd.read_csv(folder_path + "balanced_dataset.csv")

#X = data.drop(['label', 'Protocol_ICMP', 'Protocol_UDP','Protocol_TCP'], axis=1)  # Do not consider these columns
#y = data['label']  # Target variable

X = data.drop(['label'], axis=1)  # Do not consider these columns
y = data['label']  # Target variable

# Set up K-Fold Cross-Validation (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree Classifier
model = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth=10, 
    min_samples_leaf=5, 
    min_samples_split=10,
    ccp_alpha=0.01
)

# Perform cross-validation
cross_val_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("Cross-validation scores:", cross_val_scores)
print(f"Average accuracy (cross-validation): {cross_val_scores.mean():.8f}")

# Perform cross-validation predictions for entire dataset
predicted_labels = cross_val_predict(model, X, y, cv=kf)

# Evaluate cross-validation predictions
cv_accuracy = accuracy_score(y, predicted_labels)
cv_precision = precision_score(y, predicted_labels)
cv_recall = recall_score(y, predicted_labels)
cv_f1 = f1_score(y, predicted_labels)

print(f"Metrics (Cross-validation predictions):")
print(f"  Accuracy: {cv_accuracy:.8f}")
print(f"  Precision: {cv_precision:.8f}")
print(f"  Recall: {cv_recall:.8f}")
print(f"  F1-Score: {cv_f1:.8f}")

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model 
model.fit(X_train, y_train)

# Evaluate on the test set
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)
conf_matrix = confusion_matrix(y_test, test_pred)

print(f"Metrics (Test Set):")
print(f"  Accuracy: {test_accuracy:.8f}")
print(f"  Precision: {test_precision:.8f}")
print(f"  Recall: {test_recall:.8f}")
print(f"  F1-Score: {test_f1:.8f}")
print("Confusion Matrix:")
print(conf_matrix)

# feature_importances = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': model.feature_importances_
# }).sort_values(by='Importance', ascending=False)
# print(feature_importances)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=True)
plt.show()


end_time = time.time()
running_time = end_time - start_time
print(f"Running time: {running_time:.2f} seconds")
