###Decision Trees Algorithm with cross validation
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import time

start_time = time.time()

data = pd.read_csv("final_dataset.csv")

X = data.drop('label', axis=1)  # Features
y = data['label']  # Target variable 

#k-fold cross-validation object (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier()

# Perform cross-validation
cross_val_scores = cross_val_score(decision_tree_model, X, y, cv=kf, scoring='accuracy')

# Print cross-validation scores and average score
print("Cross-validation scores:", cross_val_scores)
print("Average accuracy:", cross_val_scores.mean())


# Perform cross-validation and generate predictions
predicted_labels = cross_val_predict(decision_tree_model, X, y, cv=kf)

# Generate and print the classification report
print(classification_report(y, predicted_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(y, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Perform cross-validation predictions for ROC curve
true_labels = []
predicted_probs = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    decision_tree_model.fit(X_train, y_train)
    predicted_probs.extend(decision_tree_model.predict_proba(X_test)[:, 1])
    true_labels.extend(y_test)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Perform cross-validation predictions for classification report
true_labels = []
predicted_labels = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    decision_tree_model.fit(X_train, y_train)
    predicted_labels.extend(decision_tree_model.predict(X_test))
    true_labels.extend(y_test)


# Train the model on the entire dataset
decision_tree_model.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree_model, filled=True, feature_names=X.columns, class_names=True)
plt.show()


end_time = time.time()
running_time = end_time - start_time
print(f"Running time: {running_time} seconds")