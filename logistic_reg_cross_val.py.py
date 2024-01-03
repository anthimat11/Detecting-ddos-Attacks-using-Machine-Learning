###This model is trained with the logistic regression model with kfold for cross validation###
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("final_dataset.csv")

X = data.drop('label', axis=1)  # Features
y = data['label']  # Target variable (class labels)

# Create a k-fold cross-validation object (k=5 for example)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create a logistic regression model
logreg_model = LogisticRegression()

# Perform cross-validation
cross_val_scores = cross_val_score(logreg_model, X, y, cv=kf, scoring='accuracy')

# Print cross-validation scores and average score
print("Cross-validation scores:", cross_val_scores)
print("Average accuracy:", cross_val_scores.mean())

# Perform cross-validation and generate predictions
predicted_labels = cross_val_predict(logreg_model, X, y, cv=kf)

# Generate and print the classification report
print(classification_report(y, predicted_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(y, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Initialize lists to store true labels and predicted probabilities
true_labels = []
predicted_probs = []

# Perform cross-validation and generate ROC curve
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    logreg_model.fit(X_train, y_train)
    predicted_probs.extend(logreg_model.predict_proba(X_test)[:, 1])
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
