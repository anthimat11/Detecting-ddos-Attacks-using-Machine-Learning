###This code trains a model with the Naive-Bayes Algorithm###

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


data = pd.read_csv("modified_dataset.csv")

X = data.drop('label', axis=1)  # Features
y = data['label']  # Target variable (class labels)

#Split dataset into train(80%) and test(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_train, y_train, X_test, y_test)

# Create and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Generate a detailed classification report
print("Classification Report:\n", classification_report(y_test, predictions))
