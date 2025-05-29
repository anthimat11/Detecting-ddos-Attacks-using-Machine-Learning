##Train using Tabnet using Cross-Validation WITHOUT Scaler##

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import Callback
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

start_time = time.time()

#folder_path = "Dataset Modification/"
#data = pd.read_csv(folder_path + "final_dataset.csv")

#X = data.drop(['label', 'Protocol_ICMP', 'Protocol_UDP', 'Protocol_TCP'], axis=1).values
#y = data['label'].values

folder_path = "Datasets/"
data = pd.read_csv(folder_path + "LRHR_dataset.csv")

X = data.drop(['Label'], axis=1).values
y = data['Label'].values

# Standard scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Helper function define early stopping callback
class EarlyStopping(Callback):
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_0_logloss')
        if current_loss is None:
            return

        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

# K-Fold Cross-Validation (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies, cv_precisions, cv_recalls, cv_f1s, cv_aucs = [], [], [], [], []

##########CROSS VALIDATION##########
# Iterate over each fold
# First Training Loop (Cross-Validation)
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    #Tabnet Classifier
    model = TabNetClassifier(device_name=device, verbose=0)
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    
    model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    max_epochs=100,
    patience=5,  # Built-in early stopping
    eval_metric=['logloss']
)
    
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    
    cv_accuracies.append(acc)
    cv_precisions.append(prec)
    cv_recalls.append(rec)
    cv_f1s.append(f1)
    cv_aucs.append(auc)
    
    print(f"  Accuracy: {acc:.8f}")
    print(f"  Precision: {prec:.8f}")
    print(f"  Recall: {rec:.8f}")
    print(f"  F1-Score: {f1:.8f}")
    print(f"  AUC: {auc:.8f}")


print(f"Metrics (Cross-validation predictions):")
print(f"  Accuracy: {np.mean(cv_accuracies):.8f} (±{np.std(cv_accuracies):.8f})")
print(f"  Precision: {np.mean(cv_precisions):.8f} (±{np.std(cv_precisions):.8f})")
print(f"  Recall: {np.mean(cv_recalls):.8f} (±{np.std(cv_recalls):.8f})")
print(f"  F1-Score: {np.mean(cv_f1s):.8f} (±{np.std(cv_f1s):.8f})")
print(f"  AUC: {np.mean(cv_aucs):.8f} (±{np.std(cv_aucs):.8f})")

##########FINAL##########
# Final train-test split (80% train, 20% test)
# Second Training (Final Model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = TabNetClassifier(device_name=device, verbose=0)
early_stopping = EarlyStopping(patience=5, min_delta=0.01)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    max_epochs=100,
    patience=5,  # Built-in early stopping
    eval_metric=['logloss']
)

y_pred_test = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, y_pred_test)
test_conf_matrix = confusion_matrix(y_test, y_pred_test)

print(f"Metrics (Test Set):")
print(f"  Accuracy: {test_accuracy:.8f}")
print(f"  Precision: {test_precision:.8f}")
print(f"  Recall: {test_recall:.8f}")
print(f"  F1-Score: {test_f1:.8f}")
print(f"  AUC: {test_auc:.8f}")
print(f"Confusion Matrix:")
print(test_conf_matrix)


# Feature Importances
feature_importances = model.feature_importances_
print("Feature Importances:")
print(feature_importances)

end_time = time.time()
running_time = end_time - start_time
print(f"Running time: {running_time:.2f} seconds")