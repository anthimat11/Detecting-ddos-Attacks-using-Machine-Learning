import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

start_time = time.time()

#folder_path = "Dataset Modification/"
#data = pd.read_csv(folder_path + "final_dataset.csv")


folder_path = "Datasets/"
data = pd.read_csv(folder_path + "balanced_dataset.csv")


#X = data.drop(['label', 'Protocol_ICMP', 'Protocol_UDP','Protocol_TCP'], axis=1).values
#y = data['label'].values


X = data.drop(['label'], axis=1)  # Do not consider these columns
y = data['label']  # Target variable

# Helper functions convert NumPy arrays into PyTorch tensors
def to_tensor(data):  # converts feature data
    return torch.tensor(data, dtype=torch.float32).to(device)

def to_tensor_labels(labels):  # converts labels
    return torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

# Define Neural Network Model - 3 Fully Connected Layers
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # output 64 neurons
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)  # output 32 neurons
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)  # output 1 neuron
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # Forward Pass Function
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

# K-Fold Cross-Validation (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies, cv_precisions, cv_recalls, cv_f1s, cv_avg_losses = [], [], [], [], []

########## CROSS VALIDATION ##########
# Iterate over each fold
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Convert data to tensors
    X_train, X_val = to_tensor(X_train), to_tensor(X_val)
    y_train, y_val = to_tensor_labels(y_train), to_tensor_labels(y_val)

    model = NeuralNet(X.shape[1]).to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss Function
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Optimizer

    # First Training Loop (Cross-Validation)
    epoch_losses = []
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Save loss for averaging
        epoch_losses.append(loss.item())
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.6f}")
    
    # Calculate average loss for this fold
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    cv_avg_losses.append(avg_loss)

    # Evaluate cross-validation predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).cpu().numpy().round()
    
    acc = accuracy_score(y_val.cpu().numpy(), y_pred)
    prec = precision_score(y_val.cpu().numpy(), y_pred)
    rec = recall_score(y_val.cpu().numpy(), y_pred)
    f1 = f1_score(y_val.cpu().numpy(), y_pred)
    
    cv_accuracies.append(acc)
    cv_precisions.append(prec)
    cv_recalls.append(rec)
    cv_f1s.append(f1)

print(f"Metrics (Cross-validation predictions):")
print(f"  Accuracy: {sum(cv_accuracies)/len(cv_accuracies):.8f}")
print(f"  Precision: {sum(cv_precisions)/len(cv_precisions):.8f}")
print(f"  Recall: {sum(cv_recalls)/len(cv_recalls):.8f}")
print(f"  F1-Score: {sum(cv_f1s)/len(cv_f1s):.8f}")
print(f"  Average Loss: {sum(cv_avg_losses)/len(cv_avg_losses):.8f}")

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train, X_test = to_tensor(X_train), to_tensor(X_test)
y_train, y_test = to_tensor_labels(y_train), to_tensor_labels(y_test)

model = NeuralNet(X.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

########## FINAL ##########
# Second Training Loop (Final Model)
final_epoch_losses = []
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Save loss for averaging
    final_epoch_losses.append(loss.item())
    print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.6f}")

# Calculate average loss for the final model
avg_final_loss = sum(final_epoch_losses) / len(final_epoch_losses)

# Evaluate on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy().round()

test_accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)
test_precision = precision_score(y_test.cpu().numpy(), y_pred)
test_recall = recall_score(y_test.cpu().numpy(), y_pred)
test_f1 = f1_score(y_test.cpu().numpy(), y_pred)
test_conf_matrix = confusion_matrix(y_test.cpu().numpy(), y_pred)

print(f"Metrics (Test Set):")
print(f"  Accuracy: {test_accuracy:.8f}")
print(f"  Precision: {test_precision:.8f}")
print(f"  Recall: {test_recall:.8f}")
print(f"  F1-Score: {test_f1:.8f}")
print("Confusion Matrix:")
print(test_conf_matrix)
print(f"Average Loss for the Final Model: {avg_final_loss:.8f}")

end_time = time.time()
running_time = end_time - start_time
print(f"Running time: {running_time:.2f} seconds")
