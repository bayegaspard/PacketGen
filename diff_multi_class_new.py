# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 1. Load and Preprocess Your Dataset

# Load your dataset
data = pd.read_csv('CICIDS2017_preprocessed.csv')  # Replace with your actual filename

# Display the first few rows (optional)
print("First few rows of the dataset:")
print(data.head())

# Drop unnecessary columns
data.drop(['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp'], axis=1, inplace=True)

# Handle missing and infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Check class distribution
print("\nClass distribution:")
print(data['Label'].value_counts())

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['Label'])
num_classes = len(label_encoder.classes_)
print("\nClasses found:", label_encoder.classes_)

# Convert features to numeric
features = data.drop('Label', axis=1)
features = features.apply(pd.to_numeric, errors='coerce')
features.fillna(0, inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Split into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, labels, test_size=0.3, random_state=42, stratify=labels
)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDatasets and DataLoaders
batch_size = 256

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 2. Build a Deep Learning Classification Model with PyTorch

# Define the model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# Instantiate the model
input_size = X_train.shape[1]
model = Net(input_size, num_classes)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Train the Model

num_epochs = 20
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels_batch in train_loader:
        inputs, labels_batch = inputs.to(device), labels_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs_val, labels_val in test_loader:
            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
            outputs_val = model(inputs_val)
            _, predicted = torch.max(outputs_val.data, 1)
            total += labels_val.size(0)
            correct += (predicted == labels_val).sum().item()
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# 4. Evaluate the Model on Test Data

model.eval()
y_pred = []
with torch.no_grad():
    for inputs_test, _ in test_loader:
        inputs_test = inputs_test.to(device)
        outputs_test = model(inputs_test)
        _, predicted = torch.max(outputs_test.data, 1)
        y_pred.extend(predicted.cpu().numpy())

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 5. Diffusion Model for Adversarial Example Generation

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.noise_steps = noise_steps
        self.beta = self.prepare_noise_schedule(beta_start, beta_end).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def noise_data(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        noise = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    def generate_adversarial(self, model, x, t):
        model.eval()
        with torch.no_grad():
            x_noisy, _ = self.noise_data(x, t)
            predicted_noise = model(x_noisy, t)
            alpha = self.alpha[t][:, None]
            alpha_hat = self.alpha_hat[t][:, None]
            x_adv = (1 / torch.sqrt(alpha)) * (
                x_noisy - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            )
        model.train()
        return x_adv

# Load a pre-trained diffusion model if available
def load_diffusion_model(path, input_dim):
    model = DiffusionModel(input_dim)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        logging.info(f"Loaded diffusion model from {path}")
    else:
        logging.warning(f"Diffusion model path {path} does not exist.")
    return model

# 6. Generate Adversarial Examples Using the Diffusion Model

diffusion_model_path = 'models/diffusion_model.pt'  # Adjust as needed
diffusion_model = load_diffusion_model(diffusion_model_path, input_size).to(device)

adversarial_examples = []
true_labels = []

with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)
        t = torch.randint(low=1, high=diffusion_model.noise_steps, size=(data.size(0),)).to(device)
        x_adv = diffusion_model.generate_adversarial(diffusion_model, data, t)
        adversarial_examples.append(x_adv.cpu())
        true_labels.append(labels.cpu())

adversarial_examples = torch.cat(adversarial_examples)
true_labels = torch.cat(true_labels)

# Save adversarial samples for future evaluation
torch.save({'adversarial_examples': adversarial_examples, 'true_labels': true_labels}, 'adversarial_samples.pt')

# 7. Evaluate the Model on Adversarial Examples

adv_dataset = TensorDataset(adversarial_examples, y_test_tensor)
adv_loader = DataLoader(adv_dataset, batch_size=batch_size)

model.eval()
y_adv_pred = []
with torch.no_grad():
    for inputs_adv, _ in adv_loader:
        inputs_adv = inputs_adv.to(device)
        outputs_adv = model(inputs_adv)
        _, predicted_adv = torch.max(outputs_adv.data, 1)
        y_adv_pred.extend(predicted_adv.cpu().numpy())

print('\nAdversarial Classification Report:')
print(classification_report(y_test, y_adv_pred, target_names=label_encoder.classes_))

# Calculate Accuracy Drop
original_accuracy = val_accuracies[-1]
adv_correct = sum(np.array(y_adv_pred) == y_test)
adv_accuracy = adv_correct / len(y_test)
accuracy_drop = original_accuracy - adv_accuracy
print(f'Accuracy Drop due to Adversarial Attack: {accuracy_drop:.4f}')

# 8. Visualize the Results

# Plot Training Loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Confusion Matrix for Original Test Data
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Original Test Data')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

# Plot Confusion Matrix for Adversarial Test Data
cm_adv = confusion_matrix(y_test, y_adv_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Reds', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Adversarial Test Data')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

# 9. Conclusion

print("\nThe above results demonstrate how the model's performance changes when evaluated on adversarial examples generated using a diffusion model.")
