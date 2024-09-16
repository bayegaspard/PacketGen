# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
model = Net(input_size, num_classes).to(device)

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

        # Neural network layers for the diffusion model
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, input_dim)

    def prepare_noise_schedule(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def noise_data(self, x, t):
        # Convert `t` to a tensor if it is not already
        if isinstance(t, int):
            t = torch.tensor([t], device=device)  # Convert to a tensor with a single value
        t = t.long()  # Ensure `t` is of type `long`

        # Ensure `t` is within valid range
        if torch.any(t < 0) or torch.any(t >= self.noise_steps):
            raise ValueError(f"Invalid value for t: {t}. It should be in range [0, {self.noise_steps - 1}]")

        # Add an additional dimension for broadcasting
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).unsqueeze(1).to(device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).unsqueeze(1).to(device)
        
        # Make sure input tensors are on the same device
        x = x.to(device)
        noise = torch.randn_like(x).to(device)
        
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    def forward(self, x, t):
        # Forward pass through the diffusion model's neural network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def generate_adversarial(self, model, x, t):
        model.eval()
        with torch.no_grad():
            x_noisy, _ = self.noise_data(x, t)
            predicted_noise = model(x_noisy, t)
            alpha = self.alpha[t][:, None].to(device)
            alpha_hat = self.alpha_hat[t][:, None].to(device)
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

# Allow user to specify which classes to sample
selected_classes = ['Bot', 'DDoS' ,'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest','DoS slowloris' 'FTP-Patator', 'PortScan','SSH-Patator', 'Web Attack – Brute Force']  # Example classes; replace with your target classes

def filter_data_by_classes(X, y, label_encoder, selected_classes):
    """Filter dataset based on selected classes."""
    # Convert class names to class indices
    selected_indices = [label_encoder.transform([cls])[0] for cls in selected_classes if cls in label_encoder.classes_]
    
    # Filter data
    mask = np.isin(y.cpu().numpy(), selected_indices)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    return X_filtered, y_filtered

# Filtering the data
X_test_filtered, y_test_filtered = filter_data_by_classes(X_test_tensor, y_test_tensor, label_encoder, selected_classes)

# 7. Incremental Confusion Matrix Plotting

def plot_incremental_confusion_matrices(model, diffusion_model, X_test, y_test, label_encoder, steps, selected_classes=None):
    """Plot confusion matrices incrementally with increasing noise level for selected classes."""
    
    # If specific classes are selected, filter the dataset
    if selected_classes:
        X_test, y_test = filter_data_by_classes(X_test, y_test, label_encoder, selected_classes)
        print(f"Filtered to classes: {selected_classes}")

    for step in steps:
        # Generate valid random `t` values for noise addition
        t = torch.randint(low=0, high=diffusion_model.noise_steps, size=(X_test.size(0),)).to(device)

        noisy_data, _ = diffusion_model.noise_data(X_test, t)
        adv_dataset = TensorDataset(noisy_data, y_test)
        adv_loader = DataLoader(adv_dataset, batch_size=batch_size)

        y_noisy_pred = []
        model.eval()
        with torch.no_grad():
            for inputs_adv, _ in adv_loader:
                inputs_adv = inputs_adv.to(device)
                outputs_adv = model(inputs_adv)
                _, predicted_adv = torch.max(outputs_adv.data, 1)
                y_noisy_pred.extend(predicted_adv.cpu().numpy())

        # Plot confusion matrix
        cm = confusion_matrix(y_test.cpu(), y_noisy_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix at Noise Step {step} for Selected Classes')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/incremental_confusion_matrix_step_{step}.png')
        plt.show()

# Define incremental steps for noise addition
incremental_steps = [10, 50, 100, 200, 500, 1000]

# Plot confusion matrices for the selected classes incrementally
plot_incremental_confusion_matrices(model, diffusion_model, X_test_tensor, y_test_tensor, label_encoder, incremental_steps, selected_classes)

print("\nThe above results demonstrate how the model's performance changes when evaluated on adversarial examples generated using a diffusion model.")
