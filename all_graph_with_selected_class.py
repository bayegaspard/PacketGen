# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score, recall_score
)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure the 'results' directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# Define the classes to include
selected_classes = ['DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye']  # You can change this list as needed

# 1. Load and Preprocess Your Dataset

# Load your dataset
data = pd.read_csv('CICIDS2017_preprocessed.csv')  # Replace with your actual filename

# Drop unnecessary columns
data.drop(
    ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp'],
    axis=1,
    inplace=True
)

# Handle missing and infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Filter data to include only the selected classes
data = data[data['Label'].isin(selected_classes)]

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['Label'])
num_classes = len(label_encoder.classes_)

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
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

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

num_epochs = 2
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

# 4. Evaluate the Model on Test Data (Original Non-Adversarial)

model.eval()
y_pred = []
with torch.no_grad():
    for inputs_test, labels_test in test_loader:
        inputs_test = inputs_test.to(device)
        outputs_test = model(inputs_test)
        _, predicted = torch.max(outputs_test.data, 1)
        y_pred.extend(predicted.cpu().numpy())

print('\nClassification Report (Original Data):')
print(
    classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
)

# 5. Diffusion Model for Adversarial Example Generation

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.noise_steps = noise_steps
        self.beta = self.prepare_noise_schedule(beta_start, beta_end).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)

        # Neural network layers for the diffusion model
        self.fc1 = nn.Linear(input_dim, 256).to(device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, input_dim).to(device)

    def prepare_noise_schedule(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.noise_steps).to(device)

    def noise_data(self, x, t):
        if isinstance(t, int):
            t = torch.tensor([t], device=device)
        t = t.long().to(device)

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).unsqueeze(1).to(device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).unsqueeze(1).to(device)

        x = x.to(device)
        noise = torch.randn_like(x).to(device)

        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    def forward(self, x, t):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def generate_adversarial(self, x, t):
        self.eval()
        with torch.no_grad():
            if isinstance(t, int):
                t = torch.tensor([t], device=device)
            t = t.long().to(device)
            t = torch.clamp(t, max=self.noise_steps - 1)  # Ensure t is within valid range
            x_noisy, _ = self.noise_data(x.to(device), t)
            predicted_noise = self(x_noisy.to(device), t)
            alpha = self.alpha[t][:, None].to(device)
            alpha_hat = self.alpha_hat[t][:, None].to(device)
            x_adv = (1 / torch.sqrt(alpha)) * (
                x_noisy - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            )
        self.train()
        return x_adv

# Load a pre-trained diffusion model if available
def load_diffusion_model(path, input_dim):
    model = DiffusionModel(input_dim).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        logging.info(f"Loaded diffusion model from {path}")
    else:
        logging.warning(f"Diffusion model path {path} does not exist.")
    return model

# 6. Generate Adversarial Examples Using the Diffusion Model

diffusion_model_path = 'models/diffusion_model.pt'  # Adjust as needed
diffusion_model = load_diffusion_model(diffusion_model_path, input_size)

# Define metrics to track at each time step
metrics = {
    "original": {"f1": [], "precision": [], "recall": []},
    "adversarial": {"f1": [], "precision": [], "recall": []}
}

# Adjust incremental_steps to avoid t = noise_steps
incremental_steps = [0, 50, 100, 200, 400, 600, 800, 999]

def evaluate_metrics(y_true, y_pred, step, data_type):
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics[data_type]["f1"].append(f1)
    metrics[data_type]["precision"].append(precision)
    metrics[data_type]["recall"].append(recall)

    print(f"Step {step}: {data_type} - F1: {f1}, Precision: {precision}, Recall: {recall}")

# Generate adversarial examples and evaluate
for step in incremental_steps:
    y_adv_pred = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            x_adv = diffusion_model.generate_adversarial(data, step)
            outputs_adv = model(x_adv.to(device))
            _, preds_adv = torch.max(outputs_adv.data, 1)
            y_adv_pred.extend(preds_adv.cpu().numpy())

    # Evaluate metrics for adversarial examples
    evaluate_metrics(y_test, y_adv_pred, step, "adversarial")

    # Compare original vs adversarial at each step
    evaluate_metrics(y_test, y_pred, step, "original")

    # Plot Confusion Matrices for Original and Adversarial Data
    def plot_confusion_matrices(y_true, y_pred_orig, y_pred_adv, classes, step, filename):
        # Compute confusion matrices
        cm_orig = confusion_matrix(y_true, y_pred_orig)
        cm_adv = confusion_matrix(y_true, y_pred_adv)

        # Plot side by side
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=axes[0], annot_kws={"size": 8})
        axes[0].set_title(f'Original Data - Step {step}')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=45)

        sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=axes[1], annot_kws={"size": 8})
        axes[1].set_title(f'Adversarial Data - Step {step}')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=45)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_confusion_matrices(
        y_test, y_pred, y_adv_pred, label_encoder.classes_,
        step, f'results/confusion_matrices_step_{step}.png'
    )

# Plot metrics over time
def plot_metrics(metrics, steps):
    plt.figure(figsize=(12, 8))
    
    plt.plot(steps, metrics["original"]["f1"], label="F1 (Original)", marker='o', color='blue')
    plt.plot(steps, metrics["adversarial"]["f1"], label="F1 (Adversarial)", marker='o', color='red')
    
    plt.plot(steps, metrics["original"]["precision"], label="Precision (Original)", marker='s', color='blue', linestyle='dashed')
    plt.plot(steps, metrics["adversarial"]["precision"], label="Precision (Adversarial)", marker='s', color='red', linestyle='dashed')
    
    plt.plot(steps, metrics["original"]["recall"], label="Recall (Original)", marker='^', color='blue', linestyle='dotted')
    plt.plot(steps, metrics["adversarial"]["recall"], label="Recall (Adversarial)", marker='^', color='red', linestyle='dotted')
    
    plt.xlabel("Time Steps")
    plt.ylabel("Metric Value")
    plt.title("Comparison of F1, Precision, and Recall at Different Time Steps")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("results/metrics_comparison.png")
    plt.close()

plot_metrics(metrics, incremental_steps)

# 7. Plot Alpha and Beta Values over Noise Steps

def plot_alpha_beta(diffusion_model):
    steps = np.arange(diffusion_model.noise_steps)
    beta = diffusion_model.beta.cpu().numpy()
    alpha = diffusion_model.alpha.cpu().numpy()
    alpha_hat = diffusion_model.alpha_hat.cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.plot(steps, beta, label='Beta', color='red')
    plt.plot(steps, alpha, label='Alpha', color='blue')
    plt.plot(steps, alpha_hat, label='Alpha Hat (Cumulative Product of Alpha)', color='green')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('Alpha, Beta, and Alpha Hat over Time Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/alpha_beta_plot.png')
    plt.close()

plot_alpha_beta(diffusion_model)

# 8. Plot Amount of Noise Added per Step

def plot_noise_amount(diffusion_model):
    noise_levels = np.sqrt(1 - diffusion_model.alpha_hat.cpu().numpy())
    steps = np.arange(diffusion_model.noise_steps)

    plt.figure(figsize=(12, 8))
    plt.plot(steps, noise_levels, label='Amount of Noise Added', color='purple')
    plt.xlabel('Time Steps')
    plt.ylabel('Noise Level (Standard Deviation)')
    plt.title('Amount of Noise Added per Time Step')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/noise_amount_plot.png')
    plt.close()

plot_noise_amount(diffusion_model)
