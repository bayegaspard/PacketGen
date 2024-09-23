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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Ensure the 'results' directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# Define the classes to include (including 'Benign')
selected_classes = [
    'BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
    'Bot', 'FTP-Patator', 'Web Attack – Brute Force',
    'SSH-Patator', 'DoS slowloris'
]  # Added 'Benign' class

# Define the target attack class to be converted to 'Benign'
target_attack_class = 'Web Attack – Brute Force'  # You can change this to any other attack class from your list

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

# **Add maximum number of samples per class to handle imbalance**
# Define maximum number of samples per class
max_samples_per_class = 10000  # Adjust this value as needed

# Limit the number of samples per class
data = data.groupby('Label').apply(
    lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=42)
).reset_index(drop=True)

# **Print class distribution after limiting samples**
print("\nClass distribution after limiting samples per class:")
print(data['Label'].value_counts())

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['Label'])
num_classes = len(label_encoder.classes_)
print("\nEncoded Classes:", label_encoder.classes_)

# Identify the 'Benign' class index
benign_class = 'BENIGN'
if benign_class in label_encoder.classes_:
    benign_class_index = label_encoder.transform([benign_class])[0]
    print(f"'BENIGN' class index: {benign_class_index}")
else:
    raise ValueError("'Benign' class not found in the dataset.")

# Identify the target attack class index
if target_attack_class in label_encoder.classes_:
    target_attack_class_index = label_encoder.transform([target_attack_class])[0]
    print(f"Target Attack Class '{target_attack_class}' index: {target_attack_class_index}")
else:
    raise ValueError(f"Target Attack Class '{target_attack_class}' not found in the dataset.")

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

num_epochs = 5
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

# Convert y_pred to a NumPy array
y_pred = np.array(y_pred)

print('\nClassification Report (Original Data):')
print(
    classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
)

# 5. Implement the Diffusion Model for Adversarial Example Generation

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.noise_steps = noise_steps
        self.beta = self.prepare_noise_schedule(beta_start, beta_end).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)
    
        # Neural network layers for the diffusion model
        self.fc1 = nn.Linear(input_dim + 1, 256).to(device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, input_dim).to(device)
    
    def prepare_noise_schedule(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.noise_steps).to(device)
    
    def noise_data(self, x, t):
        batch_size = x.shape[0]
        t = t.view(-1)
        beta_t = self.beta[t].view(-1, 1).to(device)
        alpha_t = self.alpha_hat[t].view(-1, 1).to(device)
        noise = torch.randn_like(x).to(device)
        x_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return x_t, noise
    
    def forward(self, x, t):
        # Append normalized time step t as a scalar to the input
        t_normalized = t / self.noise_steps  # Normalize time step
        x_input = torch.cat([x, t_normalized.unsqueeze(1)], dim=1)  # Shape: (batch_size, input_dim + 1)
        x = self.fc1(x_input)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def generate_adversarial(self, x, t, target_class, classifier, steps=10, alpha=0.1):
        """
        Generates adversarial examples targeting a specific class.
        
        Parameters:
            x (torch.Tensor): Input samples.
            t (int): Current noise step.
            target_class (int): Index of the target class.
            classifier (nn.Module): Trained classification model.
            steps (int): Number of optimization steps.
            alpha (float): Learning rate for optimization.
        
        Returns:
            torch.Tensor: Adversarial examples.
        """
        self.eval()
        classifier.eval()
        
        # Initialize adversarial examples
        x_adv = x.clone().detach().requires_grad_(True).to(device)
        target = torch.full((x_adv.size(0),), target_class, dtype=torch.long).to(device)
        
        optimizer_adv = optim.Adam([x_adv], lr=alpha)
        criterion_adv = nn.CrossEntropyLoss()
        
        for _ in range(steps):
            optimizer_adv.zero_grad()
            outputs = classifier(x_adv)
            loss = criterion_adv(outputs, target)
            loss.backward()
            optimizer_adv.step()
        
        self.train()
        classifier.train()
        return x_adv.detach()

# Initialize and Train the Diffusion Model

# Initialize diffusion model
diffusion_model = DiffusionModel(input_size).to(device)

# Define optimizer for diffusion model
diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-3)

# Define loss function (Mean Squared Error)
diffusion_criterion = nn.MSELoss()

# Training the diffusion model
diffusion_epochs = 10  # Adjust as needed
for epoch in range(diffusion_epochs):
    diffusion_model.train()
    running_loss = 0.0
    for x_batch, _ in train_loader:
        batch_size = x_batch.shape[0]
        t = torch.randint(0, diffusion_model.noise_steps, (batch_size,), device=device).long()
        x_batch = x_batch.to(device)
        x_noisy, noise = diffusion_model.noise_data(x_batch, t)
        predicted_noise = diffusion_model(x_noisy, t)
        loss = diffusion_criterion(predicted_noise, noise)
        diffusion_optimizer.zero_grad()
        loss.backward()
        diffusion_optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Diffusion Model Epoch [{epoch+1}/{diffusion_epochs}], Loss: {avg_loss:.4f}")

# 6. Generate Adversarial Examples Using the Diffusion Model

# Define metrics to track at each time step
metrics = {
    "original": {"f1": [], "precision": [], "recall": []},
    "adversarial": {"f1": [], "precision": [], "recall": []}
}

# Define incremental steps (time steps)
incremental_steps = [0, 50, 100, 200, 400, 600, 800, 999]

def evaluate_metrics(y_true, y_pred, step, data_type):
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics[data_type]["f1"].append(f1)
    metrics[data_type]["precision"].append(precision)
    metrics[data_type]["recall"].append(recall)

    print(f"Step {step}: {data_type} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Evaluate metrics for original data (only once)
evaluate_metrics(y_test, y_pred, 0, "original")

# Identify samples belonging to the target attack class
target_attack_indices = np.where(y_test == target_attack_class_index)[0]
print(f"Number of samples in target attack class '{target_attack_class}': {len(target_attack_indices)}")

# Create a TensorDataset for target attack test samples
target_test_tensor = torch.tensor(X_test[target_attack_indices], dtype=torch.float32).to(device)
target_test_labels = torch.tensor(y_test[target_attack_indices], dtype=torch.long).to(device)
target_test_dataset = TensorDataset(target_test_tensor, target_test_labels)
target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size)

# Generate adversarial examples and evaluate
for step in incremental_steps:
    # Make a copy of y_pred as a NumPy array for adversarial predictions
    y_adv_pred = y_pred.copy()
    
    for data, labels in target_test_loader:
        data = data.to(device)
        # Generate adversarial examples targeting 'Benign' class
        x_adv = diffusion_model.generate_adversarial(
            data, step, benign_class_index, model, steps=10, alpha=0.1
        )
        # Get model predictions for adversarial examples
        outputs_adv = model(x_adv)
        _, preds_adv = torch.max(outputs_adv.data, 1)
        y_adv = preds_adv.cpu().numpy()
        
        # Determine the number of adversarial samples generated
        num_adv = len(y_adv)
        
        # Get the corresponding indices in the test set
        indices_to_replace = target_attack_indices[:num_adv]
        
        # Replace the predictions for target attack samples with adversarial predictions
        y_adv_pred[indices_to_replace] = y_adv
    
    # Evaluate metrics for adversarial examples
    evaluate_metrics(y_test, y_adv_pred, step, "adversarial")

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
    
    # Adjust original metrics to match the length of steps
    original_f1 = [metrics["original"]["f1"][0]] * len(steps)
    original_precision = [metrics["original"]["precision"][0]] * len(steps)
    original_recall = [metrics["original"]["recall"][0]] * len(steps)
    
    plt.plot(steps, original_f1, label="F1 (Original)", marker='o', color='blue')
    plt.plot(steps, metrics["adversarial"]["f1"], label="F1 (Adversarial)", marker='o', color='red')
    
    plt.plot(steps, original_precision, label="Precision (Original)", marker='s', color='blue', linestyle='dashed')
    plt.plot(steps, metrics["adversarial"]["precision"], label="Precision (Adversarial)", marker='s', color='red', linestyle='dashed')
    
    plt.plot(steps, original_recall, label="Recall (Original)", marker='^', color='blue', linestyle='dotted')
    plt.plot(steps, metrics["adversarial"]["recall"], label="Recall (Adversarial)", marker='^', color='red', linestyle='dotted')
    
    plt.xlabel("Time Steps")
    plt.ylabel("Metric Value")
    plt.title("Comparison of Metrics at Different Time Steps")
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
