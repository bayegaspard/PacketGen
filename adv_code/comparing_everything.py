# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score, recall_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Suppress specific warnings (optional)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Ensure the 'results' directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# Define the classes to include
selected_classes = [
    'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
    'Bot', 'FTP-Patator', 'Web Attack – Brute Force',
    'SSH-Patator', 'DoS slowloris'
]  # You can change this list as needed

# 1. Load and Preprocess Your Dataset

# Load your dataset
data = pd.read_csv('../../../CICIDS2017_preprocessed.csv')  # Replace with your actual filename

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

# 3. Train the Classifier Model

num_epochs = 10
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
        t_normalized = t.float() / self.noise_steps  # Normalize and convert to float
        x_input = torch.cat([x, t_normalized.unsqueeze(1)], dim=1)  # Shape: (batch_size, input_dim + 1)
        x = self.fc1(x_input)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def generate_adversarial(self, x, t):
        self.eval()
        with torch.no_grad():
            batch_size = x.shape[0]
            t_tensor = torch.tensor([t]*batch_size, device=device).long()
            x_noisy, noise = self.noise_data(x.to(device), t_tensor)
            predicted_noise = self(x_noisy, t_tensor)
            alpha_t = self.alpha[t_tensor].view(-1, 1).to(device)
            alpha_hat_t = self.alpha_hat[t_tensor].view(-1, 1).to(device)
            x_adv = (1 / torch.sqrt(alpha_t)) * (
                x_noisy - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise
            )
        self.train()
        return x_adv

# 6. Define Metrics and Plotting Functions

# Define metrics to track at each time step
metrics = {
    "original": {"f1": [], "precision": [], "recall": []},
    "adversarial": {"f1": [], "precision": [], "recall": []},
    "ttpa": {"f1": [], "precision": [], "recall": []}
}

# Define incremental steps (time steps)
incremental_steps = [0, 50, 100, 200, 400, 600, 800, 999]

def evaluate_metrics(y_true, y_pred, step, data_type):
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    if data_type not in metrics:
        metrics[data_type] = {"f1": [], "precision": [], "recall": []}

    metrics[data_type]["f1"].append(f1)
    metrics[data_type]["precision"].append(precision)
    metrics[data_type]["recall"].append(recall)

    print(f"Step {step}: {data_type} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Updated plotting function for metrics
def plot_metric_comparisons(metrics, steps):
    """
    Plots F1 score comparisons between:
    - Clean vs. Adversarial vs. TTOPA
    """
    # Extract metrics
    clean_f1 = [metrics["original"]["f1"][0]] * len(steps)
    adv_f1 = metrics["adversarial"]["f1"]
    ttpa_f1 = metrics["ttpa"]["f1"]

    # Plot F1 Score Comparison (All in One)
    plt.figure(figsize=(12, 8))
    plt.plot(steps, clean_f1, label="F1 (Clean)", marker='o', color='blue')
    plt.plot(steps, adv_f1, label="F1 (Adversarial)", marker='o', color='red')
    plt.plot(steps, ttpa_f1, label="F1 (TTOPA)", marker='o', color='green')
    plt.xlabel("Time Steps")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Comparison: Clean vs. Adversarial vs. TTOPA")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/f1_comparison_all.png", dpi=300)
    plt.close()

    print("Combined F1 score comparison plot saved in the 'results' directory.")

# Updated plotting function for confusion matrices
def plot_confusion_matrices(y_true, y_pred_orig, y_pred_adv, y_pred_ttap, classes, step, filename):
    """
    Plots confusion matrices for original, adversarial, and TTOPA data side by side.
    """
    # Compute confusion matrices
    cm_orig = confusion_matrix(y_true, y_pred_orig)
    cm_adv = confusion_matrix(y_true, y_pred_adv)
    cm_ttap = confusion_matrix(y_true, y_pred_ttap)

    # Plot side by side
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=axes[0], annot_kws={"size": 8}, cbar=False)
    axes[0].set_title(f'Original Data - Step {step}')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=45)

    sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Oranges',
                xticklabels=classes, yticklabels=classes, ax=axes[1], annot_kws={"size": 8}, cbar=False)
    axes[1].set_title(f'Adversarial Data - Step {step}')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=45)

    sns.heatmap(cm_ttap, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes, ax=axes[2], annot_kws={"size": 8}, cbar=False)
    axes[2].set_title(f'TTOPA Data - Step {step}')
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_ylabel('True Label')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].tick_params(axis='y', rotation=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Confusion matrices for step {step} saved as {filename}")

# Initialize lists to collect adversarial examples and labels for plotting
adversarial_examples_dict = {}
adversarial_labels_dict = {}

# Evaluate metrics for original data (only once)
evaluate_metrics(y_test, y_pred, 0, "original")

# Define the number of epochs for training the diffusion model
diffusion_epochs = 10  # Adjust this value as needed

# 7. Initialize and Train the Diffusion Model

# Initialize diffusion model
diffusion_model = DiffusionModel(input_size).to(device)

# Define optimizer for diffusion model
diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-3)

# Define loss function (Mean Squared Error)
diffusion_criterion = nn.MSELoss()

# Training the diffusion model
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

# 8. Generate Adversarial Examples, Apply TTOPA, and Evaluate

# Initialize dictionaries to collect adversarial and TTOPA examples per step
adversarial_examples_dict = {}
adversarial_labels_dict = {}
ttpa_examples_dict = {}
ttpa_labels_dict = {}

# Initialize dictionaries to store gradient norms and losses per step
step_grad_norms = {}
step_losses = {}
ttpa_num_steps = 100  # Number of adaptation steps in TTOPA

def ttpa(model, x_adv, y_true, num_steps=30, learning_rate=0.005):
    """
    Test Time Open Packet Adaptation (TTOPA) using an optimizer.
    """
    x_adapted = x_adv.clone().detach().requires_grad_(True).to(device)
    y_true = y_true.long().to(device)
    
    # Save the model's current mode and set it to training mode
    model_mode = model.training
    model.train()
    
    # Use an optimizer for x_adapted
    optimizer_ttap = optim.Adam([x_adapted], lr=learning_rate)
    
    # Lists to store gradient norms and losses
    grad_norms = []
    losses = []
    
    for step_num in range(num_steps):
        optimizer_ttap.zero_grad()
        outputs = model(x_adapted)
        loss = criterion(outputs, y_true)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([x_adapted], max_norm=1.0)
        
        grad_norm = x_adapted.grad.norm().item()
        grad_norms.append(grad_norm)
        losses.append(loss.item())
        
        optimizer_ttap.step()
        
        # Ensure x_adapted stays within valid range
        with torch.no_grad():
            x_adapted.clamp_(min=0, max=1)
        
    # Restore the model's original mode
    if not model_mode:
        model.eval()
    
    return x_adapted.detach(), grad_norms, losses

model.eval()  # Set model to evaluation mode

for step in incremental_steps:
    y_adv_pred = []
    y_ttap_pred = []
    adversarial_examples = []
    adversarial_labels = []
    ttpa_examples = []
    ttpa_labels = []
    
    # Initialize lists to collect gradient norms and losses per TTOPA step
    all_grad_norms = [[] for _ in range(ttpa_num_steps)]
    all_losses = [[] for _ in range(ttpa_num_steps)]
    
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)

        # Generate adversarial examples at current step
        with torch.no_grad():
            x_adv = diffusion_model.generate_adversarial(data, step)

        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs_adv = model(x_adv)
            _, preds_adv = torch.max(outputs_adv.data, 1)
            y_adv_pred.extend(preds_adv.cpu().numpy())

        # Apply TTOPA to adversarial examples and collect gradient norms and losses
        x_ttap, grad_norms, losses = ttpa(
            model, x_adv, labels, num_steps=ttpa_num_steps, learning_rate=0.005
        )

        # Aggregate gradient norms and losses
        for i in range(len(grad_norms)):
            all_grad_norms[i].append(grad_norms[i])
            all_losses[i].append(losses[i])

        # Evaluate on adapted adversarial examples
        with torch.no_grad():
            outputs_ttap = model(x_ttap)
            _, preds_ttap = torch.max(outputs_ttap.data, 1)
            y_ttap_pred.extend(preds_ttap.cpu().numpy())

        # Collect adversarial examples and their labels for plotting
        adversarial_examples.append(x_adv.cpu().numpy())
        adversarial_labels.extend(labels.cpu().numpy())

        # Collect TTOPA examples and their labels for plotting
        ttpa_examples.append(x_ttap.cpu().numpy())
        ttpa_labels.extend(labels.cpu().numpy())

    # Evaluate metrics for adversarial examples
    evaluate_metrics(y_test, y_adv_pred, step, "adversarial")

    # Evaluate metrics for TTOPA adapted examples
    evaluate_metrics(y_test, y_ttap_pred, step, "ttpa")

    # After processing all batches for the current step
    avg_grad_norms = [np.mean(grad_norms) for grad_norms in all_grad_norms if grad_norms]
    avg_losses = [np.mean(losses) for losses in all_losses if losses]

    # Store the average gradient norms and losses for plotting
    step_grad_norms[step] = avg_grad_norms
    step_losses[step] = avg_losses

    # Save adversarial and TTOPA examples for the current step
    X_adv_test = np.concatenate(adversarial_examples, axis=0)
    y_adv_test = np.array(adversarial_labels)
    adversarial_examples_dict[step] = X_adv_test
    adversarial_labels_dict[step] = y_adv_test

    X_ttap_test = np.concatenate(ttpa_examples, axis=0)
    y_ttap_test = np.array(ttpa_labels)
    ttpa_examples_dict[step] = X_ttap_test
    ttpa_labels_dict[step] = y_ttap_test

    # Plot Confusion Matrices for Original, Adversarial, and TTOPA Data
    plot_confusion_matrices(
        y_test, y_pred, y_adv_pred, y_ttap_pred, label_encoder.classes_,
        step, f'results/confusion_matrices_all_step_{step}.png'
    )

# 9. Plot Metric Comparisons Over Time

# After all steps have been processed and metrics collected
plot_metric_comparisons(metrics, incremental_steps)

# 10. Plot Alpha and Beta Values over Noise Steps

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
    plt.savefig('results/alpha_beta_plot.png', dpi=300)
    plt.close()
    print("Alpha and Beta plot saved as results/alpha_beta_plot.png")

plot_alpha_beta(diffusion_model)

# 11. Plot Amount of Noise Added per Step

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
    plt.savefig('results/noise_amount_plot.png', dpi=300)
    plt.close()
    print("Noise amount plot saved as results/noise_amount_plot.png")

plot_noise_amount(diffusion_model)

# 12. Plot TTOPA Gradient Norms and Losses

def plot_ttap_gradients_losses(step_grad_norms, step_losses, ttpa_num_steps):
    """
    Plots the average gradient norms and losses per TTOPA step.

    Parameters:
        step_grad_norms (dict): Dictionary mapping step to list of average gradient norms per TTOPA step.
        step_losses (dict): Dictionary mapping step to list of average losses per TTOPA step.
        ttpa_num_steps (int): Number of TTOPA adaptation steps.
    """
    for step in incremental_steps:
        grad_norms = step_grad_norms[step]
        losses = step_losses[step]
        ttpa_steps = np.arange(1, len(grad_norms) + 1)
        
        # Plot gradient norms
        plt.figure(figsize=(12, 6))
        plt.plot(ttpa_steps, grad_norms, marker='o')
        plt.xlabel('TTOPA Step')
        plt.ylabel('Average Gradient Norm')
        plt.title(f'Average Gradient Norm per TTOPA Step (Diffusion Step {step})')
        plt.grid(True)
        plt.savefig(f'results/grad_norms_step_{step}.png', dpi=300)
        plt.close()
        
        # Plot losses
        plt.figure(figsize=(12, 6))
        plt.plot(ttpa_steps, losses, marker='o', color='red')
        plt.xlabel('TTOPA Step')
        plt.ylabel('Average Loss')
        plt.title(f'Average Loss per TTOPA Step (Diffusion Step {step})')
        plt.grid(True)
        plt.savefig(f'results/losses_step_{step}.png', dpi=300)
        plt.close()
        
        print(f"Gradient norms and losses plots for step {step} saved.")

# After all steps have been processed
plot_ttap_gradients_losses(step_grad_norms, step_losses, ttpa_num_steps)
