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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

# Initialize lists to collect adversarial examples and labels for plotting
adversarial_examples = []
adversarial_labels = []

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

# 7.5. (Optional) Save the Diffusion Model
# torch.save(diffusion_model.state_dict(), 'results/diffusion_model.pth')

# 8. Generate Adversarial Examples and Evaluate

for step in incremental_steps:
    y_adv_pred = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            # Generate adversarial examples at current step
            x_adv = diffusion_model.generate_adversarial(data, step)
            outputs_adv = model(x_adv.to(device))
            _, preds_adv = torch.max(outputs_adv.data, 1)
            y_adv_pred.extend(preds_adv.cpu().numpy())
            
            # Collect adversarial examples and their labels for plotting
            adversarial_examples.append(x_adv.cpu().numpy())
            adversarial_labels.extend(preds_adv.cpu().numpy())
    
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
                    xticklabels=classes, yticklabels=classes, ax=axes[0], annot_kws={"size": 8}, cbar=False)
        axes[0].set_title(f'Original Data - Step {step}')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=45)

        sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=axes[1], annot_kws={"size": 8}, cbar=False)
        axes[1].set_title(f'Adversarial Data - Step {step}')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=45)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Confusion matrices for step {step} saved as {filename}")

    plot_confusion_matrices(
        y_test, y_pred, y_adv_pred, label_encoder.classes_,
        step, f'results/confusion_matrices_step_{step}.png'
    )

# 9. Concatenate All Adversarial Examples and Labels

if adversarial_examples:
    try:
        X_adv_test = np.concatenate(adversarial_examples, axis=0)
        y_adv_test = np.array(adversarial_labels)
        print(f"Total adversarial examples collected: {X_adv_test.shape[0]}")
    except ValueError as ve:
        print(f"Error concatenating adversarial examples: {ve}")
        X_adv_test = np.empty((0, input_size))
        y_adv_test = np.array([])
else:
    X_adv_test = np.empty((0, input_size))  # Ensure 2D array
    y_adv_test = np.array([])
    print("No adversarial examples were collected.")

# 10. Define Plotting Functions

def plot_tsne(X, y, classes, title, filename, sample_size=5000):
    """
    Plots t-SNE visualization of the data with optional sampling.

    Parameters:
        X (numpy.ndarray): Feature data.
        y (numpy.ndarray): Labels.
        classes (list): List of class names.
        title (str): Title of the plot.
        filename (str): Path to save the plot.
        sample_size (int): Number of samples to plot. Set to None to plot all data.
    """
    if sample_size and X.shape[0] > sample_size:
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=y_sample,
        palette='tab10',
        legend='full',
        alpha=0.6
    )
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"t-SNE plot saved as {filename}")

def plot_pca(X, y, classes, title, filename):
    """
    Plots PCA visualization of the data.

    Parameters:
        X (numpy.ndarray): Feature data.
        y (numpy.ndarray): Labels.
        classes (list): List of class names.
        title (str): Title of the plot.
        filename (str): Path to save the plot.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=y,
        palette='tab10',
        legend='full',
        alpha=0.6
    )
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"PCA plot saved as {filename}")

def plot_combined_distributions(X_original, y_original, X_adversarial, y_adversarial, classes, title, filename):
    """
    Plots combined t-SNE and PCA visualizations of original and adversarial data.

    Parameters:
        X_original (numpy.ndarray): Original test features.
        y_original (numpy.ndarray): Original test labels.
        X_adversarial (numpy.ndarray): Adversarial test features.
        y_adversarial (numpy.ndarray): Adversarial test labels.
        classes (list): List of class names.
        title (str): Title of the plot.
        filename (str): Path to save the plot.
    """
    if X_adversarial.size == 0:
        print("No adversarial data to plot for combined distributions.")
        return

    # Combine data
    X_combined = np.vstack((X_original, X_adversarial))
    y_combined = np.hstack((y_original, y_adversarial))
    type_combined = np.hstack((
        np.array(['Original'] * len(y_original)),
        np.array(['Adversarial'] * len(y_adversarial))
    ))

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_combined)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_combined)

    # Create DataFrame for t-SNE
    df_plot_tsne = pd.DataFrame({
        'Component 1': X_tsne[:, 0],
        'Component 2': X_tsne[:, 1],
        'Type': type_combined,
        'Class': label_encoder.inverse_transform(y_combined)
    })

    # Create DataFrame for PCA
    df_plot_pca = pd.DataFrame({
        'Component 1': X_pca[:, 0],
        'Component 2': X_pca[:, 1],
        'Type': type_combined,
        'Class': label_encoder.inverse_transform(y_combined)
    })

    # Plot t-SNE
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='Component 1',
        y='Component 2',
        hue='Class',
        style='Type',
        palette='tab10',
        data=df_plot_tsne,
        alpha=0.6,
        edgecolor=None
    )
    plt.title(f'{title} - t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{filename}_tsne.png", dpi=300)
    plt.close()
    print(f"Combined t-SNE plot saved as {filename}_tsne.png")

    # Plot PCA
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='Component 1',
        y='Component 2',
        hue='Class',
        style='Type',
        palette='tab10',
        data=df_plot_pca,
        alpha=0.6,
        edgecolor=None
    )
    plt.title(f'{title} - PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{filename}_pca.png", dpi=300)
    plt.close()
    print(f"Combined PCA plot saved as {filename}_pca.png")

def plot_feature_kde(X_original, y_original, X_adversarial, y_adversarial, features, classes, filename):
    """
    Plots KDE plots for selected features across classes before and after adversarial attacks.

    Parameters:
        X_original (numpy.ndarray): Original test features.
        y_original (numpy.ndarray): Original test labels.
        X_adversarial (numpy.ndarray): Adversarial test features.
        y_adversarial (numpy.ndarray): Adversarial test labels.
        features (list): List of feature names.
        classes (list): List of class names.
        filename (str): Path to save the plot.
    """
    selected_features = features[:3]  # Select first 3 features for demonstration
    num_features = len(selected_features)
    
    for idx, feature in enumerate(selected_features):
        plt.figure(figsize=(12, 8))
        # Original data
        for cls in classes:
            cls_index = label_encoder.transform([cls])[0]
            subset = X_original[y_original == cls_index, idx]
            if np.var(subset) == 0:
                print(f"Feature '{feature}' for class '{cls}' has zero variance. Skipping KDE plot.")
                continue
            sns.kdeplot(subset, label=f'Original {cls}', fill=True)
        
        # Adversarial data
        if X_adversarial.size > 0:
            for cls in classes:
                cls_index = label_encoder.transform([cls])[0]
                subset = X_adversarial[y_adversarial == cls_index, idx]
                if len(subset) == 0:
                    print(f"Adversarial Feature '{feature}' for class '{cls}' has no data. Skipping KDE plot.")
                    continue
                if np.var(subset) == 0:
                    print(f"Adversarial Feature '{feature}' for class '{cls}' has zero variance. Skipping KDE plot.")
                    continue
                sns.kdeplot(subset, label=f'Adversarial {cls}', fill=True, linestyle='--')
        
        plt.title(f'Feature Distribution: {feature}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{filename}_feature_{feature}.png", dpi=300)
        plt.close()
        print(f"Feature KDE plot for {feature} saved as {filename}_feature_{feature}.png")

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
    plt.title("Comparison of Classification Metrics at Different Time Steps")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("results/metrics_comparison.png", dpi=300)
    plt.close()
    print("Metrics comparison plot saved as results/metrics_comparison.png")

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

# 11. Plotting Class Distributions Before Adversarial Generation

# Extract feature names
feature_names = features.columns.tolist()

# Plot t-SNE for original data
plot_tsne(
    X_test, y_test,
    label_encoder.classes_,
    'Original Test Data Distribution',
    'results/original_data_tsne.png'
)

# Plot PCA for original data
plot_pca(
    X_test, y_test,
    label_encoder.classes_,
    'Original Test Data Distribution',
    'results/original_data_pca.png'
)

# Plot Gaussian KDEs for original data
# Passing empty numpy arrays with correct dimensions
plot_feature_kde(
    X_test, y_test,
    np.empty((0, input_size)), np.array([]),  # Empty adversarial data as 2D numpy array
    feature_names,
    label_encoder.classes_,
    'results/original_vs_adversarial'
)

# 12. Generate Adversarial Examples and Plot After Generation

if adversarial_examples:
    # Concatenate all adversarial examples and labels collected
    try:
        X_adv_test = np.concatenate(adversarial_examples, axis=0)
        y_adv_test = np.array(adversarial_labels)
        print(f"Total adversarial examples collected: {X_adv_test.shape[0]}")
    except ValueError as ve:
        print(f"Error concatenating adversarial examples: {ve}")
        X_adv_test = np.empty((0, input_size))
        y_adv_test = np.array([])

    # Plot t-SNE for adversarial data
    plot_tsne(
        X_adv_test, y_adv_test,
        label_encoder.classes_,
        'Adversarial Test Data Distribution',
        'results/adversarial_data_tsne.png'
    )

    # Plot PCA for adversarial data
    plot_pca(
        X_adv_test, y_adv_test,
        label_encoder.classes_,
        'Adversarial Test Data Distribution',
        'results/adversarial_data_pca.png'
    )

    # Plot Gaussian KDEs for adversarial data
    plot_feature_kde(
        X_test, y_test,
        X_adv_test, y_adv_test,  # Pass adversarial data as positional arguments
        feature_names,
        label_encoder.classes_,
        'results/original_vs_adversarial'
    )
    
    # Plot combined distributions
    plot_combined_distributions(
        X_test, y_test,
        X_adv_test, y_adv_test,
        label_encoder.classes_,
        'Original vs. Adversarial Data Distribution',
        'results/combined_data_distribution'
    )
else:
    print("No adversarial examples to plot after generation.")

# 13. Plot Metrics Over Time

plot_metrics(metrics, incremental_steps)

# 14. Plot Alpha and Beta Values over Noise Steps

plot_alpha_beta(diffusion_model)

# 15. Plot Amount of Noise Added per Step

plot_noise_amount(diffusion_model)
