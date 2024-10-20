# # var-diff3.py

# import os
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')  # Use 'Agg' backend for environments without a display
# import matplotlib.pyplot as plt
# import seaborn as sns

# from tqdm import tqdm

# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     accuracy_score,
#     roc_auc_score
# )

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from torch.autograd import Variable

# import shap

# # ============================
# # 1. Configuration
# # ============================

# # Define known and unknown classes
# known_classes = [
#     'BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
#     'DoS slowloris'
# ]

# unknown_classes = [
#     'Bot', 'FTP-Patator', 'Web Attack – Brute Force',
#     'SSH-Patator'
# ]

# max_samples_per_class = 10000  # Maximum number of samples per class

# # Define batch size early in the script
# batch_size = 256  # Ensure this is defined before any DataLoader is created

# # Define incremental diffusion steps
# incremental_steps = [0, 50, 100, 200, 400, 600, 800, 999]

# # ============================
# # 2. Data Loading and Preprocessing
# # ============================

# def load_and_preprocess_data():
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Ensure the 'results' directory exists
#     if not os.path.exists('results'):
#         os.makedirs('results')

#     # Load your dataset
#     data_path = '../CICIDS2017_preprocessed.csv'  # Replace with your actual filename
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"The dataset file '{data_path}' was not found.")

#     data = pd.read_csv(data_path)

#     # Drop unnecessary columns if they exist
#     columns_to_drop = [
#         'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
#         'Destination Port', 'Timestamp'
#     ]
#     existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
#     data.drop(existing_columns_to_drop, axis=1, inplace=True)

#     # Handle missing and infinite values
#     data.replace([np.inf, -np.inf], np.nan, inplace=True)
#     data.dropna(inplace=True)

#     # Limit the number of samples per class to handle imbalance
#     data = data.groupby('Label').apply(
#         lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=42)
#     ).reset_index(drop=True)

#     # Print class distribution after limiting samples
#     print("\nClass distribution after limiting samples per class:")
#     print(data['Label'].value_counts())

#     # Split data into known and unknown
#     data_known = data[data['Label'].isin(known_classes)].copy()
#     data_unknown = data[data['Label'].isin(unknown_classes)].copy()

#     # Encode labels for known classes only
#     label_encoder = LabelEncoder()
#     y_known = label_encoder.fit_transform(data_known['Label'])
#     num_classes = len(label_encoder.classes_)  # Should be 6
#     print("\nEncoded Known Classes:", label_encoder.classes_)

#     # Define a unique integer for 'Unknown' class
#     unknown_label = num_classes  # e.g., if num_classes = 6, unknown_label = 6

#     # Features for known and unknown
#     features_known = data_known.drop('Label', axis=1)
#     features_unknown = data_unknown.drop('Label', axis=1)

#     # Ensure all features are numeric
#     features_known = features_known.apply(pd.to_numeric, errors='coerce')
#     features_unknown = features_unknown.apply(pd.to_numeric, errors='coerce')

#     # Fill any remaining NaNs
#     features_known.fillna(0, inplace=True)
#     features_unknown.fillna(0, inplace=True)

#     # Define 'features' for later use
#     features = features_known  # Now 'features' is defined

#     # Standardize the features based on known data
#     scaler = StandardScaler()
#     X_known_scaled = scaler.fit_transform(features_known)
#     X_unknown_scaled = scaler.transform(features_unknown)  # Use the same scaler

#     # Define input_size based on the number of features
#     input_size = X_known_scaled.shape[1]
#     print(f"Number of input features: {input_size}")

#     # Split known data into training and test sets with stratification
#     X_train, X_test_known, y_train, y_test_known = train_test_split(
#         X_known_scaled, y_known, test_size=0.3, random_state=42, stratify=y_known
#     )

#     # Unknown data will be part of the test set
#     X_test_unknown = X_unknown_scaled
#     y_test_unknown = np.array([-1] * X_unknown_scaled.shape[0])  # Label for 'Unknown'

#     # Combine known and unknown test data
#     X_test_combined = np.vstack((X_test_known, X_test_unknown))
#     y_test_combined = np.concatenate((y_test_known, y_test_unknown))

#     # Convert known test data to PyTorch tensors
#     X_test_known_tensor = torch.tensor(X_test_known, dtype=torch.float32).to(device)
#     y_test_known_tensor = torch.tensor(y_test_known, dtype=torch.long).to(device)

#     # Create TensorDataset and DataLoader for known test data
#     test_dataset_known = TensorDataset(X_test_known_tensor, y_test_known_tensor)
#     test_loader_known = DataLoader(test_dataset_known, batch_size=batch_size)

#     # Convert combined test data to PyTorch tensors
#     X_test_combined_tensor = torch.tensor(X_test_combined, dtype=torch.float32).to(device)
#     y_test_combined_tensor = torch.tensor(y_test_combined, dtype=torch.long).to(device)

#     # Create TensorDataset and DataLoader for combined test data
#     test_dataset_combined = TensorDataset(X_test_combined_tensor, y_test_combined_tensor)
#     test_loader_combined = DataLoader(test_dataset_combined, batch_size=batch_size)

#     # Convert training data to PyTorch tensors
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

#     # Create TensorDataset and DataLoader for training data
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     return (
#         device,
#         features,
#         scaler,
#         label_encoder,
#         num_classes,
#         unknown_label,
#         train_loader,
#         test_loader_known,
#         test_loader_combined,
#         X_train,
#         X_test_combined,
#         y_test_combined,
#         input_size  # Add input_size to returned values
#     )

# # ============================
# # 3. Define DNN Classifier
# # ============================

# class Net(nn.Module):
#     def __init__(self, input_size, num_classes, hidden_sizes=[128, 64], activation='ReLU', dropout_rate=0.0):
#         super(Net, self).__init__()
#         self.hidden_sizes = hidden_sizes
#         layers = []
#         in_size = input_size
#         for hidden_size in hidden_sizes:
#             layers.append(nn.Linear(in_size, hidden_size))
#             if activation == 'ReLU':
#                 layers.append(nn.ReLU())
#             elif activation == 'LeakyReLU':
#                 layers.append(nn.LeakyReLU())
#             elif activation == 'ELU':
#                 layers.append(nn.ELU())
#             else:
#                 raise ValueError(f"Unsupported activation function: {activation}")
#             if dropout_rate > 0.0:
#                 layers.append(nn.Dropout(dropout_rate))
#             in_size = hidden_size
#         layers.append(nn.Linear(in_size, num_classes))  # Output layer
#         self.network = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.network(x)

# # ============================
# # 4. Training the Classifier
# # ============================

# def train_classifier(model, train_loader, criterion, optimizer, num_epochs=10, validation_loader=None, device='cpu'):
#     model.train()
#     train_losses = []
#     val_f1_scores = []
#     val_accuracy_scores = []
#     val_roc_auc_scores = []
    
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for inputs, labels_batch in tqdm(train_loader, desc=f"Training Classifier Epoch {epoch+1}/{num_epochs}"):
#             inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels_batch)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
        
#         avg_loss = running_loss / len(train_loader)
#         train_losses.append(avg_loss)
        
#         # Validation Metrics if validation_loader is provided
#         if validation_loader is not None:
#             model.eval()
#             y_true_val = []
#             y_pred_val = []
#             y_scores_val = []
#             with torch.no_grad():
#                 for inputs_val, labels_val in validation_loader:
#                     inputs_val = inputs_val.to(device)
#                     labels_val = labels_val.to(device)
#                     outputs_val = model(inputs_val)
#                     probabilities = nn.functional.softmax(outputs_val, dim=1).cpu().numpy()
#                     _, predicted = torch.max(outputs_val.data, 1)
#                     y_true_val.extend(labels_val.cpu().numpy())
#                     y_pred_val.extend(predicted.cpu().numpy())
#                     y_scores_val.extend(probabilities)
#             if len(y_true_val) == 0:
#                 val_f1 = np.nan
#                 val_accuracy = np.nan
#                 val_roc_auc = np.nan
#             else:
#                 val_f1 = f1_score(y_true_val, y_pred_val, average="weighted", zero_division=0)
#                 val_accuracy = accuracy_score(y_true_val, y_pred_val)
#                 try:
#                     val_roc_auc = roc_auc_score(
#                         pd.get_dummies(y_true_val), y_scores_val, average="weighted", multi_class="ovr"
#                     )
#                 except ValueError:
#                     val_roc_auc = np.nan  # Handle cases where ROC AUC cannot be computed
#             val_f1_scores.append(val_f1)
#             val_accuracy_scores.append(val_accuracy)
#             val_roc_auc_scores.append(val_roc_auc)
            
#             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation F1 Score: {val_f1:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation ROC AUC: {val_roc_auc:.4f}")
            
#             model.train()
#         else:
#             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
#     return train_losses, val_f1_scores, val_accuracy_scores, val_roc_auc_scores

# # ============================
# # 5. Evaluating Classifier Performance
# # ============================

# def evaluate_classifier(model, data_loader, label_encoder, description='', features=None):
#     model.eval()
#     y_pred = []
#     y_true = []
#     y_scores = []
#     with torch.no_grad():
#         for inputs_test, labels_test in data_loader:
#             inputs_test = inputs_test.to(device)
#             labels_test = labels_test.to(device)
#             outputs_test = model(inputs_test)
#             probabilities = nn.functional.softmax(outputs_test, dim=1).cpu().numpy()
#             _, predicted = torch.max(outputs_test.data, 1)
#             y_pred.extend(predicted.cpu().numpy())
#             y_true.extend(labels_test.cpu().numpy())
#             y_scores.extend(probabilities)
    
#     y_pred = np.array(y_pred)
#     y_true = np.array(y_true)
#     y_scores = np.array(y_scores)
    
#     print(f'\nClassification Report ({description}):')
#     report = classification_report(
#         y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
#     )
#     print(report)
    
#     # Compute ROC AUC
#     try:
#         roc_auc = roc_auc_score(
#             pd.get_dummies(y_true), y_scores, average="weighted", multi_class="ovr"
#         )
#     except ValueError:
#         roc_auc = np.nan  # Handle cases where ROC AUC cannot be computed
    
#     print(f"ROC AUC Score ({description}): {roc_auc:.4f}\n")
    
#     # Save confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=label_encoder.classes_,
#                 yticklabels=label_encoder.classes_)
#     plt.title(f'Confusion Matrix ({description})', fontsize=16)
#     plt.xlabel('Predicted Label', fontsize=14)
#     plt.ylabel('True Label', fontsize=14)
#     plt.xticks(rotation=45, ha='right', fontsize=10)
#     plt.yticks(rotation=45, va='center', fontsize=10)
#     plt.tight_layout()
#     plt.savefig(f'results/confusion_matrix_{description}.png', dpi=300)
#     plt.close()
#     print(f"Confusion matrix saved as 'results/confusion_matrix_{description}.png'\n")

# # ============================
# # 6. Gradient-Based Sensitivity Analysis
# # ============================

# def sensitivity_analysis_classifier(model, data_loader, label_encoder, num_classes, *, description='', features=None, input_size=77):
#     """
#     Computes and plots the sensitivity of the classifier to input features based on gradients.
    
#     Parameters:
#         model (nn.Module): Trained classification model.
#         data_loader (DataLoader): DataLoader for the test dataset.
#         label_encoder (LabelEncoder): Encoder for label classes.
#         num_classes (int): Number of known classes.
#         description (str, optional): Description of the current model configuration.
#         features (pd.DataFrame, optional): DataFrame containing feature names.
#         input_size (int, optional): Number of input features.
    
#     Returns:
#         numpy.ndarray: Array of average absolute gradients per feature.
#     """
#     model.eval()
#     sensitivity_per_feature = np.zeros(input_size)
#     sample_count = 0
#     skipped_samples = 0  # To track skipped samples due to 'Unknown' labels
    
#     for inputs_test, labels_test in data_loader:
#         # Create mask for labels within [0, num_classes - 1]
#         mask = labels_test < num_classes
#         valid_count = mask.sum().item()
#         skipped_count = (~mask).sum().item()
        
#         if valid_count == 0:
#             skipped_samples += len(labels_test)
#             continue  # Skip if no valid labels in batch
        
#         inputs_valid = inputs_test[mask]
#         labels_valid = labels_test[mask]
        
#         if inputs_valid.size(0) == 0:
#             skipped_samples += len(labels_test)
#             continue  # Skip if no valid samples
        
#         inputs_valid = Variable(inputs_valid, requires_grad=True).to(device)
#         labels_valid = labels_valid.to(device)
        
#         # Optional: Add assertion to verify label ranges
#         assert labels_valid.min() >= 0 and labels_valid.max() < num_classes, "Invalid label detected."
        
#         outputs = model(inputs_valid)
#         loss = nn.CrossEntropyLoss()(outputs, labels_valid)
#         model.zero_grad()
#         loss.backward()
#         gradients = inputs_valid.grad.data.cpu().numpy()
#         sensitivity_per_feature += np.mean(np.abs(gradients), axis=0)  # Average over batch
#         sample_count += 1
#         skipped_samples += skipped_count
    
#     if sample_count == 0:
#         print("No valid samples found in the test loader for sensitivity analysis.")
#         return sensitivity_per_feature
    
#     sensitivity_per_feature /= sample_count
    
#     # Plot sensitivity per feature with adjusted font sizes
#     plt.figure(figsize=(20, 10))
#     sns.barplot(x=list(range(len(sensitivity_per_feature))), y=sensitivity_per_feature, color='blue')
#     plt.title(f"Sensitivity of Classifier to Input Features ({description})", fontsize=20)
#     plt.xlabel("Feature Index", fontsize=14)
#     plt.ylabel("Average Absolute Gradient", fontsize=14)
#     plt.xticks(ticks=list(range(len(sensitivity_per_feature))), labels=features.columns, rotation=90, fontsize=8)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f'results/classifier_feature_sensitivity_{description}.png', dpi=300)
#     plt.close()
#     print(f"Classifier feature sensitivity plot saved as 'results/classifier_feature_sensitivity_{description}.png'")
    
#     # Log skipped samples
#     print(f"Skipped {skipped_samples} samples due to 'Unknown' labels during sensitivity analysis.")
    
#     return sensitivity_per_feature

# # ============================
# # 7. SHAP Analysis for Classifier
# # ============================

# def shap_analysis_classifier(model, data_loader, label_encoder, description='', features=None, X_train=None):
#     """
#     Performs SHAP analysis to identify feature importance for the classifier.
    
#     Parameters:
#         model (nn.Module): Trained classification model.
#         data_loader (DataLoader): DataLoader for the test dataset.
#         label_encoder (LabelEncoder): Encoder for label classes.
#         description (str): Description of the current model configuration.
#         features (pd.DataFrame): DataFrame containing feature names.
#         X_train (np.ndarray): Training data for background in SHAP.
    
#     Returns:
#         shap.Explainer: SHAP explainer object.
#         shap_values: Computed SHAP values.
#     """
#     model.eval()
    
#     # Prepare background data for SHAP (a subset of training data)
#     background = X_train[:1000]
    
#     # Define a function for SHAP to predict probabilities
#     def model_predict(x):
#         with torch.no_grad():
#             inputs = torch.tensor(x, dtype=torch.float32).to(device)
#             outputs = model(inputs)
#             probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()
#         return probabilities
    
#     explainer = shap.Explainer(model_predict, background)
    
#     # Select a subset of test data for explanation (only known classes)
#     X_explain = []
#     for inputs_test, labels_test in data_loader:
#         X_explain.extend(inputs_test.cpu().numpy())
#         if len(X_explain) >= 100:
#             break
#     X_explain = np.array(X_explain[:100])
    
#     # Compute SHAP values
#     shap_values = explainer(X_explain)
    
#     # ============================
#     # Debugging: Inspect shap_values.values
#     # ============================
#     print(f"Type of shap_values.values: {type(shap_values.values)}")
    
#     if isinstance(shap_values.values, list):
#         # Multi-class classification: list of arrays, one per class
#         print(f"Number of classes in SHAP values: {len(shap_values.values)}")
#         print(f"Shape of first class SHAP values: {shap_values.values[0].shape}")
#     elif isinstance(shap_values.values, np.ndarray):
#         print(f"Shape of shap_values.values: {shap_values.values.shape}")
#     else:
#         raise TypeError(f"Unexpected type for shap_values.values: {type(shap_values.values)}")
    
#     # ============================
#     # Plot summary (beeswarm) with limited features to avoid text overflow
#     # ============================
#     plt.figure(figsize=(20, 10))  # Adjusted figure size
#     shap.summary_plot(
#         shap_values,
#         features=X_explain,
#         feature_names=features.columns,
#         show=False,
#         plot_type='dot',
#         max_display=10  # Limit to top 10 features
#     )
#     plt.title(f'SHAP Summary Plot ({description})', fontsize=20, pad=20)
#     plt.tight_layout()
#     plt.savefig(f'results/shap_summary_classifier_{description}.png', dpi=300)
#     plt.close()
#     print(f"SHAP summary plot saved as 'results/shap_summary_classifier_{description}.png'")
    
#     # Plot feature importance (bar) with limited features and adjusted font sizes
#     plt.figure(figsize=(10, 8))  # Reduced figure size for clarity
#     if isinstance(shap_values.values, list):
#         shap_feature_importance = pd.DataFrame({
#             'Feature': features.columns,
#             'Mean_Abs_SHAP': np.mean([np.mean(np.abs(class_shap), axis=0) for class_shap in shap_values.values], axis=0)
#         }).sort_values(by='Mean_Abs_SHAP', ascending=False).head(10)
#     elif isinstance(shap_values.values, np.ndarray):
#         if shap_values.values.ndim == 3:
#             # (samples, features, classes)
#             shap_feature_importance = pd.DataFrame({
#                 'Feature': features.columns,
#                 'Mean_Abs_SHAP': np.mean(np.abs(shap_values.values), axis=(0,2))
#             }).sort_values(by='Mean_Abs_SHAP', ascending=False).head(10)
#         elif shap_values.values.ndim == 2:
#             # (samples, features)
#             shap_feature_importance = pd.DataFrame({
#                 'Feature': features.columns,
#                 'Mean_Abs_SHAP': np.mean(np.abs(shap_values.values), axis=0)
#             }).sort_values(by='Mean_Abs_SHAP', ascending=False).head(10)
#         else:
#             raise ValueError(f"Unexpected number of dimensions in shap_values.values: {shap_values.values.ndim}")
#     else:
#         raise TypeError(f"Unexpected type for shap_values.values: {type(shap_values.values)}")
    
#     sns.barplot(
#         x='Mean_Abs_SHAP',
#         y='Feature',
#         data=shap_feature_importance,
#         color='blue'  # Changed from palette to color to fix FutureWarning
#     )
#     plt.title(f'SHAP Feature Importance Bar Plot ({description})', fontsize=16)
#     plt.xlabel('Mean Absolute SHAP Value', fontsize=14)
#     plt.ylabel('Feature', fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f'results/shap_feature_importance_classifier_{description}.png', dpi=300)
#     plt.close()
#     print(f"SHAP feature importance plot saved as 'results/shap_feature_importance_classifier_{description}.png'")
    
#     # ============================
#     # Add Absolute Mean SHAP Plot
#     # ============================
    
#     # Handle both binary and multi-class classification
#     if isinstance(shap_values.values, list):
#         # Multi-class classification: shap_values.values is a list of arrays, one per class
#         # Each array has shape (samples, features)
#         # Compute the mean absolute SHAP value per feature across all classes and samples
#         mean_abs_shap_per_class = []
#         for idx, class_shap in enumerate(shap_values.values):
#             mean_abs = np.abs(class_shap).mean(axis=0)  # Mean over samples
#             mean_abs_shap_per_class.append(mean_abs)
#             print(f"Mean absolute SHAP for class {idx}: {mean_abs[:5]}...")  # Show first 5 features for brevity
        
#         # Now, average over classes to get overall mean absolute SHAP per feature
#         mean_abs_shap = np.mean(mean_abs_shap_per_class, axis=0)
#         print(f"Shape of mean_abs_shap after averaging over classes: {mean_abs_shap.shape}")
#     elif isinstance(shap_values.values, np.ndarray):
#         if shap_values.values.ndim == 3:
#             # Possible shape: (samples, features, classes)
#             # Average over samples and classes
#             mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0,2))  # Shape: (features,)
#             print(f"Shape of mean_abs_shap after averaging over samples and classes: {mean_abs_shap.shape}")
#         elif shap_values.values.ndim == 2:
#             # Binary classification or regression: (samples, features)
#             # Average over samples
#             mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
#             print(f"Shape of mean_abs_shap after averaging over samples: {mean_abs_shap.shape}")
#         else:
#             raise ValueError(f"Unexpected number of dimensions in shap_values.values: {shap_values.values.ndim}")
#     else:
#         raise TypeError(f"Unexpected type for shap_values.values: {type(shap_values.values)}")
    
#     # Verify that mean_abs_shap is 1-dimensional and matches the number of features
#     if mean_abs_shap.ndim != 1:
#         raise ValueError(f"mean_abs_shap must be 1-dimensional, but got shape {mean_abs_shap.shape}")
#     if mean_abs_shap.shape[0] != len(features.columns):
#         raise ValueError(f"Number of features in mean_abs_shap ({mean_abs_shap.shape[0]}) does not match number of feature_names ({len(features.columns)})")
    
#     feature_names = features.columns
#     shap_abs_mean_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Mean_Abs_SHAP': mean_abs_shap
#     }).sort_values(by='Mean_Abs_SHAP', ascending=False).head(10)  # Top 10 features
    
#     # Plot Absolute Mean SHAP Values (Top 10) with adjusted figure size and font sizes
#     plt.figure(figsize=(12, 8))  # Adjusted figure size for clarity
#     sns.barplot(
#         x='Mean_Abs_SHAP',
#         y='Feature',
#         data=shap_abs_mean_df,
#         color='blue'  # Changed from palette to color to fix FutureWarning
#     )
#     plt.title(f'Absolute Mean SHAP Values ({description})', fontsize=16)
#     plt.xlabel('Mean Absolute SHAP Value', fontsize=14)
#     plt.ylabel('Feature', fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f'results/shap_absolute_mean_classifier_{description}.png', dpi=300)
#     plt.close()
#     print(f"SHAP absolute mean plot saved as 'results/shap_absolute_mean_classifier_{description}.png'")
    
#     # ============================
#     # Add Dependence Plot for the Top Feature
#     # ============================
    
#     # Identify the top feature based on absolute mean SHAP values
#     top_feature = shap_abs_mean_df.iloc[0]['Feature']
#     print(f"Top feature for dependence plot: {top_feature}")
    
#     # Handle multi-class by selecting SHAP values for the first class
#     plt.figure(figsize=(12, 8))  # Adjusted figure size
#     if isinstance(shap_values.values, list):
#         shap_values_class0 = shap_values.values[0]  # SHAP values for the first class
#         shap.dependence_plot(
#             top_feature,
#             shap_values_class0,
#             X_explain,
#             feature_names=features.columns,
#             show=False
#         )
#     elif isinstance(shap_values.values, np.ndarray):
#         if shap_values.values.ndim == 3:
#             # (samples, features, classes) - select first class
#             shap_values_class0 = shap_values.values[:, :, 0]
#             shap.dependence_plot(
#                 top_feature,
#                 shap_values_class0,
#                 X_explain,
#                 feature_names=features.columns,
#                 show=False
#             )
#         else:
#             shap.dependence_plot(
#                 top_feature,
#                 shap_values.values,
#                 X_explain,
#                 feature_names=features.columns,
#                 show=False
#             )
#     else:
#         raise TypeError(f"Unexpected type for shap_values.values: {type(shap_values.values)}")
    
#     plt.title(f'SHAP Dependence Plot for {top_feature} ({description})', fontsize=16, pad=20)
#     plt.xlabel(f'{top_feature}', fontsize=14)
#     plt.ylabel('SHAP Value', fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f'results/shap_dependence_{top_feature}_{description}.png', dpi=300)
#     plt.close()
#     print(f"SHAP dependence plot for {top_feature} saved as 'results/shap_dependence_{top_feature}_{description}.png'")
    
#     return explainer, shap_values

# # ============================
# # 8. Analyze and Visualize Model Parameter Characteristics
# # ============================

# def analyze_model_parameters(model, description=''):
#     """
#     Analyzes and visualizes key characteristics of model parameters.
    
#     Parameters:
#         model (nn.Module): Trained classification model.
#         description (str): Description of the current model configuration.
#     """
#     # Collect all weights and biases
#     weights = []
#     biases = []
#     layer_names = []
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             weights.append(param.data.cpu().numpy().flatten())
#             layer_names.append(name)
#         elif 'bias' in name:
#             biases.append(param.data.cpu().numpy().flatten())
#             layer_names.append(name)
    
#     # Plot histograms of weights and biases with adjusted font sizes
#     num_layers = len(weights)
#     plt.figure(figsize=(20, 15))
#     for i, (w, name) in enumerate(zip(weights, layer_names)):
#         plt.subplot(num_layers, 2, 2*i + 1)
#         sns.histplot(w, bins=50, kde=True, color='blue')
#         plt.title(f'Weight Distribution: {name}', fontsize=14)
#         plt.xlabel('Weight Value', fontsize=12)
#         plt.ylabel('Frequency', fontsize=12)
        
#         # Plot histogram of biases if available
#         if i < len(biases):
#             b = biases[i]
#             plt.subplot(num_layers, 2, 2*i + 2)
#             sns.histplot(b, bins=50, kde=True, color='orange')
#             plt.title(f'Bias Distribution: {name.replace("weight", "bias")}', fontsize=14)
#             plt.xlabel('Bias Value', fontsize=12)
#             plt.ylabel('Frequency', fontsize=12)
    
#     plt.tight_layout()
#     plt.savefig(f'results/parameter_distributions_{description}.png', dpi=300)
#     plt.close()
#     print(f"Parameter distributions plot saved as 'results/parameter_distributions_{description}.png'")
    
#     # Plot parameter norms per layer with adjusted font sizes
#     weight_norms = []
#     bias_norms = []
#     layers = []
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             weight_norm = np.linalg.norm(param.data.cpu().numpy())
#             weight_norms.append(weight_norm)
#             layers.append(name)
#         elif 'bias' in name:
#             bias_norm = np.linalg.norm(param.data.cpu().numpy())
#             bias_norms.append(bias_norm)
    
#     plt.figure(figsize=(15, 7))
#     index = np.arange(len(layers))
#     bar_width = 0.35
#     plt.bar(index, weight_norms, bar_width, label='Weight Norms', color='blue')
#     plt.bar(index + bar_width, bias_norms, bar_width, label='Bias Norms', color='orange')
#     plt.xlabel('Layer', fontsize=14)
#     plt.ylabel('L2 Norm', fontsize=14)
#     plt.title(f'Parameter Norms per Layer ({description})', fontsize=16)
#     plt.xticks(index + bar_width / 2, layers, rotation=90, fontsize=10)
#     plt.yticks(fontsize=12)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f'results/parameter_norms_{description}.png', dpi=300)
#     plt.close()
#     print(f"Parameter norms plot saved as 'results/parameter_norms_{description}.png'")
    
#     # Plot distribution of weight magnitudes across layers with adjusted font sizes
#     weight_magnitudes = [np.abs(w) for w in weights]
#     plt.figure(figsize=(20, 10))
#     for i, (w, name) in enumerate(zip(weight_magnitudes, layer_names)):
#         plt.subplot(len(weight_magnitudes), 1, i + 1)
#         sns.kdeplot(w, fill=True, color='blue')
#         plt.title(f'Weight Magnitude Distribution: {name}', fontsize=14)
#         plt.xlabel('Absolute Weight Value', fontsize=12)
#         plt.ylabel('Density', fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f'results/weight_magnitude_distributions_{description}.png', dpi=300)
#     plt.close()
#     print(f"Weight magnitude distributions plot saved as 'results/weight_magnitude_distributions_{description}.png'")

# # ============================
# # 9. Implement Top Two Difference and VarMax Algorithms
# # ============================

# def classify_with_unknown(model, x, threshold=0.5, varmax_threshold=0.1):
#     """
#     Classifies input data using Top Two Difference and VarMax.
#     If the difference between top two softmax scores is less than threshold,
#     uses VarMax to decide if it's an unknown class.
    
#     Parameters:
#         model (nn.Module): Trained classification model.
#         x (torch.Tensor): Input data tensor.
#         threshold (float): Threshold for top two probability difference.
#         varmax_threshold (float): Threshold for VarMax variance.
    
#     Returns:
#         list: List of predicted class indices or 'Unknown'.
#     """
#     with torch.no_grad():
#         logits = model(x)
#         softmax = nn.functional.softmax(logits, dim=1)
#         top_probs, top_indices = torch.topk(softmax, 2, dim=1)
#         top_diff = (top_probs[:,0] - top_probs[:,1]).cpu().numpy()
#         logits_np = logits.cpu().numpy()
        
#     predictions = []
#     for i in range(x.size(0)):
#         if top_diff[i] > threshold:
#             predictions.append(top_indices[i,0].item())
#         else:
#             # Apply VarMax
#             logit = logits_np[i]
#             variance = np.var(np.abs(logit))
#             if variance < varmax_threshold:
#                 predictions.append('Unknown')  # Unknown class
#             else:
#                 predictions.append(top_indices[i,0].item())
#     return predictions

# # ============================
# # 10. Diffusion Model for Adversarial Perturbations
# # ============================

# class SimpleDiffusionModel(nn.Module):
#     def __init__(self, input_size, hidden_size=512, num_layers=4):
#         super(SimpleDiffusionModel, self).__init__()
#         layers = []
#         in_size = input_size
#         for _ in range(num_layers):
#             layers.append(nn.Linear(in_size, hidden_size))
#             layers.append(nn.ReLU())
#             in_size = hidden_size
#         layers.append(nn.Linear(hidden_size, input_size))  # Output layer
#         self.network = nn.Sequential(*layers)
    
#     def forward(self, x, t):
#         return self.network(x)

# class Diffusion:
#     def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
#         self.T = T
#         self.beta_start = beta_start
#         self.beta_end = beta_end
#         self.device = device
        
#         self.betas = torch.linspace(beta_start, beta_end, T).to(device)
#         self.alphas = 1.0 - self.betas
#         self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
#         self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_cumprod[:-1]], dim=0)
        
#         self.sqrt_alpha = torch.sqrt(self.alphas)
#         self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
#         self.beta = self.betas  # Added for sampling step
    
#     def q_sample(self, x_start, t, noise=None):
#         """
#         Diffuse the data (forward process)
#         """
#         if noise is None:
#             noise = torch.randn_like(x_start)
#         sqrt_alpha = self.sqrt_alpha[t].reshape(-1, 1)
#         sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t].reshape(-1, 1)
#         return sqrt_alpha * x_start + sqrt_one_minus_alpha_cumprod * noise
    
#     def p_losses(self, model, x_start, t, noise=None):
#         """
#         Compute the loss for training the diffusion model
#         """
#         if noise is None:
#             noise = torch.randn_like(x_start)
#         x_noisy = self.q_sample(x_start, t, noise)
#         predicted_noise = model(x_noisy, t)
#         loss = nn.MSELoss()(predicted_noise, noise)
#         return loss

# def train_diffusion_model(model, diffusion, train_loader, optimizer, num_epochs=10):
#     model.train()
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for batch in tqdm(train_loader, desc=f"Training Diffusion Model Epoch {epoch+1}/{num_epochs}"):
#             x, _ = batch  # Ignore labels
#             x = x.to(device)
#             t = torch.randint(0, diffusion.T, (x.size(0),), device=device).long()
#             loss = diffusion.p_losses(model, x, t)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         avg_loss = epoch_loss / len(train_loader)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
#     print("Diffusion Model Training Completed.\n")

# @torch.no_grad()
# def generate_adversarial_data_at_step(diffusion_model, diffusion, X_original, t, batch_size=256):
#     """
#     Generates adversarial perturbations at a specific diffusion time step.
    
#     Parameters:
#         diffusion_model (nn.Module): Trained diffusion model.
#         diffusion (Diffusion): Diffusion process parameters.
#         X_original (np.ndarray): Original test data.
#         t (int): Diffusion time step.
#         batch_size (int): Batch size for processing.
    
#     Returns:
#         np.ndarray: Perturbed test data.
#     """
#     diffusion_model.eval()
#     X_original_tensor = torch.tensor(X_original, dtype=torch.float32).to(device)
#     num_samples = X_original_tensor.size(0)
#     perturbed_data = []
    
#     for i in tqdm(range(0, num_samples, batch_size), desc=f"Generating Perturbations at t={t}"):
#         batch = X_original_tensor[i:i+batch_size]
#         batch_size_actual = batch.size(0)
#         # Create a tensor filled with the current time step
#         t_tensor = torch.full((batch_size_actual,), t, dtype=torch.long, device=device)
#         # Generate noise scaled by perturbation strength (optional)
#         perturbation_strength = t / diffusion.T  # Example scaling; adjust as needed
#         noise = torch.randn_like(batch) * perturbation_strength
#         # Add noise to the original data
#         x_noisy = diffusion.q_sample(batch, t_tensor, noise)
#         # Predict the noise using the diffusion model
#         predicted_noise = diffusion_model(x_noisy, t_tensor)
#         # Generate perturbed data by removing predicted noise
#         x_perturbed = (batch - predicted_noise)  # Simple perturbation; can be adjusted
#         # Ensure the perturbed data stays within bounds
#         x_perturbed = torch.clamp(x_perturbed, -5, 5)  # Assuming standardized data
#         perturbed_data.append(x_perturbed.cpu().numpy())
    
#     perturbed_data = np.vstack(perturbed_data)[:num_samples]
#     return perturbed_data

# # ============================
# # 11. Evaluate Classifier on Perturbed Data
# # ============================

# def evaluate_on_perturbed_data_at_steps(model, diffusion_model, diffusion, X_test_combined, y_test_combined, label_encoder, incremental_steps, unknown_label, batch_size=256, model_description=''):
#     """
#     Evaluates the classifier on perturbed data at specified diffusion steps.
    
#     Parameters:
#         model (nn.Module): Trained classification model.
#         diffusion_model (nn.Module): Trained diffusion model.
#         diffusion (Diffusion): Diffusion process parameters.
#         X_test_combined (np.ndarray): Combined test data (known + unknown).
#         y_test_combined (np.ndarray): Combined test labels (-1 for 'Unknown').
#         label_encoder (LabelEncoder): Encoder for known classes.
#         incremental_steps (list): List of diffusion time steps to evaluate.
#         unknown_label (int): Integer label for 'Unknown' class.
#         batch_size (int): Batch size for data processing.
#         model_description (str): Description of the current model configuration.
#     """
#     metrics = {
#         'Model': [],
#         'Perturbation_Step': [],
#         'Accuracy': [],
#         'F1_Score': [],
#         'Precision': [],
#         'Recall': []
#     }
    
#     # Prepare separate DataLoaders for clean and perturbed data
#     for t in incremental_steps:
#         if t == 0:
#             # Clean Data
#             perturbed_X = X_test_combined.copy()
#             perturbed_y = y_test_combined.copy()
#             print(f"\n--- Evaluating on Clean Data (t={t}) ---")
#         else:
#             # Perturbed Data
#             perturbed_X = generate_adversarial_data_at_step(diffusion_model, diffusion, X_test_combined, t, batch_size)
#             perturbed_y = y_test_combined.copy()
#             print(f"\n--- Evaluating on Perturbed Data (t={t}) ---")
        
#         # Replace -1 with unknown_label in y_true
#         y_true_mapped = np.where(perturbed_y == -1, unknown_label, perturbed_y)
        
#         # Create a DataLoader for the current perturbation step
#         X_perturbed_tensor = torch.tensor(perturbed_X, dtype=torch.float32).to(device)
#         y_perturbed_tensor = torch.tensor(y_true_mapped, dtype=torch.long).to(device)
#         test_dataset_current = TensorDataset(X_perturbed_tensor, y_perturbed_tensor)
#         test_loader_current = DataLoader(test_dataset_current, batch_size=batch_size)
        
#         # Predict with unknown handling
#         predictions = []
#         for inputs_batch, _ in test_loader_current:
#             preds = classify_with_unknown(model, inputs_batch)
#             predictions.extend(preds)
        
#         # Map 'Unknown' to unknown_label
#         y_pred_labels = []
#         for pred in predictions:
#             if pred == 'Unknown':
#                 y_pred_labels.append(unknown_label)
#             else:
#                 y_pred_labels.append(pred)
#         y_pred_labels = np.array(y_pred_labels)
        
#         # Compute metrics
#         accuracy = accuracy_score(y_true_mapped, y_pred_labels)
#         f1 = f1_score(y_true_mapped, y_pred_labels, average='weighted', zero_division=0)
#         precision = precision_score(y_true_mapped, y_pred_labels, average='weighted', zero_division=0)
#         recall = recall_score(y_true_mapped, y_pred_labels, average='weighted', zero_division=0)
        
#         # Append metrics
#         metrics['Model'].append(model_description)
#         metrics['Perturbation_Step'].append(t)
#         metrics['Accuracy'].append(accuracy)
#         metrics['F1_Score'].append(f1)
#         metrics['Precision'].append(precision)
#         metrics['Recall'].append(recall)
        
#         # Save confusion matrix
#         extended_label_encoder = LabelEncoder()
#         extended_label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')
#         cm = confusion_matrix(y_true_mapped, y_pred_labels, labels=list(range(num_classes)) + [unknown_label])
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
#                     xticklabels=extended_label_encoder.classes_,
#                     yticklabels=extended_label_encoder.classes_)
#         plt.title(f'Confusion Matrix (t={t}) - {model_description}', fontsize=16)
#         plt.xlabel('Predicted Label', fontsize=14)
#         plt.ylabel('True Label', fontsize=14)
#         plt.xticks(rotation=45, ha='right', fontsize=10)
#         plt.yticks(rotation=45, va='center', fontsize=10)
#         plt.tight_layout()
#         plt.savefig(f'results/confusion_matrix_{model_description}_t_{t}.png', dpi=300)
#         plt.close()
#         print(f"Confusion matrix at t={t} saved as 'results/confusion_matrix_{model_description}_t_{t}.png'\n")
    
#     # ============================
#     # 5. Plotting Comparative Metrics
#     # ============================
    
#     print("\n=== Plotting Comparative Metrics ===")
    
#     # Convert metrics to DataFrame
#     metrics_df = pd.DataFrame(metrics)
    
#     # Check if metrics_df is not empty
#     if metrics_df.empty:
#         print("No metrics to plot.")
#         return
    
#     # Melt the DataFrame for easier plotting
#     metrics_melted = metrics_df.melt(id_vars=['Model', 'Perturbation_Step'], 
#                                      value_vars=['Accuracy', 'F1_Score', 'Precision', 'Recall'],
#                                      var_name='Metric', value_name='Value')
    
#     # Plot each metric for each model with adjusted font sizes
#     models = metrics_melted['Model'].unique()
#     metrics_list = metrics_melted['Metric'].unique()
    
#     for metric in metrics_list:
#         plt.figure(figsize=(12, 8))
#         for model in models:
#             subset = metrics_melted[(metrics_melted['Model'] == model) & (metrics_melted['Metric'] == metric)]
#             plt.plot(subset['Perturbation_Step'], subset['Value'], marker='o', label=model)
#         plt.title(f'{metric} vs Perturbation Step', fontsize=16)
#         plt.xlabel('Perturbation Step', fontsize=14)
#         plt.ylabel(metric, fontsize=14)
#         plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)  # Adjusted legend font size
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(f'results/{metric}_vs_Perturbation_Step.png', dpi=300, bbox_inches='tight')  # Save with tight layout
#         plt.close()
#         print(f"{metric} vs Perturbation Step plot saved as 'results/{metric}_vs_Perturbation_Step.png'")
    
#     # Additionally, plot all metrics in a single plot for each model with adjusted font sizes
#     for model in models:
#         subset = metrics_melted[metrics_melted['Model'] == model]
#         plt.figure(figsize=(12, 8))
#         sns.lineplot(data=subset, x='Perturbation_Step', y='Value', hue='Metric', marker='o')
#         plt.title(f'Performance Metrics vs Perturbation Step - {model}', fontsize=16)
#         plt.xlabel('Perturbation Step', fontsize=14)
#         plt.ylabel('Metric Value', fontsize=14)
#         plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)  # Adjusted legend font size
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(f'results/Metrics_vs_Perturbation_Step_{model}.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"All metrics vs Perturbation Step plot for {model} saved as 'results/Metrics_vs_Perturbation_Step_{model}.png'")
    
#     # Save the metrics DataFrame for further analysis if needed
#     metrics_df.to_csv('results/classifier_performance_metrics.csv', index=False)
#     print("\nClassifier performance metrics saved as 'results/classifier_performance_metrics.csv'\n")
    
#     # ============================
#     # 6. Analyze and Visualize Model Weaknesses
#     # ============================
    
#     # Placeholder for plotting weaknesses
#     # Implement as needed based on specific weaknesses identified
#     # For example, analyze changes in feature sensitivities before and after perturbations
    
#     print("\n=== Analysis and Visualization Completed ===")
#     print("All plots and results have been saved in the 'results/' directory.")

# # ============================
# # 12. Main Experiment Loop
# # ============================

# def main():
#     # Load and preprocess data
#     (
#         device_global,
#         features,
#         scaler,
#         label_encoder,
#         num_classes,
#         unknown_label,
#         train_loader,
#         test_loader_known,
#         test_loader_combined,
#         X_train,
#         X_test_combined,
#         y_test_combined,
#         input_size  # Unpack input_size here
#     ) = load_and_preprocess_data()
    
#     global device
#     device = device_global  # Set the global device variable for other functions

#     # Print input_size to confirm it's correctly unpacked
#     print(f"Input size in main: {input_size}")
    
#     # ============================
#     # 1. Train the Diffusion Model
#     # ============================
#     diffusion = Diffusion(T=1000, beta_start=1e-4, beta_end=0.02, device=device)
#     diffusion_model = SimpleDiffusionModel(input_size=input_size, hidden_size=512, num_layers=4).to(device)
#     diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-3)
    
#     print("\n=== Training the Diffusion Model for Adversarial Perturbations ===")
#     train_diffusion_model(diffusion_model, diffusion, train_loader, diffusion_optimizer, num_epochs=10)
    
#     # ============================
#     # 2. Generate Adversarial Perturbations
#     # ============================
#     print("\n=== Generating Adversarial Perturbations Using Diffusion Model ===")
#     # Initial perturbation with t=0 (clean data)
#     perturbed_X_test = generate_adversarial_data_at_step(diffusion_model, diffusion, X_test_combined, t=0, batch_size=256)
#     perturbed_y_test = y_test_combined.copy()  # -1 for 'Unknown'
    
#     # Save perturbed data as CSV
#     # Map 'Unknown' labels to their original classes for unknown samples
#     perturbed_labels = []
#     for label in perturbed_y_test:
#         if label == -1:
#             perturbed_labels.append('Unknown')
#         else:
#             perturbed_labels.append(label_encoder.inverse_transform([label])[0])
    
#     perturbed_data_df = pd.DataFrame(perturbed_X_test, columns=features.columns)
#     perturbed_data_df['Label'] = perturbed_labels
#     perturbed_data_df.to_csv('results/perturbed_test_data_t0.csv', index=False)
#     print("Perturbed test data at t=0 saved as 'results/perturbed_test_data_t0.csv'\n")
    
#     # ============================
#     # 3. Train the Classifier on Clean Data
#     # ============================
#     print("\n=== Training the Classifier on Clean Data ===")
#     model_configs = [
#         {'hidden_sizes': [128, 64], 'activation': 'ReLU', 'dropout_rate': 0.0, 'description': '2_Layers_ReLU_NoDropout'},
#         {'hidden_sizes': [256, 128, 64], 'activation': 'ReLU', 'dropout_rate': 0.0, 'description': '3_Layers_ReLU_NoDropout'},
#         {'hidden_sizes': [128, 64], 'activation': 'LeakyReLU', 'dropout_rate': 0.0, 'description': '2_Layers_LeakyReLU_NoDropout'},
#         {'hidden_sizes': [256, 128, 64], 'activation': 'LeakyReLU', 'dropout_rate': 0.0, 'description': '3_Layers_LeakyReLU_NoDropout'},
#         {'hidden_sizes': [128, 64], 'activation': 'ELU', 'dropout_rate': 0.0, 'description': '2_Layers_ELU_NoDropout'},
#         {'hidden_sizes': [256, 128, 64], 'activation': 'ELU', 'dropout_rate': 0.0, 'description': '3_Layers_ELU_NoDropout'},
#         # With Dropout
#         {'hidden_sizes': [128, 64], 'activation': 'ReLU', 'dropout_rate': 0.5, 'description': '2_Layers_ReLU_Dropout'},
#         {'hidden_sizes': [256, 128, 64], 'activation': 'ReLU', 'dropout_rate': 0.5, 'description': '3_Layers_ReLU_Dropout'},
#         {'hidden_sizes': [128, 64], 'activation': 'LeakyReLU', 'dropout_rate': 0.5, 'description': '2_Layers_LeakyReLU_Dropout'},
#         {'hidden_sizes': [256, 128, 64], 'activation': 'LeakyReLU', 'dropout_rate': 0.5, 'description': '3_Layers_LeakyReLU_Dropout'},
#         {'hidden_sizes': [128, 64], 'activation': 'ELU', 'dropout_rate': 0.5, 'description': '2_Layers_ELU_Dropout'},
#         {'hidden_sizes': [256, 128, 64], 'activation': 'ELU', 'dropout_rate': 0.5, 'description': '3_Layers_ELU_Dropout'},
#     ]

#     experiment_results = []

#     for config in model_configs:
#         print(f"\n=== Training Classifier: {config['description']} ===")
#         model = Net(
#             input_size=input_size,
#             num_classes=num_classes,  # Only known classes
#             hidden_sizes=config['hidden_sizes'],
#             activation=config['activation'],
#             dropout_rate=config['dropout_rate']
#         ).to(device)
        
#         # Define loss function and optimizer
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
#         # Train the classifier with test_loader_known as validation_loader
#         train_losses, val_f1_scores, val_accuracy_scores, val_roc_auc_scores = train_classifier(
#             model, train_loader, criterion, optimizer, num_epochs=10, validation_loader=test_loader_known, device=device
#         )
        
#         # Evaluate on clean test data (known classes only)
#         evaluate_classifier(model, test_loader_known, label_encoder, description=config['description'] + '_clean', features=features)
        
#         # Sensitivity Analysis (Gradients) on Clean Data
#         sensitivity_clean = sensitivity_analysis_classifier(
#             model,
#             test_loader_known,
#             label_encoder,
#             num_classes,  # Passed as argument
#             description=f"{config['description']}_clean",
#             features=features,
#             input_size=input_size
#         )
        
#         # SHAP Analysis on Clean Data
#         explainer_clean, shap_values_clean = shap_analysis_classifier(
#             model,
#             test_loader_known,
#             label_encoder,
#             description=f"{config['description']}_clean",
#             features=features,
#             X_train=X_train
#         )
        
#         # Analyze and visualize model parameters
#         analyze_model_parameters(model, description=config['description'])
        
#         # Evaluate on Adversarial Data
#         print(f"\n=== Performing Sensitivity and SHAP Analysis on Adversarial Data for {config['description']} ===")
        
#         # Generate adversarial data for evaluation (Example: t=200)
#         adversarial_X = generate_adversarial_data_at_step(
#             diffusion_model, diffusion, X_test_combined, t=200, batch_size=256
#         )
#         adversarial_y = y_test_combined.copy()
        
#         # Replace -1 with unknown_label in y_true
#         y_true_adversarial = np.where(adversarial_y == -1, unknown_label, adversarial_y)
        
#         # Create DataLoader for adversarial data
#         adversarial_X_tensor = torch.tensor(adversarial_X, dtype=torch.float32).to(device)
#         adversarial_y_tensor = torch.tensor(y_true_adversarial, dtype=torch.long).to(device)
#         adversarial_dataset = TensorDataset(adversarial_X_tensor, adversarial_y_tensor)
#         adversarial_loader = DataLoader(adversarial_dataset, batch_size=batch_size)
        
#         # Sensitivity Analysis (Gradients) on Adversarial Data
#         sensitivity_adversarial = sensitivity_analysis_classifier(
#             model,
#             adversarial_loader,
#             label_encoder,
#             num_classes,  # Passed as argument
#             description=f"{config['description']}_adversarial_t200",
#             features=features,
#             input_size=input_size
#         )
        
#         # SHAP Analysis on Adversarial Data
#         explainer_adversarial, shap_values_adversarial = shap_analysis_classifier(
#             model,
#             adversarial_loader,
#             label_encoder,
#             description=f"{config['description']}_adversarial_t200",
#             features=features,
#             X_train=X_train
#         )
#         # Store results
#         experiment_results.append({
#             'model_description': config['description'],
#             'model': model,  # Store model for later evaluation
#             'train_losses': train_losses,
#             'val_f1_scores': val_f1_scores,
#             'val_accuracy_scores': val_accuracy_scores,
#             'val_roc_auc_scores': val_roc_auc_scores,
#             'sensitivity_clean': sensitivity_clean,
#             'sensitivity_adversarial': sensitivity_adversarial,
#             'shap_values_clean': shap_values_clean,
#             'shap_values_adversarial': shap_values_adversarial
#         })


#     # ============================
#     # 4. Evaluate Classifier on Clean and Perturbed Data
#     # ============================

#     print("\n=== Evaluating Classifiers on Clean and Perturbed Data ===")
    
#     # Initialize a DataFrame to store metrics
#     metrics_df = pd.DataFrame(columns=['Model', 'Perturbation_Step', 'Accuracy', 'F1_Score', 'Precision', 'Recall'])
    
#     for result in experiment_results:
#         description = result['model_description']
#         model = result['model']
#         print(f"\n=== Evaluating Classifier: {description} ===")
        
#         for t in incremental_steps:
#             if t == 0:
#                 # Clean Data
#                 perturbed_X = X_test_combined.copy()
#                 perturbed_y = y_test_combined.copy()
#                 print(f"\n--- Evaluating on Clean Data (t={t}) ---")
#             else:
#                 # Perturbed Data
#                 perturbed_X = generate_adversarial_data_at_step(diffusion_model, diffusion, X_test_combined, t, batch_size=256)
#                 perturbed_y = y_test_combined.copy()
#                 print(f"\n--- Evaluating on Perturbed Data (t={t}) ---")
            
#             # Replace -1 with unknown_label in y_true
#             y_true_mapped = np.where(perturbed_y == -1, unknown_label, perturbed_y)
            
#             # Predict with unknown handling
#             X_perturbed_tensor = torch.tensor(perturbed_X, dtype=torch.float32).to(device)
#             predictions = []
#             for i in range(0, X_perturbed_tensor.size(0), batch_size):
#                 batch = X_perturbed_tensor[i:i+batch_size]
#                 preds = classify_with_unknown(model, batch)
#                 predictions.extend(preds)
            
#             # Map 'Unknown' to unknown_label
#             y_pred_labels = []
#             for pred in predictions:
#                 if pred == 'Unknown':
#                     y_pred_labels.append(unknown_label)
#                 else:
#                     y_pred_labels.append(pred)
#             y_pred_labels = np.array(y_pred_labels)
            
#             # Compute metrics
#             accuracy = accuracy_score(y_true_mapped, y_pred_labels)
#             f1 = f1_score(y_true_mapped, y_pred_labels, average='weighted', zero_division=0)
#             precision = precision_score(y_true_mapped, y_pred_labels, average='weighted', zero_division=0)
#             recall = recall_score(y_true_mapped, y_pred_labels, average='weighted', zero_division=0)
            
#             # Append metrics
#             metrics_df = metrics_df.append({
#                 'Model': description,
#                 'Perturbation_Step': t,
#                 'Accuracy': accuracy,
#                 'F1_Score': f1,
#                 'Precision': precision,
#                 'Recall': recall
#             }, ignore_index=True)
            
#             # Save confusion matrix
#             extended_label_encoder = LabelEncoder()
#             extended_label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')
#             cm = confusion_matrix(y_true_mapped, y_pred_labels, labels=list(range(num_classes)) + [unknown_label])
#             plt.figure(figsize=(12, 10))
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
#                         xticklabels=extended_label_encoder.classes_,
#                         yticklabels=extended_label_encoder.classes_)
#             plt.title(f'Confusion Matrix (t={t}) - {description}', fontsize=16)
#             plt.xlabel('Predicted Label', fontsize=14)
#             plt.ylabel('True Label', fontsize=14)
#             plt.xticks(rotation=45, ha='right', fontsize=10)
#             plt.yticks(rotation=45, va='center', fontsize=10)
#             plt.tight_layout()
#             plt.savefig(f'results/confusion_matrix_{description}_t_{t}.png', dpi=300)
#             plt.close()
#             print(f"Confusion matrix at t={t} saved as 'results/confusion_matrix_{description}_t_{t}.png'\n")
    
#     # ============================
#     # 5. Plotting Comparative Metrics
#     # ============================

#     print("\n=== Plotting Comparative Metrics ===")
    
#     # Convert metrics to DataFrame
#     metrics_df = pd.DataFrame(metrics)
    
#     # Check if metrics_df is not empty
#     if metrics_df.empty:
#         print("No metrics to plot.")
#         return
    
#     # Melt the DataFrame for easier plotting
#     metrics_melted = metrics_df.melt(id_vars=['Model', 'Perturbation_Step'], 
#                                      value_vars=['Accuracy', 'F1_Score', 'Precision', 'Recall'],
#                                      var_name='Metric', value_name='Value')
    
#     # Plot each metric for each model with adjusted font sizes
#     models = metrics_melted['Model'].unique()
#     metrics_list = metrics_melted['Metric'].unique()
    
#     for metric in metrics_list:
#         plt.figure(figsize=(12, 8))
#         for model in models:
#             subset = metrics_melted[(metrics_melted['Model'] == model) & (metrics_melted['Metric'] == metric)]
#             plt.plot(subset['Perturbation_Step'], subset['Value'], marker='o', label=model)
#         plt.title(f'{metric} vs Perturbation Step', fontsize=16)
#         plt.xlabel('Perturbation Step', fontsize=14)
#         plt.ylabel(metric, fontsize=14)
#         plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)  # Adjusted legend font size
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(f'results/{metric}_vs_Perturbation_Step.png', dpi=300, bbox_inches='tight')  # Save with tight layout
#         plt.close()
#         print(f"{metric} vs Perturbation Step plot saved as 'results/{metric}_vs_Perturbation_Step.png'")
    
#     # Additionally, plot all metrics in a single plot for each model with adjusted font sizes
#     for model in models:
#         subset = metrics_melted[metrics_melted['Model'] == model]
#         plt.figure(figsize=(12, 8))
#         sns.lineplot(data=subset, x='Perturbation_Step', y='Value', hue='Metric', marker='o')
#         plt.title(f'Performance Metrics vs Perturbation Step - {model}', fontsize=16)
#         plt.xlabel('Perturbation Step', fontsize=14)
#         plt.ylabel('Metric Value', fontsize=14)
#         plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)  # Adjusted legend font size
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(f'results/Metrics_vs_Perturbation_Step_{model}.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"All metrics vs Perturbation Step plot for {model} saved as 'results/Metrics_vs_Perturbation_Step_{model}.png'")
    
#     # Save the metrics DataFrame for further analysis if needed
#     metrics_df.to_csv('results/classifier_performance_metrics.csv', index=False)
#     print("\nClassifier performance metrics saved as 'results/classifier_performance_metrics.csv'\n")
    
#     # ============================
#     # 6. Analyze and Visualize Model Weaknesses
#     # ============================
    
#     # Placeholder for plotting weaknesses
#     # Implement as needed based on specific weaknesses identified
#     # For example, analyze changes in feature sensitivities before and after perturbations
    
#     print("\n=== Analysis and Visualization Completed ===")
#     print("All plots and results have been saved in the 'results/' directory.")

# if __name__ == "__main__":
#     main()


# var-diff3.py

# var-diff3.py

# var-diff3.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for environments without a display
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

import shap

# ============================
# 1. Configuration
# ============================

# Define known and unknown classes
known_classes = [
    'BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
    'DoS slowloris'
]

unknown_classes = [
    'Bot', 'FTP-Patator', 'Web Attack – Brute Force',
    'SSH-Patator'
]

max_samples_per_class = 10000  # Maximum number of samples per class

# Define batch size early in the script
batch_size = 256  # Ensure this is defined before any DataLoader is created

# Define incremental diffusion steps
incremental_steps = [0, 50, 100, 200, 400, 600, 800, 999]

# ============================
# 2. Data Loading and Preprocessing
# ============================

def load_and_preprocess_data():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure the 'results' directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # Load your dataset
    data_path = '../CICIDS2017_preprocessed.csv'  # Replace with your actual filename
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The dataset file '{data_path}' was not found.")

    data = pd.read_csv(data_path)

    # Drop unnecessary columns if they exist
    columns_to_drop = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
        'Destination Port', 'Timestamp'
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(existing_columns_to_drop, axis=1, inplace=True)

    # Handle missing and infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Limit the number of samples per class to handle imbalance
    data = data.groupby('Label').apply(
        lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=42)
    ).reset_index(drop=True)

    # Print class distribution after limiting samples
    print("\nClass distribution after limiting samples per class:")
    print(data['Label'].value_counts())

    # Split data into known and unknown
    data_known = data[data['Label'].isin(known_classes)].copy()
    data_unknown = data[data['Label'].isin(unknown_classes)].copy()

    # Encode labels for known classes only
    label_encoder = LabelEncoder()
    y_known = label_encoder.fit_transform(data_known['Label'])
    num_classes = len(label_encoder.classes_)  # Should be 6
    print("\nEncoded Known Classes:", label_encoder.classes_)

    # Define a unique integer for 'Unknown' class
    unknown_label = num_classes  # e.g., if num_classes = 6, unknown_label = 6

    # Features for known and unknown
    features_known = data_known.drop('Label', axis=1)
    features_unknown = data_unknown.drop('Label', axis=1)

    # Ensure all features are numeric
    features_known = features_known.apply(pd.to_numeric, errors='coerce')
    features_unknown = features_unknown.apply(pd.to_numeric, errors='coerce')

    # Fill any remaining NaNs
    features_known.fillna(0, inplace=True)
    features_unknown.fillna(0, inplace=True)

    # Define 'features' for later use
    features = features_known  # Now 'features' is defined

    # Standardize the features based on known data
    scaler = StandardScaler()
    X_known_scaled = scaler.fit_transform(features_known)
    X_unknown_scaled = scaler.transform(features_unknown)  # Use the same scaler

    # Define input_size based on the number of features
    input_size = X_known_scaled.shape[1]
    print(f"Number of input features: {input_size}")

    # Split known data into training and test sets with stratification
    X_train, X_test_known, y_train, y_test_known = train_test_split(
        X_known_scaled, y_known, test_size=0.3, random_state=42, stratify=y_known
    )

    # Unknown data will be part of the test set
    X_test_unknown = X_unknown_scaled
    y_test_unknown = np.array([-1] * X_unknown_scaled.shape[0])  # Label for 'Unknown'

    # Combine known and unknown test data
    X_test_combined = np.vstack((X_test_known, X_test_unknown))
    y_test_combined = np.concatenate((y_test_known, y_test_unknown))

    # Convert known test data to PyTorch tensors
    X_test_known_tensor = torch.tensor(X_test_known, dtype=torch.float32).to(device)
    y_test_known_tensor = torch.tensor(y_test_known, dtype=torch.long).to(device)

    # Create TensorDataset and DataLoader for known test data
    test_dataset_known = TensorDataset(X_test_known_tensor, y_test_known_tensor)
    test_loader_known = DataLoader(test_dataset_known, batch_size=batch_size)

    # Convert combined test data to PyTorch tensors
    X_test_combined_tensor = torch.tensor(X_test_combined, dtype=torch.float32).to(device)
    y_test_combined_tensor = torch.tensor(y_test_combined, dtype=torch.long).to(device)

    # Create TensorDataset and DataLoader for combined test data
    test_dataset_combined = TensorDataset(X_test_combined_tensor, y_test_combined_tensor)
    test_loader_combined = DataLoader(test_dataset_combined, batch_size=batch_size)

    # Convert training data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    # Create TensorDataset and DataLoader for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return (
        device,
        features,
        scaler,
        label_encoder,
        num_classes,
        unknown_label,
        train_loader,
        test_loader_known,
        test_loader_combined,
        X_train,
        X_test_combined,
        y_test_combined,
        input_size  # Add input_size to returned values
    )

# ============================
# 3. Define DNN Classifier
# ============================

class Net(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[128, 64], activation='ReLU', dropout_rate=0.0):
        super(Net, self).__init__()
        self.hidden_sizes = hidden_sizes
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            elif activation == 'ELU':
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, num_classes))  # Output layer
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================
# 4. Training the Classifier
# ============================

def train_classifier(model, train_loader, criterion, optimizer, num_epochs=10, validation_loader=None, device='cpu', master_df=None, model_description=''):
    """
    Trains the classifier model and appends metrics to the master DataFrame.
    
    Parameters:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        num_epochs (int): Number of training epochs.
        validation_loader (DataLoader, optional): DataLoader for validation data.
        device (torch.device): Device to run training on.
        master_df (pd.DataFrame, optional): Master DataFrame to append metrics.
        model_description (str): Description of the current model configuration.
    
    Returns:
        tuple: Lists containing training losses and validation metrics, updated master_df.
    """
    model.train()
    train_losses = []
    val_f1_scores = []
    val_accuracy_scores = []
    val_roc_auc_scores = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels_batch in tqdm(train_loader, desc=f"Training Classifier Epoch {epoch+1}/{num_epochs}"):
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Append training loss to master_df
        if master_df is not None:
            master_df = master_df.append({
                'Model': model_description,
                'Perturbation_Step': None,
                'Epoch': epoch + 1,
                'Metric_Type': 'Training',
                'Metric_Name': 'Loss',
                'Metric_Value': avg_loss
            }, ignore_index=True)
        
        # Validation Metrics if validation_loader is provided
        if validation_loader is not None:
            model.eval()
            y_true_val = []
            y_pred_val = []
            y_scores_val = []
            with torch.no_grad():
                for inputs_val, labels_val in validation_loader:
                    inputs_val = inputs_val.to(device)
                    labels_val = labels_val.to(device)
                    outputs_val = model(inputs_val)
                    probabilities = nn.functional.softmax(outputs_val, dim=1).cpu().numpy()
                    _, predicted = torch.max(outputs_val.data, 1)
                    y_true_val.extend(labels_val.cpu().numpy())
                    y_pred_val.extend(predicted.cpu().numpy())
                    y_scores_val.extend(probabilities)
            if len(y_true_val) == 0:
                val_f1 = np.nan
                val_accuracy = np.nan
                val_roc_auc = np.nan
            else:
                val_f1 = f1_score(y_true_val, y_pred_val, average="weighted", zero_division=0)
                val_accuracy = accuracy_score(y_true_val, y_pred_val)
                try:
                    val_roc_auc = roc_auc_score(
                        pd.get_dummies(y_true_val), y_scores_val, average="weighted", multi_class="ovr"
                    )
                except ValueError:
                    val_roc_auc = np.nan  # Handle cases where ROC AUC cannot be computed
            val_f1_scores.append(val_f1)
            val_accuracy_scores.append(val_accuracy)
            val_roc_auc_scores.append(val_roc_auc)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation F1 Score: {val_f1:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation ROC AUC: {val_roc_auc:.4f}")
            
            # Append validation metrics to master_df
            if master_df is not None:
                master_df = master_df.append([
                    {
                        'Model': model_description,
                        'Perturbation_Step': None,
                        'Epoch': epoch + 1,
                        'Metric_Type': 'Validation',
                        'Metric_Name': 'F1_Score',
                        'Metric_Value': val_f1
                    },
                    {
                        'Model': model_description,
                        'Perturbation_Step': None,
                        'Epoch': epoch + 1,
                        'Metric_Type': 'Validation',
                        'Metric_Name': 'Accuracy',
                        'Metric_Value': val_accuracy
                    },
                    {
                        'Model': model_description,
                        'Perturbation_Step': None,
                        'Epoch': epoch + 1,
                        'Metric_Type': 'Validation',
                        'Metric_Name': 'ROC_AUC',
                        'Metric_Value': val_roc_auc
                    }
                ], ignore_index=True)
            
            model.train()
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return train_losses, val_f1_scores, val_accuracy_scores, val_roc_auc_scores, master_df

# ============================
# 5. Evaluating Classifier Performance
# ============================

def evaluate_classifier(model, data_loader, label_encoder, description='', features=None):
    model.eval()
    y_pred = []
    y_true = []
    y_scores = []
    with torch.no_grad():
        for inputs_test, labels_test in data_loader:
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = model(inputs_test)
            probabilities = nn.functional.softmax(outputs_test, dim=1).cpu().numpy()
            _, predicted = torch.max(outputs_test.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels_test.cpu().numpy())
            y_scores.extend(probabilities)
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    print(f'\nClassification Report ({description}):')
    report = classification_report(
        y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    print(report)
    
    # Compute ROC AUC
    try:
        roc_auc = roc_auc_score(
            pd.get_dummies(y_true),
            y_scores,
            average="weighted",
            multi_class="ovr"
        )
    except ValueError:
        roc_auc = np.nan  # Handle cases where ROC AUC cannot be computed
    
    print(f"ROC AUC Score ({description}): {roc_auc:.4f}\n")
    
    # Save confusion matrix
    extended_label_encoder = LabelEncoder()
    extended_label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix ({description})', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=45, va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{description}.png', dpi=300)
    plt.close()
    print(f"Confusion matrix saved as 'results/confusion_matrix_{description}.png'\n")

# ============================
# 6. Gradient-Based Sensitivity Analysis
# ============================

def sensitivity_analysis_classifier(model, data_loader, label_encoder, num_classes, *, description='', features=None, input_size=77, master_df=None):
    """
    Computes and plots the sensitivity of the classifier to input features based on gradients.
    
    Parameters:
        model (nn.Module): Trained classification model.
        data_loader (DataLoader): DataLoader for the test dataset.
        label_encoder (LabelEncoder): Encoder for label classes.
        num_classes (int): Number of known classes.
        description (str, optional): Description of the current model configuration.
        features (pd.DataFrame, optional): DataFrame containing feature names.
        input_size (int, optional): Number of input features.
        master_df (pd.DataFrame, optional): Master DataFrame to append metrics.
    
    Returns:
        numpy.ndarray: Array of average absolute gradients per feature.
    """
    model.eval()
    sensitivity_per_feature = np.zeros(input_size)
    sample_count = 0
    skipped_samples = 0  # To track skipped samples due to 'Unknown' labels
    
    for inputs_test, labels_test in data_loader:
        # Create mask for labels within [0, num_classes - 1]
        mask = labels_test < num_classes
        valid_count = mask.sum().item()
        skipped_count = (~mask).sum().item()
        
        if valid_count == 0:
            skipped_samples += len(labels_test)
            continue  # Skip if no valid labels in batch
        
        inputs_valid = inputs_test[mask]
        labels_valid = labels_test[mask]
        
        if inputs_valid.size(0) == 0:
            skipped_samples += len(labels_test)
            continue  # Skip if no valid samples
        
        inputs_valid = Variable(inputs_valid, requires_grad=True).to(device)
        labels_valid = labels_valid.to(device)
        
        # Optional: Add assertion to verify label ranges
        assert labels_valid.min() >= 0 and labels_valid.max() < num_classes, "Invalid label detected."
        
        outputs = model(inputs_valid)
        loss = nn.CrossEntropyLoss()(outputs, labels_valid)
        model.zero_grad()
        loss.backward()
        gradients = inputs_valid.grad.data.cpu().numpy()
        sensitivity_per_feature += np.mean(np.abs(gradients), axis=0)  # Average over batch
        sample_count += 1
        skipped_samples += skipped_count
    
    if sample_count == 0:
        print("No valid samples found in the test loader for sensitivity analysis.")
        return sensitivity_per_feature
    
    sensitivity_per_feature /= sample_count
    
    # Append sensitivity data to master_df
    if master_df is not None:
        for idx, value in enumerate(sensitivity_per_feature):
            master_df = master_df.append({
                'Model': description,
                'Perturbation_Step': None,
                'Epoch': None,
                'Metric_Type': 'Sensitivity',
                'Metric_Name': f'Feature_{idx}_Sensitivity',
                'Metric_Value': value
            }, ignore_index=True)
    
    # Create a DataFrame for sensitivity and save
    sensitivity_df = pd.DataFrame({
        'Model': [description] * input_size,
        'Feature': features.columns,
        'Sensitivity': sensitivity_per_feature
    })
    sensitivity_df.to_csv(f'results/sensitivity_analysis_{description}.csv', index=False)
    print(f"Sensitivity analysis data saved as 'results/sensitivity_analysis_{description}.csv'")
    
    # Plot sensitivity per feature with adjusted font sizes
    plt.figure(figsize=(20, 10))
    sns.barplot(x=list(range(len(sensitivity_per_feature))), y=sensitivity_per_feature, color='blue')
    plt.title(f"Sensitivity of Classifier to Input Features ({description})", fontsize=20)
    plt.xlabel("Feature Index", fontsize=14)
    plt.ylabel("Average Absolute Gradient", fontsize=14)
    plt.xticks(ticks=list(range(len(sensitivity_per_feature))), labels=features.columns, rotation=90, fontsize=8)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/classifier_feature_sensitivity_{description}.png', dpi=300)
    plt.close()
    print(f"Classifier feature sensitivity plot saved as 'results/classifier_feature_sensitivity_{description}.png'")
    
    # Log skipped samples
    print(f"Skipped {skipped_samples} samples due to 'Unknown' labels during sensitivity analysis.")
    
    return sensitivity_per_feature

# ============================
# 7. SHAP Analysis for Classifier
# ============================

def shap_analysis_classifier(model, data_loader, label_encoder, description='', features=None, X_train=None, master_df=None):
    """
    Performs SHAP analysis to identify feature importance for the classifier.
    
    Parameters:
        model (nn.Module): Trained classification model.
        data_loader (DataLoader): DataLoader for the test dataset.
        label_encoder (LabelEncoder): Encoder for label classes.
        description (str): Description of the current model configuration.
        features (pd.DataFrame): DataFrame containing feature names.
        X_train (np.ndarray): Training data for background in SHAP.
        master_df (pd.DataFrame, optional): Master DataFrame to append metrics.
    
    Returns:
        tuple: SHAP explainer and SHAP values.
    """
    model.eval()
    
    # Prepare background data for SHAP (a subset of training data)
    background = X_train[:1000]
    
    # Define a function for SHAP to predict probabilities
    def model_predict(x):
        with torch.no_grad():
            inputs = torch.tensor(x, dtype=torch.float32).to(device)
            outputs = model(inputs)
            probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()
        return probabilities
    
    explainer = shap.Explainer(model_predict, background)
    
    # Select a subset of test data for explanation (only known classes)
    X_explain = []
    for inputs_test, labels_test in data_loader:
        X_explain.extend(inputs_test.cpu().numpy())
        if len(X_explain) >= 100:
            break
    X_explain = np.array(X_explain[:100])
    
    # Compute SHAP values
    shap_values = explainer(X_explain)
    
    # ============================
    # Debugging: Inspect shap_values.values
    # ============================
    print(f"Type of shap_values.values: {type(shap_values.values)}")
    
    if isinstance(shap_values.values, list):
        # Multi-class classification: list of arrays, one per class
        print(f"Number of classes in SHAP values: {len(shap_values.values)}")
        print(f"Shape of first class SHAP values: {shap_values.values[0].shape}")
    elif isinstance(shap_values.values, np.ndarray):
        print(f"Shape of shap_values.values: {shap_values.values.shape}")
    else:
        raise TypeError(f"Unexpected type for shap_values.values: {type(shap_values.values)}")
    
    # ============================
    # Plot summary (beeswarm) with limited features to avoid text overflow
    # ============================
    plt.figure(figsize=(20, 10))  # Adjusted figure size
    shap.summary_plot(
        shap_values,
        features=X_explain,
        feature_names=features.columns,
        show=False,
        plot_type='dot',
        max_display=10  # Limit to top 10 features
    )
    plt.title(f'SHAP Summary Plot ({description})', fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(f'results/shap_summary_classifier_{description}.png', dpi=300)
    plt.close()
    print(f"SHAP summary plot saved as 'results/shap_summary_classifier_{description}.png'")
    
    # Plot feature importance (bar) with limited features and adjusted font sizes
    plt.figure(figsize=(10, 8))  # Reduced figure size for clarity
    if isinstance(shap_values.values, list):
        shap_feature_importance = pd.DataFrame({
            'Feature': features.columns,
            'Mean_Abs_SHAP': np.mean([np.mean(np.abs(class_shap), axis=0) for class_shap in shap_values.values], axis=0)
        }).sort_values(by='Mean_Abs_SHAP', ascending=False).head(10)
    elif isinstance(shap_values.values, np.ndarray):
        if shap_values.values.ndim == 3:
            # (samples, features, classes)
            shap_feature_importance = pd.DataFrame({
                'Feature': features.columns,
                'Mean_Abs_SHAP': np.mean(np.abs(shap_values.values), axis=(0,2))
            }).sort_values(by='Mean_Abs_SHAP', ascending=False).head(10)
        elif shap_values.values.ndim == 2:
            # (samples, features)
            shap_feature_importance = pd.DataFrame({
                'Feature': features.columns,
                'Mean_Abs_SHAP': np.mean(np.abs(shap_values.values), axis=0)
            }).sort_values(by='Mean_Abs_SHAP', ascending=False).head(10)
        else:
            raise ValueError(f"Unexpected number of dimensions in shap_values.values: {shap_values.values.ndim}")
    else:
        raise TypeError(f"Unexpected type for shap_values.values: {type(shap_values.values)}")
    
    # Save SHAP feature importance to master_df
    if master_df is not None:
        for _, row in shap_feature_importance.iterrows():
            master_df = master_df.append({
                'Model': description,
                'Perturbation_Step': None,
                'Epoch': None,
                'Metric_Type': 'SHAP',
                'Metric_Name': f"{row['Feature']}_Mean_Abs_SHAP",
                'Metric_Value': row['Mean_Abs_SHAP']
            }, ignore_index=True)
    
    sns.barplot(
        x='Mean_Abs_SHAP',
        y='Feature',
        data=shap_feature_importance,
        color='blue'  # Changed from palette to color to fix FutureWarning
    )
    plt.title(f'SHAP Feature Importance Bar Plot ({description})', fontsize=16)
    plt.xlabel('Mean Absolute SHAP Value', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/shap_feature_importance_classifier_{description}.png', dpi=300)
    plt.close()
    print(f"SHAP feature importance plot saved as 'results/shap_feature_importance_classifier_{description}.png'")
    
    # ============================
    # Add Absolute Mean SHAP Plot
    # ============================
    
    # Handle both binary and multi-class classification
    if isinstance(shap_values.values, list):
        # Multi-class classification: shap_values.values is a list of arrays, one per class
        # Each array has shape (samples, features)
        # Compute the mean absolute SHAP value per feature across all classes and samples
        mean_abs_shap_per_class = []
        for idx, class_shap in enumerate(shap_values.values):
            mean_abs = np.abs(class_shap).mean(axis=0)  # Mean over samples
            mean_abs_shap_per_class.append(mean_abs)
            print(f"Mean absolute SHAP for class {idx}: {mean_abs[:5]}...")  # Show first 5 features for brevity
        
        # Now, average over classes to get overall mean absolute SHAP per feature
        mean_abs_shap = np.mean(mean_abs_shap_per_class, axis=0)
        print(f"Shape of mean_abs_shap after averaging over classes: {mean_abs_shap.shape}")
    elif isinstance(shap_values.values, np.ndarray):
        if shap_values.values.ndim == 3:
            # Possible shape: (samples, features, classes)
            # Average over samples and classes
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0,2))  # Shape: (features,)
            print(f"Shape of mean_abs_shap after averaging over samples and classes: {mean_abs_shap.shape}")
        elif shap_values.values.ndim == 2:
            # Binary classification or regression: (samples, features)
            # Average over samples
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            print(f"Shape of mean_abs_shap after averaging over samples: {mean_abs_shap.shape}")
        else:
            raise ValueError(f"Unexpected number of dimensions in shap_values.values: {shap_values.values.ndim}")
    else:
        raise TypeError(f"Unexpected type for shap_values.values: {type(shap_values.values)}")
    
    # Verify that mean_abs_shap is 1-dimensional and matches the number of features
    if mean_abs_shap.ndim != 1:
        raise ValueError(f"mean_abs_shap must be 1-dimensional, but got shape {mean_abs_shap.shape}")
    if mean_abs_shap.shape[0] != len(features.columns):
        raise ValueError(f"Number of features in mean_abs_shap ({mean_abs_shap.shape[0]}) does not match number of feature_names ({len(features.columns)})")
    
    feature_names = features.columns
    shap_abs_mean_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False).head(10)  # Top 10 features
    
    # Plot Absolute Mean SHAP Values (Top 10) with adjusted figure size and font sizes
    plt.figure(figsize=(12, 8))  # Adjusted figure size for clarity
    sns.barplot(
        x='Mean_Abs_SHAP',
        y='Feature',
        data=shap_abs_mean_df,
        color='blue'  # Changed from palette to color to fix FutureWarning
    )
    plt.title(f'Absolute Mean SHAP Values ({description})', fontsize=16)
    plt.xlabel('Mean Absolute SHAP Value', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/shap_absolute_mean_classifier_{description}.png', dpi=300)
    plt.close()
    print(f"SHAP absolute mean plot saved as 'results/shap_absolute_mean_classifier_{description}.png'")
    
    # ============================
    # Add Dependence Plot for the Top Feature
    # ============================
    
    # Identify the top feature based on absolute mean SHAP values
    top_feature = shap_abs_mean_df.iloc[0]['Feature']
    print(f"Top feature for dependence plot: {top_feature}")
    
    # Handle multi-class by selecting SHAP values for the first class
    plt.figure(figsize=(12, 8))  # Adjusted figure size
    if isinstance(shap_values.values, list):
        shap_values_class0 = shap_values.values[0]  # SHAP values for the first class
        shap.dependence_plot(
            top_feature,
            shap_values_class0,
            X_explain,
            feature_names=features.columns,
            show=False
        )
    elif isinstance(shap_values.values, np.ndarray):
        if shap_values.values.ndim == 3:
            # (samples, features, classes) - select first class
            shap_values_class0 = shap_values.values[:, :, 0]
            shap.dependence_plot(
                top_feature,
                shap_values_class0,
                X_explain,
                feature_names=features.columns,
                show=False
            )
        else:
            shap.dependence_plot(
                top_feature,
                shap_values.values,
                X_explain,
                feature_names=features.columns,
                show=False
            )
    else:
        raise TypeError(f"Unexpected type for shap_values.values: {type(shap_values.values)}")
    
    plt.title(f'SHAP Dependence Plot for {top_feature} ({description})', fontsize=16, pad=20)
    plt.xlabel(f'{top_feature}', fontsize=14)
    plt.ylabel('SHAP Value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/shap_dependence_{top_feature}_{description}.png', dpi=300)
    plt.close()
    print(f"SHAP dependence plot for {top_feature} saved as 'results/shap_dependence_{top_feature}_{description}.png'")
    
    return explainer, shap_values

def analyze_model_parameters(model, description=''):
    """
    Analyzes and visualizes key characteristics of model parameters and saves the data.

    Parameters:
        model (nn.Module): Trained classification model.
        description (str): Description of the current model configuration.

    Returns:
        pd.DataFrame: DataFrame containing parameter norms.
    """
    # Collect all weights and biases
    norms = []
    layers = []
    parameter_types = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            norm = np.linalg.norm(param.data.cpu().numpy())
            norms.append(norm)
            layers.append(name)
            parameter_types.append('Weight')
        elif 'bias' in name:
            norm = np.linalg.norm(param.data.cpu().numpy())
            norms.append(norm)
            layers.append(name)
            parameter_types.append('Bias')
        else:
            # Skip parameters that are neither weights nor biases
            continue

    # Ensure all lists have the same length
    assert len(layers) == len(norms) == len(parameter_types), "Mismatch in list lengths."

    # Create a DataFrame for parameter norms
    parameter_norms_df = pd.DataFrame({
        'Model': [description] * len(layers),
        'Layer': layers,
        'Parameter_Type': parameter_types,
        'Norm': norms
    })
    parameter_norms_df.to_csv(f'results/parameter_norms_{description}.csv', index=False)
    print(f"Parameter norms data saved as 'results/parameter_norms_{description}.csv'")

    # Plot parameter norms per layer with adjusted font sizes
    plt.figure(figsize=(15, 7))
    index = np.arange(len(layers))
    bar_width = 0.35
    plt.bar(index, norms, bar_width, label='Parameter Norms', color='blue')
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('L2 Norm', fontsize=14)
    plt.title(f'Parameter Norms per Layer ({description})', fontsize=16)
    plt.xticks(index, layers, rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/parameter_norms_plot_{description}.png', dpi=300)
    plt.close()
    print(f"Parameter norms plot saved as 'results/parameter_norms_plot_{description}.png'")

    # Plot distribution of weight magnitudes across layers with adjusted font sizes
    weight_magnitudes = []
    weight_layer_names = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            abs_weights = np.abs(param.data.cpu().numpy()).flatten()
            weight_magnitudes.append(abs_weights)
            weight_layer_names.append(name)

    plt.figure(figsize=(20, 10))
    for i, (w, name) in enumerate(zip(weight_magnitudes, weight_layer_names)):
        plt.subplot(len(weight_magnitudes), 1, i + 1)
        sns.kdeplot(w, fill=True, color='blue')
        plt.title(f'Weight Magnitude Distribution: {name}', fontsize=14)
        plt.xlabel('Absolute Weight Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/weight_magnitude_distributions_{description}.png', dpi=300)
    plt.close()
    print(f"Weight magnitude distributions plot saved as 'results/weight_magnitude_distributions_{description}.png'")

    # Save parameter distributions data
    parameter_distributions_df = pd.DataFrame({
        'Model': [],
        'Layer': [],
        'Parameter_Type': [],
        'Value': []
    })
    for i, name in enumerate(layers):
        param_values = model.state_dict()[name].cpu().numpy().flatten()
        param_type = parameter_types[i]
        df_temp = pd.DataFrame({
            'Model': [description] * len(param_values),
            'Layer': [name] * len(param_values),
            'Parameter_Type': [param_type] * len(param_values),
            'Value': param_values
        })
        parameter_distributions_df = pd.concat([parameter_distributions_df, df_temp], ignore_index=True)

    parameter_distributions_df.to_csv(f'results/parameter_distributions_data_{description}.csv', index=False)
    print(f"Parameter distributions data saved as 'results/parameter_distributions_data_{description}.csv'")

    return parameter_norms_df


# ============================
# 9. Implement Top Two Difference and VarMax Algorithms
# ============================

def classify_with_unknown(model, x, threshold=0.5, varmax_threshold=0.1):
    """
    Classifies input data using Top Two Difference and VarMax.
    If the difference between top two softmax scores is less than threshold,
    uses VarMax to decide if it's an unknown class.
    
    Parameters:
        model (nn.Module): Trained classification model.
        x (torch.Tensor): Input data tensor.
        threshold (float): Threshold for top two probability difference.
        varmax_threshold (float): Threshold for VarMax variance.
    
    Returns:
        list: List of predicted class indices or 'Unknown'.
    """
    with torch.no_grad():
        logits = model(x)
        softmax = nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(softmax, 2, dim=1)
        top_diff = (top_probs[:,0] - top_probs[:,1]).cpu().numpy()
        logits_np = logits.cpu().numpy()
        
    predictions = []
    for i in range(x.size(0)):
        if top_diff[i] > threshold:
            predictions.append(top_indices[i,0].item())
        else:
            # Apply VarMax
            logit = logits_np[i]
            variance = np.var(np.abs(logit))
            if variance < varmax_threshold:
                predictions.append('Unknown')  # Unknown class
            else:
                predictions.append(top_indices[i,0].item())
    return predictions

# ============================
# 10. Diffusion Model for Adversarial Perturbations
# ============================

class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=4):
        super(SimpleDiffusionModel, self).__init__()
        layers = []
        in_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(hidden_size, input_size))  # Output layer
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, t):
        return self.network(x)

class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_cumprod[:-1]], dim=0)
        
        self.sqrt_alpha = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        self.beta = self.betas  # Added for sampling step
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (forward process)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alpha[t].reshape(-1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t].reshape(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha_cumprod * noise
    
    def p_losses(self, model, x_start, t, noise=None):
        """
        Compute the loss for training the diffusion model
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t)
        loss = nn.MSELoss()(predicted_noise, noise)
        return loss

def train_diffusion_model(model, diffusion, train_loader, optimizer, num_epochs=10):
    """
    Trains the diffusion model.
    
    Parameters:
        model (nn.Module): Diffusion model to train.
        diffusion (Diffusion): Diffusion process parameters.
        train_loader (DataLoader): DataLoader for training data.
        optimizer: Optimizer for training.
        num_epochs (int): Number of training epochs.
    
    Returns:
        None
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Diffusion Model Epoch {epoch+1}/{num_epochs}"):
            x, _ = batch  # Ignore labels
            x = x.to(device)
            t = torch.randint(0, diffusion.T, (x.size(0),), device=device).long()
            loss = diffusion.p_losses(model, x, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    print("Diffusion Model Training Completed.\n")

@torch.no_grad()
def generate_adversarial_data_at_step(diffusion_model, diffusion, X_original, t, batch_size=256):
    """
    Generates adversarial perturbations at a specific diffusion time step.
    
    Parameters:
        diffusion_model (nn.Module): Trained diffusion model.
        diffusion (Diffusion): Diffusion process parameters.
        X_original (np.ndarray): Original test data.
        t (int): Diffusion time step.
        batch_size (int): Batch size for processing.
    
    Returns:
        np.ndarray: Perturbed test data.
    """
    diffusion_model.eval()
    X_original_tensor = torch.tensor(X_original, dtype=torch.float32).to(device)
    num_samples = X_original_tensor.size(0)
    perturbed_data = []
    
    for i in tqdm(range(0, num_samples, batch_size), desc=f"Generating Perturbations at t={t}"):
        batch = X_original_tensor[i:i+batch_size]
        batch_size_actual = batch.size(0)
        # Create a tensor filled with the current time step
        t_tensor = torch.full((batch_size_actual,), t, dtype=torch.long, device=device)
        # Generate noise scaled by perturbation strength (optional)
        perturbation_strength = t / diffusion.T  # Example scaling; adjust as needed
        noise = torch.randn_like(batch) * perturbation_strength
        # Add noise to the original data
        x_noisy = diffusion.q_sample(batch, t_tensor, noise)
        # Predict the noise using the diffusion model
        predicted_noise = diffusion_model(x_noisy, t_tensor)
        # Generate perturbed data by removing predicted noise
        x_perturbed = (batch - predicted_noise)  # Simple perturbation; can be adjusted
        # Ensure the perturbed data stays within bounds
        x_perturbed = torch.clamp(x_perturbed, -5, 5)  # Assuming standardized data
        perturbed_data.append(x_perturbed.cpu().numpy())
    
    perturbed_data = np.vstack(perturbed_data)[:num_samples]
    return perturbed_data

# ============================
# 11. Evaluate Classifier on Perturbed Data
# ============================

def evaluate_on_perturbed_data_at_steps(model, diffusion_model, diffusion, X_test_combined, y_test_combined, label_encoder, incremental_steps, unknown_label, num_classes, input_size, batch_size=256, model_description='', master_df=None):
    """
    Evaluates the classifier on perturbed data at specified diffusion steps and appends metrics to the master DataFrame.
    
    Parameters:
        model (nn.Module): Trained classification model.
        diffusion_model (nn.Module): Trained diffusion model.
        diffusion (Diffusion): Diffusion process parameters.
        X_test_combined (np.ndarray): Combined test data (known + unknown).
        y_test_combined (np.ndarray): Combined test labels (-1 for 'Unknown').
        label_encoder (LabelEncoder): Encoder for known classes.
        incremental_steps (list): List of diffusion time steps to evaluate.
        unknown_label (int): Integer label for 'Unknown' class.
        num_classes (int): Number of known classes.
        input_size (int): Number of input features.
        batch_size (int): Batch size for data processing.
        model_description (str): Description of the current model configuration.
        master_df (pd.DataFrame, optional): Master DataFrame to append metrics.
    
    Returns:
        None
    """
    for t in incremental_steps:
        if t == 0:
            # Clean Data
            perturbed_X = X_test_combined.copy()
            perturbed_y = y_test_combined.copy()
            print(f"\n--- Evaluating on Clean Data (t={t}) ---")
        else:
            # Perturbed Data
            perturbed_X = generate_adversarial_data_at_step(diffusion_model, diffusion, X_test_combined, t, batch_size)
            perturbed_y = y_test_combined.copy()
            print(f"\n--- Evaluating on Perturbed Data (t={t}) ---")
        
        # Replace -1 with unknown_label in y_true
        y_true_mapped = np.where(perturbed_y == -1, unknown_label, perturbed_y)
        
        # Predict with unknown handling
        X_perturbed_tensor = torch.tensor(perturbed_X, dtype=torch.float32).to(device)
        predictions = []
        for i in range(0, X_perturbed_tensor.size(0), batch_size):
            batch = X_perturbed_tensor[i:i+batch_size]
            preds = classify_with_unknown(model, batch)
            predictions.extend(preds)
        
        # Map 'Unknown' to unknown_label
        y_pred_labels = []
        for pred in predictions:
            if pred == 'Unknown':
                y_pred_labels.append(unknown_label)
            else:
                y_pred_labels.append(pred)
        y_pred_labels = np.array(y_pred_labels)
        
        # Compute metrics
        accuracy = accuracy_score(y_true_mapped, y_pred_labels)
        f1 = f1_score(y_true_mapped, y_pred_labels, average='weighted', zero_division=0)
        precision = precision_score(y_true_mapped, y_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(y_true_mapped, y_pred_labels, average='weighted', zero_division=0)
        
        # Append metrics to master_df
        if master_df is not None:
            master_df = master_df.append({
                'Model': model_description,
                'Perturbation_Step': t,
                'Epoch': None,
                'Metric_Type': 'Evaluation',
                'Metric_Name': 'Accuracy',
                'Metric_Value': accuracy
            }, ignore_index=True)
            master_df = master_df.append({
                'Model': model_description,
                'Perturbation_Step': t,
                'Epoch': None,
                'Metric_Type': 'Evaluation',
                'Metric_Name': 'F1_Score',
                'Metric_Value': f1
            }, ignore_index=True)
            master_df = master_df.append({
                'Model': model_description,
                'Perturbation_Step': t,
                'Epoch': None,
                'Metric_Type': 'Evaluation',
                'Metric_Name': 'Precision',
                'Metric_Value': precision
            }, ignore_index=True)
            master_df = master_df.append({
                'Model': model_description,
                'Perturbation_Step': t,
                'Epoch': None,
                'Metric_Type': 'Evaluation',
                'Metric_Name': 'Recall',
                'Metric_Value': recall
            }, ignore_index=True)
        
        # Save confusion matrix
        extended_label_encoder = LabelEncoder()
        extended_label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')
        cm = confusion_matrix(y_true_mapped, y_pred_labels, labels=list(range(num_classes)) + [unknown_label])
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=extended_label_encoder.classes_,
                    yticklabels=extended_label_encoder.classes_)
        plt.title(f'Confusion Matrix (t={t}) - {model_description}', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=45, va='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{model_description}_t_{t}.png', dpi=300)
        plt.close()
        print(f"Confusion matrix at t={t} saved as 'results/confusion_matrix_{model_description}_t_{t}.png'\n")

# ============================
# 12. Main Experiment Loop
# ============================

def main():
    # Load and preprocess data
    (
        device_global,
        features,
        scaler,
        label_encoder,
        num_classes,
        unknown_label,
        train_loader,
        test_loader_known,
        test_loader_combined,
        X_train,
        X_test_combined,
        y_test_combined,
        input_size  # Unpack input_size here
    ) = load_and_preprocess_data()
    
    global device
    device = device_global  # Set the global device variable for other functions

    # Print input_size to confirm it's correctly unpacked
    print(f"Input size in main: {input_size}")
    
    # Initialize a Master DataFrame to collect all graph data
    master_df = pd.DataFrame(columns=['Model', 'Perturbation_Step', 'Epoch', 'Metric_Type', 'Metric_Name', 'Metric_Value'])
    
    # Initialize DataFrames for Sensitivity, SHAP, and Parameter Distributions
    sensitivity_data = []
    shap_data = []
    parameter_distributions_data = []
    
    # ============================
    # 1. Train the Diffusion Model
    # ============================
    diffusion = Diffusion(T=1000, beta_start=1e-4, beta_end=0.02, device=device)
    diffusion_model = SimpleDiffusionModel(input_size=input_size, hidden_size=512, num_layers=4).to(device)
    diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-3)
    
    print("\n=== Training the Diffusion Model for Adversarial Perturbations ===")
    train_diffusion_model(diffusion_model, diffusion, train_loader, diffusion_optimizer, num_epochs=10)
    
    # ============================
    # 2. Generate Adversarial Perturbations
    # ============================
    print("\n=== Generating Adversarial Perturbations Using Diffusion Model ===")
    # Initial perturbation with t=0 (clean data)
    perturbed_X_test = generate_adversarial_data_at_step(diffusion_model, diffusion, X_test_combined, t=0, batch_size=256)
    perturbed_y_test = y_test_combined.copy()  # -1 for 'Unknown'
    
    # Save perturbed data as CSV
    # Map 'Unknown' labels to their original classes for unknown samples
    perturbed_labels = []
    for label in perturbed_y_test:
        if label == -1:
            perturbed_labels.append('Unknown')
        else:
            perturbed_labels.append(label_encoder.inverse_transform([label])[0])
    
    perturbed_data_df = pd.DataFrame(perturbed_X_test, columns=features.columns)
    perturbed_data_df['Label'] = perturbed_labels
    perturbed_data_df.to_csv('results/perturbed_test_data_t0.csv', index=False)
    print("Perturbed test data at t=0 saved as 'results/perturbed_test_data_t0.csv'\n")
    
    # ============================
    # 3. Train the Classifier on Clean Data
    # ============================
    print("\n=== Training the Classifier on Clean Data ===")
    model_configs = [
        {'hidden_sizes': [128, 64], 'activation': 'ReLU', 'dropout_rate': 0.0, 'description': '2_Layers_ReLU_NoDropout'},
        {'hidden_sizes': [256, 128, 64], 'activation': 'ReLU', 'dropout_rate': 0.0, 'description': '3_Layers_ReLU_NoDropout'},
        {'hidden_sizes': [128, 64], 'activation': 'LeakyReLU', 'dropout_rate': 0.0, 'description': '2_Layers_LeakyReLU_NoDropout'},
        {'hidden_sizes': [256, 128, 64], 'activation': 'LeakyReLU', 'dropout_rate': 0.0, 'description': '3_Layers_LeakyReLU_NoDropout'},
        {'hidden_sizes': [128, 64], 'activation': 'ELU', 'dropout_rate': 0.0, 'description': '2_Layers_ELU_NoDropout'},
        {'hidden_sizes': [256, 128, 64], 'activation': 'ELU', 'dropout_rate': 0.0, 'description': '3_Layers_ELU_NoDropout'},
        # With Dropout
        {'hidden_sizes': [128, 64], 'activation': 'ReLU', 'dropout_rate': 0.5, 'description': '2_Layers_ReLU_Dropout'},
        {'hidden_sizes': [256, 128, 64], 'activation': 'ReLU', 'dropout_rate': 0.5, 'description': '3_Layers_ReLU_Dropout'},
        {'hidden_sizes': [128, 64], 'activation': 'LeakyReLU', 'dropout_rate': 0.5, 'description': '2_Layers_LeakyReLU_Dropout'},
        {'hidden_sizes': [256, 128, 64], 'activation': 'LeakyReLU', 'dropout_rate': 0.5, 'description': '3_Layers_LeakyReLU_Dropout'},
        {'hidden_sizes': [128, 64], 'activation': 'ELU', 'dropout_rate': 0.5, 'description': '2_Layers_ELU_Dropout'},
        {'hidden_sizes': [256, 128, 64], 'activation': 'ELU', 'dropout_rate': 0.5, 'description': '3_Layers_ELU_Dropout'},
    ]

    experiment_results = []

    for config in model_configs:
        print(f"\n=== Training Classifier: {config['description']} ===")
        model = Net(
            input_size=input_size,
            num_classes=num_classes,  # Only known classes
            hidden_sizes=config['hidden_sizes'],
            activation=config['activation'],
            dropout_rate=config['dropout_rate']
        ).to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train the classifier with test_loader_known as validation_loader and append metrics to master_df
        train_losses, val_f1_scores, val_accuracy_scores, val_roc_auc_scores, master_df = train_classifier(
            model, train_loader, criterion, optimizer, num_epochs=10, validation_loader=test_loader_known, device=device, master_df=master_df, model_description=config['description']
        )
        
        # Evaluate on clean test data (known classes only)
        evaluate_classifier(model, test_loader_known, label_encoder, description=config['description'] + '_clean', features=features)
        
        # Sensitivity Analysis (Gradients) on Clean Data
        sensitivity_clean = sensitivity_analysis_classifier(
            model,
            test_loader_known,
            label_encoder,
            num_classes,  # Passed as argument
            description=f"{config['description']}_clean",
            features=features,
            input_size=input_size,
            master_df=master_df
        )
        
        # SHAP Analysis on Clean Data
        explainer_clean, shap_values_clean = shap_analysis_classifier(
            model,
            test_loader_known,
            label_encoder,
            description=f"{config['description']}_clean",
            features=features,
            X_train=X_train,
            master_df=master_df
        )
        
        # Analyze and visualize model parameters
        parameter_norms_df = analyze_model_parameters(model, description=config['description'])
        
        # Evaluate on Adversarial Data
        print(f"\n=== Performing Sensitivity and SHAP Analysis on Adversarial Data for {config['description']} ===")
        
        # Generate adversarial data for evaluation (Example: t=200)
        adversarial_X = generate_adversarial_data_at_step(
            diffusion_model, diffusion, X_test_combined, t=200, batch_size=256
        )
        adversarial_y = y_test_combined.copy()
        
        # Replace -1 with unknown_label in y_true
        y_true_adversarial = np.where(adversarial_y == -1, unknown_label, adversarial_y)
        
        # Create DataLoader for adversarial data
        adversarial_X_tensor = torch.tensor(adversarial_X, dtype=torch.float32).to(device)
        adversarial_y_tensor = torch.tensor(y_true_adversarial, dtype=torch.long).to(device)
        adversarial_dataset = TensorDataset(adversarial_X_tensor, adversarial_y_tensor)
        adversarial_loader = DataLoader(adversarial_dataset, batch_size=batch_size)
        
        # Sensitivity Analysis (Gradients) on Adversarial Data
        sensitivity_adversarial = sensitivity_analysis_classifier(
            model,
            adversarial_loader,
            label_encoder,
            num_classes,  # Passed as argument
            description=f"{config['description']}_adversarial_t200",
            features=features,
            input_size=input_size,
            master_df=master_df
        )
        
        # SHAP Analysis on Adversarial Data
        explainer_adversarial, shap_values_adversarial = shap_analysis_classifier(
            model,
            adversarial_loader,
            label_encoder,
            description=f"{config['description']}_adversarial_t200",
            features=features,
            X_train=X_train,
            master_df=master_df
        )
        
        # Evaluate on multiple diffusion steps and append metrics to master_df
        evaluate_on_perturbed_data_at_steps(
            model,
            diffusion_model,
            diffusion,
            X_test_combined,
            y_test_combined,
            label_encoder,
            incremental_steps,
            unknown_label,
            num_classes,
            input_size,
            batch_size=batch_size,
            model_description=config['description'],
            master_df=master_df
        )
        
        # Store results
        experiment_results.append({
            'model_description': config['description'],
            'model': model,  # Store model for later evaluation
            'train_losses': train_losses,
            'val_f1_scores': val_f1_scores,
            'val_accuracy_scores': val_accuracy_scores,
            'val_roc_auc_scores': val_roc_auc_scores,
            'sensitivity_clean': sensitivity_clean,
            'sensitivity_adversarial': sensitivity_adversarial,
            'shap_values_clean': shap_values_clean,
            'shap_values_adversarial': shap_values_adversarial,
            'parameter_norms': parameter_norms_df
        })

    # ============================
    # 4. Save the Master DataFrame
    # ============================

    if not master_df.empty:
        master_df.to_csv('results/graph_data.csv', index=False)
        print("All graph-related metrics saved to 'results/graph_data.csv'")
    else:
        print("Master DataFrame is empty. No graph data to save.")
    
    print("\n=== All Experiments Completed ===")
    print("All results and data have been saved in the 'results/' directory.")

if __name__ == "__main__":
    main()

