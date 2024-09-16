import os
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def get_data(args):
    # Load the dataset
    df = pd.read_csv(args.dataset_path)

    # Preprocess the dataframe
    labels = df['Label']
    df = df.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label'], axis=1)
    df = df.fillna(df.mean())

    # Feature scaling
    scaler = StandardScaler()
    features = scaler.fit_transform(df.values)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Reshape features to 2D for diffusion model
    data_dim = features.shape[1]
    img_size = int(np.ceil(np.sqrt(data_dim)))
    padded_features = np.zeros((features.shape[0], img_size * img_size))
    padded_features[:, :data_dim] = features
    reshaped_features = padded_features.reshape(-1, 1, img_size, img_size)

    # Create datasets and dataloaders
    class DiffusionDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    class ClassifierDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    diffusion_dataset = DiffusionDataset(reshaped_features, labels)
    classifier_dataset = ClassifierDataset(features, labels)

    diffusion_dataloader = DataLoader(diffusion_dataset, batch_size=args.batch_size, shuffle=True)
    classifier_dataloader = DataLoader(classifier_dataset, batch_size=args.batch_size, shuffle=True)

    return diffusion_dataloader, classifier_dataloader, data_dim, scaler, label_encoder


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def plot_data(sampled_data_np, epoch, args):
    # Plot histograms of selected features
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.hist(sampled_data_np[:, i], bins=50, alpha=0.7, label='Generated')
        plt.title(f'Feature {i + 1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join("results", args.run_name, f"hist_epoch_{epoch}.png"))
    plt.close()



def evaluate_classifier(classifier, classifier_dataloader, adversarial_examples, true_labels, epoch, args, label_encoder, device):
    from sklearn.metrics import classification_report

    classifier.eval()
    # Evaluate on normal data
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in classifier_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = classifier(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion matrix per class for normal data
    cm_normal = confusion_matrix(all_labels, all_preds, labels=range(len(label_encoder.classes_)))
    disp_normal = ConfusionMatrixDisplay(confusion_matrix=cm_normal, display_labels=label_encoder.classes_)
    disp_normal.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f'Confusion Matrix on Normal Data at Epoch {epoch+1}')
    plt.tight_layout()
    plt.savefig(os.path.join("results", args.run_name, f"confusion_matrix_normal_epoch_{epoch+1}.png"))
    plt.close()

    # Evaluate on adversarial data
    adversarial_examples = adversarial_examples.to(device)
    true_labels = true_labels.to(device)
    with torch.no_grad():
        outputs_adv = classifier(adversarial_examples)
        _, preds_adv = torch.max(outputs_adv, 1)

    # Confusion matrix per class for adversarial data
    cm_adv = confusion_matrix(true_labels.cpu().numpy(), preds_adv.cpu().numpy(), labels=range(len(label_encoder.classes_)))
    disp_adv = ConfusionMatrixDisplay(confusion_matrix=cm_adv, display_labels=label_encoder.classes_)
    disp_adv.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f'Confusion Matrix on Adversarial Data at Epoch {epoch+1}')
    plt.tight_layout()
    plt.savefig(os.path.join("results", args.run_name, f"confusion_matrix_adversarial_epoch_{epoch+1}.png"))
    plt.close()

    # Save confusion matrices as CSV
    np.savetxt(os.path.join("results", args.run_name, f"cm_normal_epoch_{epoch+1}.csv"), cm_normal, delimiter=",")
    np.savetxt(os.path.join("results", args.run_name, f"cm_adversarial_epoch_{epoch+1}.csv"), cm_adv, delimiter=",")

    # Print classification reports
    report_normal = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
    report_adv = classification_report(true_labels.cpu().numpy(), preds_adv.cpu().numpy(), target_names=label_encoder.classes_)
    with open(os.path.join("results", args.run_name, f"classification_report_epoch_{epoch+1}.txt"), 'w') as f:
        f.write("Classification Report on Normal Data:\n")
        f.write(report_normal)
        f.write("\nClassification Report on Adversarial Data:\n")
        f.write(report_adv)



def compute_mmd(sampled_data_np, args):
    from sklearn.metrics.pairwise import polynomial_kernel

    # Load real data
    df_real = pd.read_csv(args.dataset_path)
    df_real = df_real.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label'], axis=1)
    df_real = df_real.fillna(df_real.mean())
    scaler = StandardScaler()
    features_real = scaler.fit_transform(df_real.values)

    X = features_real
    Y = sampled_data_np

    XX = polynomial_kernel(X, X)
    YY = polynomial_kernel(Y, Y)
    XY = polynomial_kernel(X, Y)

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    with open(os.path.join("results", args.run_name, "mmd_scores.txt"), 'a') as f:
        f.write(f"MMD Score: {mmd}\n")

def visualize_diffusion_process(model, diffusion, data_shape, args):
    from sklearn.manifold import TSNE

    n_samples = 500
    x = torch.randn((n_samples, 1, *data_shape)).to(args.device)
    timesteps_to_plot = [0, diffusion.noise_steps // 4, diffusion.noise_steps // 2, diffusion.noise_steps - 1]
    embeddings = []

    with torch.no_grad():
        for i in reversed(range(1, diffusion.noise_steps)):
            t = torch.full((n_samples,), i, dtype=torch.long).to(args.device)
            predicted_noise = model(x, t)
            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(beta) * noise

            if i in timesteps_to_plot:
                x_flat = x.view(n_samples, -1).cpu().numpy()
                embeddings.append((i, x_flat))

    # Apply t-SNE and plot
    for timestep, data in embeddings:
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        data_2d = tsne.fit_transform(data)
        plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5)
        plt.title(f'Data at Timestep {timestep}')
        plt.savefig(os.path.join("results", args.run_name, f"diffusion_step_{timestep}.png"))
        plt.close()



def visualize_packet_changes(classifier_dataloader, adversarial_examples, epoch, args):
    # Get some original data samples
    data_iter = iter(classifier_dataloader)
    original_data, _ = next(data_iter)
    num_samples = min(5, original_data.size(0), adversarial_examples.size(0))
    original_data = original_data[:num_samples].cpu().numpy()
    adversarial_data = adversarial_examples[:num_samples].cpu().numpy()

    # Calculate perturbations
    perturbations = adversarial_data - original_data

    # Plot the changes in packet features
    for i in range(num_samples):
        orig = original_data[i]
        adv = adversarial_data[i]
        pert = perturbations[i]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.plot(orig)
        plt.title(f'Original Sample {i+1}')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')

        plt.subplot(1, 3, 2)
        plt.plot(adv)
        plt.title(f'Adversarial Sample {i+1}')
        plt.xlabel('Feature Index')

        plt.subplot(1, 3, 3)
        plt.plot(pert)
        plt.title(f'Perturbation {i+1}')
        plt.xlabel('Feature Index')

        plt.tight_layout()
        plt.savefig(os.path.join("results", args.run_name, f"packet_changes_sample_{i+1}_epoch_{epoch+1}.png"))
        plt.close()

        # Heatmap of perturbations
        plt.figure(figsize=(8, 4))
        sns.heatmap(pert.reshape(1, -1), cmap='coolwarm', center=0, cbar=True)
        plt.title(f'Perturbation Heatmap for Sample {i+1} at Epoch {epoch+1}')
        plt.xlabel('Feature Index')
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join("results", args.run_name, f"perturbation_heatmap_sample_{i+1}_epoch_{epoch+1}.png"))
        plt.close()

        # Bar plot of absolute perturbations
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(pert)), np.abs(pert))
        plt.title(f'Absolute Perturbation Magnitude for Sample {i+1} at Epoch {epoch+1}')
        plt.xlabel('Feature Index')
        plt.ylabel('Absolute Perturbation')
        plt.tight_layout()
        plt.savefig(os.path.join("results", args.run_name, f"perturbation_magnitude_sample_{i+1}_epoch_{epoch+1}.png"))
        plt.close()

