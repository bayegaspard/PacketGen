import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%I:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join("results", run_name, "log.txt")),
            logging.StreamHandler()
        ]
    )


def get_data(args):
    # Load the dataset
    df = pd.read_csv(args.dataset_path)

    # Preprocess the dataframe
    labels = df['Label']
    df = df.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label'], axis=1)
    df = df.fillna(df.mean())

    # Save feature names
    feature_names = df.columns.tolist()

    # Feature scaling
    scaler = StandardScaler()
    features = scaler.fit_transform(df.values)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Reshape features to 2D for diffusion model
    data_dim = features.shape[1]
    img_size = 2 ** int(np.ceil(np.log2(np.sqrt(data_dim))))
    padded_features = np.zeros((features.shape[0], img_size * img_size))
    padded_features[:, :data_dim] = features
    reshaped_features = padded_features.reshape(-1, 1, img_size, img_size)

    # Create datasets and dataloaders

    # Diffusion dataset and dataloader (include labels)
    diffusion_dataset = TensorDataset(
        torch.tensor(reshaped_features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    diffusion_dataloader = DataLoader(diffusion_dataset, batch_size=args.batch_size, shuffle=True)

    # Classifier dataset and dataloader
    classifier_dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    classifier_dataloader = DataLoader(classifier_dataset, batch_size=args.batch_size, shuffle=True)

    return diffusion_dataloader, classifier_dataloader, data_dim, scaler, label_encoder, feature_names


def plot_data(sampled_data_np, epoch, args):
    # Plot histograms of selected features
    plt.figure(figsize=(15, 5))
    num_features = sampled_data_np.shape[1]
    num_plots = min(5, num_features)
    for i in range(num_plots):
        plt.subplot(1, num_plots, i + 1)
        plt.hist(sampled_data_np[:, i], bins=50, alpha=0.7, label='Generated')
        plt.title(f'Feature {i + 1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join("results", args.run_name, f"hist_epoch_{epoch+1}.png"))
    plt.close()


def evaluate_classifier(classifier, classifier_dataloader, adversarial_examples, true_labels, epoch, args, label_encoder, device, feature_names):
    classifier.eval()
    # Evaluate on normal data
    all_preds = []
    all_labels = []
    all_data = []
    with torch.no_grad():
        for data, labels in classifier_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = classifier(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_data.append(data.cpu().numpy())

    all_data = np.concatenate(all_data, axis=0)

    # Get unique labels
    unique_labels = np.unique(np.concatenate((all_labels, all_preds)))
    class_names = label_encoder.inverse_transform(unique_labels)

    # Confusion matrix for normal data
    cm_normal = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    cm_normal_normalized = cm_normal.astype('float') / cm_normal.sum(axis=1)[:, np.newaxis]
    disp_normal = ConfusionMatrixDisplay(confusion_matrix=cm_normal_normalized, display_labels=class_names)
    disp_normal.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='.2f')
    plt.title(f'Normalized Confusion Matrix on Normal Data at Epoch {epoch+1}')
    plt.tight_layout()
    plt.savefig(os.path.join("results", args.run_name, f"normalized_confusion_matrix_normal_epoch_{epoch+1}.png"))
    plt.close()

    # Evaluate on adversarial data
    adversarial_examples = adversarial_examples.to(device)
    true_labels = true_labels.to(device)
    
    # Reshape adversarial examples to match classifier input
    adversarial_examples = adversarial_examples.view(adversarial_examples.size(0), -1)

    # Ensure input shape matches classifier input dimension
    adversarial_examples = adversarial_examples[:, :args.data_dim]  # Crop or pad to match data_dim

    with torch.no_grad():
        outputs_adv = classifier(adversarial_examples)
        _, preds_adv = torch.max(outputs_adv, 1)
    preds_adv = preds_adv.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()

    # Get unique labels for adversarial data
    unique_labels_adv = np.unique(np.concatenate((true_labels_np, preds_adv)))
    class_names_adv = label_encoder.inverse_transform(unique_labels_adv)

    # Confusion matrix for adversarial data
    cm_adv = confusion_matrix(true_labels_np, preds_adv, labels=unique_labels_adv)
    cm_adv_normalized = cm_adv.astype('float') / cm_adv.sum(axis=1)[:, np.newaxis]
    disp_adv = ConfusionMatrixDisplay(confusion_matrix=cm_adv_normalized, display_labels=class_names_adv)
    disp_adv.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='.2f')
    plt.title(f'Normalized Confusion Matrix on Adversarial Data at Epoch {epoch+1}')
    plt.tight_layout()
    plt.savefig(os.path.join("results", args.run_name, f"normalized_confusion_matrix_adversarial_epoch_{epoch+1}.png"))
    plt.close()

    # Save confusion matrices as CSV
    np.savetxt(os.path.join("results", args.run_name, f"cm_normal_epoch_{epoch+1}.csv"), cm_normal, delimiter=",")
    np.savetxt(os.path.join("results", args.run_name, f"cm_adversarial_epoch_{epoch+1}.csv"), cm_adv, delimiter=",")

    # Print classification reports
    report_normal = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    report_adv = classification_report(true_labels_np, preds_adv, target_names=class_names_adv, zero_division=0)
    with open(os.path.join("results", args.run_name, f"classification_report_epoch_{epoch+1}.txt"), 'w') as f:
        f.write("Classification Report on Normal Data:\n")
        f.write(report_normal)
        f.write("\nClassification Report on Adversarial Data:\n")
        f.write(report_adv)


def compute_mmd(sampled_data_np, args):
    from sklearn.metrics.pairwise import polynomial_kernel
    import pandas as pd

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
    import torch
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import os

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


def visualize_packet_changes(classifier_dataloader, adversarial_examples, epoch, args, feature_names):
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

        # Plot specific features
        plt.figure(figsize=(12, 6))
        selected_indices = np.argsort(np.abs(pert))[-5:]  # Top 5 most perturbed features
        selected_features = [feature_names[idx] for idx in selected_indices]

        for idx, feature_idx in enumerate(selected_indices):
            feature_name = feature_names[feature_idx]
            plt.subplot(2, 3, idx+1)
            plt.bar(['Original', 'Adversarial'], [orig[feature_idx], adv[feature_idx]])
            plt.title(f'Feature: {feature_name}')
            plt.ylabel('Value')

        plt.suptitle(f'Perturbations for Sample {i+1} at Epoch {epoch+1}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join("results", args.run_name, f"perturbations_sample_{i+1}_epoch_{epoch+1}.png"))
        plt.close()

        # Plot the perturbation vector
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(pert)), pert)
        plt.title(f'Perturbation Vector for Sample {i+1} at Epoch {epoch+1}')
        plt.xlabel('Feature Index')
        plt.ylabel('Perturbation Value')
        plt.tight_layout()
        plt.savefig(os.path.join("results", args.run_name, f"perturbation_vector_sample_{i+1}_epoch_{epoch+1}.png"))
        plt.close()

        # Save perturbations as CSV
        pert_df = pd.DataFrame({
            'Feature': feature_names,
            'Original Value': orig,
            'Adversarial Value': adv,
            'Perturbation': pert
        })
        pert_df.to_csv(os.path.join("results", args.run_name, f"perturbations_sample_{i+1}_epoch_{epoch+1}.csv"), index=False)


def save_adversarial_samples(adversarial_examples, true_labels, args):
    adversarial_samples_path = args.adversarial_samples_path
    torch.save({
        'adversarial_examples': adversarial_examples,
        'true_labels': true_labels
    }, adversarial_samples_path)
    logging.info(f"Adversarial samples saved to {adversarial_samples_path}")


def load_adversarial_samples(args):
    adversarial_samples_path = args.adversarial_samples_path
    checkpoint = torch.load(adversarial_samples_path)
    logging.info(f"Adversarial samples loaded from {adversarial_samples_path}")
    return checkpoint['adversarial_examples'], checkpoint['true_labels']


def load_model(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        logging.info(f"Model loaded from {path}")
    else:
        logging.error(f"Model path {path} does not exist.")
    return model
