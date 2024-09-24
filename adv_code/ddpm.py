import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import (
    setup_logging,
    get_data,
    plot_data,
    evaluate_classifier,
    compute_mmd,
    visualize_diffusion_process,
    visualize_packet_changes,
    save_adversarial_samples,
    load_adversarial_samples,
    load_model
)
from modules import UNet, Classifier
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S"
)


class Diffusion:
    def __init__(self, img_size, device, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.img_size = img_size
        self.device = device
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_data(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def generate_adversarial(self, model, x, t):
        model.eval()
        with torch.no_grad():
            x_noisy, _ = self.noise_data(x, t)
            predicted_noise = model(x_noisy, t)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            x_adv = (1 / torch.sqrt(alpha)) * (
                x_noisy - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            )
        model.train()
        return x_adv.squeeze()


def main(args):
    setup_logging(args.run_name)
    device = args.device
    (
        diffusion_dataloader,
        classifier_dataloader,
        data_dim,
        scaler,
        label_encoder,
        feature_names
    ) = get_data(args)
    img_size = 2 ** int(np.ceil(np.log2(np.sqrt(data_dim))))
    args.data_dim = data_dim  # Save data_dim in args for later use

    # Initialize models
    diffusion_model = UNet(c_in=1, c_out=1, img_size=img_size).to(device)
    classifier = Classifier(input_dim=data_dim, num_classes=len(label_encoder.classes_)).to(device)

    # Load pre-trained models if specified
    if args.load_classifier and os.path.exists(args.classifier_model_path):
        classifier.load_state_dict(torch.load(args.classifier_model_path))
        logging.info(f"Loaded pre-trained classifier model from {args.classifier_model_path}")

    if args.load_diffusion and os.path.exists(args.diffusion_model_path):
        diffusion_model.load_state_dict(torch.load(args.diffusion_model_path))
        logging.info(f"Loaded pre-trained diffusion model from {args.diffusion_model_path}")

    # Optimizers and loss functions
    optimizer_diffusion = optim.AdamW(diffusion_model.parameters(), lr=args.lr)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    diffusion = Diffusion(img_size=img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))

    # Perform the specified operation
    if args.operation == 'train_classifier':
        train_classifier(classifier, classifier_dataloader, optimizer_classifier, ce_loss, args, logger, device)
    elif args.operation == 'train_diffusion':
        train_diffusion(diffusion_model, diffusion_dataloader, optimizer_diffusion, mse_loss, diffusion, args, logger, device)
    elif args.operation == 'generate_adversarial':
        generate_and_save_adversarial_samples(diffusion_model, diffusion_dataloader, classifier, diffusion, scaler, label_encoder, feature_names, args, device)
    elif args.operation == 'evaluate_classifier':
        evaluate_saved_adversarial_samples(classifier, classifier_dataloader, label_encoder, args, device)


def train_classifier(classifier, dataloader, optimizer, loss_fn, args, logger, device):
    classifier.train()
    logging.info("Training classifier...")
    for epoch in range(args.classifier_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Classifier Training Epoch {epoch+1}/{args.classifier_epochs}")
        for data, labels in pbar:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(Loss=loss.item())
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{args.classifier_epochs}, Loss: {avg_loss:.4f}")
        torch.save(classifier.state_dict(), os.path.join("models", args.run_name, f"classifier_epoch_{epoch+1}.pt"))


def train_diffusion(model, dataloader, optimizer, loss_fn, diffusion, args, logger, device):
    model.train()
    logging.info("Training diffusion model...")
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Diffusion Training Epoch {epoch+1}/{args.epochs}")
        for i, (data, _) in enumerate(pbar):
            data = data.to(device)
            t = diffusion.sample_timesteps(data.shape[0]).to(device)
            x_t, noise = diffusion.noise_data(data, t)
            predicted_noise = model(x_t, t)
            loss = loss_fn(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * len(dataloader) + i)
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"diffusion_epoch_{epoch+1}.pt"))


def generate_and_save_adversarial_samples(model, dataloader, classifier, diffusion, scaler, label_encoder, feature_names, args, device):
    # Load the diffusion model
    model = load_model(model, args.diffusion_model_path, device)
    model.eval()
    classifier.eval()
    logging.info("Generating adversarial examples...")
    adversarial_examples = []
    true_labels = []

    for data, labels in tqdm(dataloader, desc="Generating Adversarial Samples"):
        data = data.to(device)
        labels = labels.to(device)
        t = diffusion.sample_timesteps(data.shape[0]).to(device)
        x_adv = diffusion.generate_adversarial(model, data, t)
        adversarial_examples.append(x_adv)
        true_labels.append(labels)

    adversarial_examples = torch.cat(adversarial_examples)
    true_labels = torch.cat(true_labels)

    # Reshape adversarial examples to match classifier input
    adversarial_examples = adversarial_examples.view(adversarial_examples.size(0), -1)
    adversarial_examples = adversarial_examples[:, :args.data_dim]  # Crop or pad to match data_dim

    # Save adversarial samples
    save_adversarial_samples(adversarial_examples, true_labels, args)
    logging.info("Adversarial samples generated and saved successfully.")


def evaluate_saved_adversarial_samples(classifier, classifier_dataloader, label_encoder, args, device):
    # Load saved adversarial samples
    adversarial_examples, true_labels = load_adversarial_samples(args)
    adversarial_examples = adversarial_examples.to(device)
    true_labels = true_labels.to(device)

    # Evaluate classifier on normal and adversarial data
    logging.info("Evaluating classifier on saved adversarial data...")
    evaluate_classifier(
        classifier,
        classifier_dataloader,
        adversarial_examples,
        true_labels,
        epoch=0,  # Assuming single epoch for evaluation
        args=args,
        label_encoder=label_encoder,
        device=device,
        feature_names=None  # Adjust based on your use case
    )


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="DDPM_CIC")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--classifier_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset_path', type=str, default="path_to_cic_dataset.csv")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--operation', type=str, choices=['train_classifier', 'train_diffusion', 'generate_adversarial', 'evaluate_classifier'], default='generate_adversarial', help='Operation to perform')
    parser.add_argument('--classifier_model_path', type=str, default='models/DDPM_CIC/classifier.pt', help='Path to the classifier model')
    parser.add_argument('--diffusion_model_path', type=str, default='models/DDPM_CIC/diffusion.pt', help='Path to the diffusion model')
    parser.add_argument('--load_classifier', action='store_true', help='Flag to load a pre-trained classifier model')
    parser.add_argument('--load_diffusion', action='store_true', help='Flag to load a pre-trained diffusion model')
    parser.add_argument('--adversarial_samples_path', type=str, default='adversarial_samples.pt', help='Path to save/load adversarial samples')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    launch()
