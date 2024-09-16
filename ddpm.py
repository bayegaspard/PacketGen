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
    visualize_packet_changes
)
from modules import UNet, Classifier
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=16, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size  # Represents the size of the reshaped data
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_data(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def generate_adversarial(self, model, x, t):
        model.eval()
        x = x.clone().detach().to(self.device)
        t = t.to(self.device)
        x.requires_grad = True

        # Ensure model parameters are not updated
        for param in model.parameters():
            param.requires_grad = False

        # Enable gradient computation
        with torch.enable_grad():
            # Forward pass through the diffusion model
            predicted_noise = model(x, t)
            loss = -predicted_noise.mean()  # Maximize the noise to create adversarial effect
            loss.backward()
            perturbation = x.grad.sign() * 0.1  # Scaled perturbation
            x_adv = x + perturbation
            x_adv = torch.clamp(x_adv, -1, 1)  # Ensure data stays within valid range

        model.train()
        # Reset model parameters to require gradients for future training
        for param in model.parameters():
            param.requires_grad = True

        return x_adv.detach()

    def sample(self, model, n):
        logging.info(f"Sampling {n} new data points....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = torch.full((n,), i, dtype=torch.long).to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
                ) + torch.sqrt(beta) * noise
        model.train()
        # Reshape back to original feature dimensions
        x = x.view(n, -1)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    diffusion_dataloader, classifier_dataloader, data_dim, scaler, label_encoder = get_data(args)
    img_size = int(np.ceil(np.sqrt(data_dim)))
    args.data_dim = data_dim  # Save data_dim in args for later use

    # Initialize models
    diffusion_model = UNet(c_in=1, c_out=1, img_size=img_size).to(device)
    classifier = Classifier(input_dim=data_dim, num_classes=len(label_encoder.classes_)).to(device)

    # Optimizers and loss functions
    optimizer_diffusion = optim.AdamW(diffusion_model.parameters(), lr=args.lr)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    diffusion = Diffusion(img_size=img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(diffusion_dataloader)

    # Train classifier on real data
    logging.info("Training classifier on real data...")
    classifier.train()
    for epoch_cls in range(args.classifier_epochs):
        total_loss = 0
        pbar_cls = tqdm(classifier_dataloader, desc=f"Classifier Training Epoch {epoch_cls+1}/{args.classifier_epochs}")
        for data, labels in pbar_cls:
            data = data.to(device)
            labels = labels.to(device)
            optimizer_classifier.zero_grad()
            outputs = classifier(data)
            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer_classifier.step()
            total_loss += loss.item()
            pbar_cls.set_postfix(Loss=loss.item())
        avg_loss = total_loss / len(classifier_dataloader)
        logging.info(f"Classifier Training Epoch {epoch_cls + 1}/{args.classifier_epochs}, Loss: {avg_loss:.4f}")

        # Save classifier model
        torch.save(classifier.state_dict(), os.path.join("models", args.run_name, f"classifier_epoch_{epoch_cls}.pt"))

    # Training diffusion model
    for epoch in range(args.epochs):
        logging.info(f"Starting diffusion model training epoch {epoch+1}/{args.epochs}:")
        pbar = tqdm(diffusion_dataloader)
        diffusion_model.train()
        for i, (data, labels) in enumerate(pbar):
            data = data.to(device)
            t = diffusion.sample_timesteps(data.shape[0]).to(device)
            x_t, noise = diffusion.noise_data(data, t)
            predicted_noise = diffusion_model(x_t, t)
            loss = mse_loss(noise, predicted_noise)

            optimizer_diffusion.zero_grad()
            loss.backward()
            optimizer_diffusion.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # Save diffusion model
        torch.save(diffusion_model.state_dict(), os.path.join("models", args.run_name, f"diffusion_epoch_{epoch}.pt"))

        # Sample new data points
        sampled_data = diffusion.sample(diffusion_model, n=1000)
        # Inverse transform the data back to original scale
        sampled_data_np = scaler.inverse_transform(sampled_data.cpu().numpy()[:, :data_dim])
        # Save the sampled data
        np.savetxt(os.path.join("results", args.run_name, f"sampled_epoch_{epoch}.csv"), sampled_data_np, delimiter=",")

        # Generate adversarial examples
        logging.info("Generating adversarial examples...")
        adversarial_examples = []
        true_labels = []
        classifier.eval()
        diffusion_model.eval()

        for data, labels in diffusion_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(data.shape[0]).to(device)
            x_adv = diffusion.generate_adversarial(diffusion_model, data, t)
            adversarial_examples.append(x_adv)
            true_labels.append(labels)
        adversarial_examples = torch.cat(adversarial_examples)
        true_labels = torch.cat(true_labels)

        # Prepare adversarial examples for classifier
        adversarial_examples = adversarial_examples.view(adversarial_examples.size(0), -1)
        # Take only the first data_dim features (original features before padding)
        adversarial_examples = adversarial_examples[:, :data_dim]

        # Evaluate classifier on normal and adversarial data
        evaluate_classifier(classifier, classifier_dataloader, adversarial_examples, true_labels, epoch, args, label_encoder, device)

        # Plotting
        plot_data(sampled_data_np, epoch, args)
        visualize_packet_changes(classifier_dataloader, adversarial_examples, epoch, args)

        # Compute MMD
        compute_mmd(sampled_data_np, args)

        # Visualize diffusion process
        visualize_diffusion_process(diffusion_model, diffusion, (img_size, img_size), args)


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
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    launch()
