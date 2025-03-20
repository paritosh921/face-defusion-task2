import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# Import DINOv2 model for embeddings extraction
from torchvision.models import vits16_dinov2

class EmbeddingFaceDataset(Dataset):
    def __init__(self, img_dir, embedding_path=None, transform=None, encoder=None, device=None):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-computed embeddings if available
        if embedding_path and os.path.exists(embedding_path):
            self.embeddings = torch.load(embedding_path)
            print(f"Loaded {len(self.embeddings)} pre-computed embeddings")
        else:
            self.embeddings = None
            self.encoder = encoder
            if self.encoder is None:
                raise ValueError("Either embedding_path or encoder must be provided")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get embedding
        if self.embeddings is not None:
            embedding = self.embeddings[idx]
        else:
            with torch.no_grad():
                self.encoder.eval()
                embedding = self.encoder(image.unsqueeze(0).to(self.device))
                embedding = embedding.squeeze(0).cpu()
        
        return image, embedding

def get_transforms():
    # Define transforms for the generative model (128x128 images)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform

# Define U-Net blocks for the diffusion model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t_emb, c_emb):
        x = self.maxpool_conv(x)
        
        # Combine time and condition embeddings
        emb = t_emb + c_emb
        emb = self.emb_layer(emb)
        emb = emb.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t_emb, c_emb):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        
        # Combine time and condition embeddings
        emb = t_emb + c_emb
        emb = self.emb_layer(emb)
        emb = emb.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        return x + emb

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(1, 2).view(-1, self.channels, size[0], size[1])

# Conditional U-Net architecture for the diffusion model
class UNetConditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, embed_dim=384, n_channels=32):
        super().__init__()
        self.time_dim = time_dim
        self.embed_dim = embed_dim
        
        # Initial convolution to process the image
        self.inc = DoubleConv(c_in, n_channels)
        
        # Downsampling path
        self.down1 = Down(n_channels, n_channels*2, time_dim)
        self.sa1 = SelfAttention(n_channels*2)
        self.down2 = Down(n_channels*2, n_channels*4, time_dim)
        self.sa2 = SelfAttention(n_channels*4)
        self.down3 = Down(n_channels*4, n_channels*8, time_dim)
        self.sa3 = SelfAttention(n_channels*8)
        
        # Bottleneck
        self.bot1 = DoubleConv(n_channels*8, n_channels*8)
        self.bot2 = DoubleConv(n_channels*8, n_channels*8)
        self.bot3 = DoubleConv(n_channels*8, n_channels*8)
        
        # Upsampling path
        self.up1 = Up(n_channels*16, n_channels*4, time_dim)
        self.sa4 = SelfAttention(n_channels*4)
        self.up2 = Up(n_channels*8, n_channels*2, time_dim)
        self.sa5 = SelfAttention(n_channels*2)
        self.up3 = Up(n_channels*4, n_channels, time_dim)
        self.sa6 = SelfAttention(n_channels)
        
        # Final convolution
        self.outc = nn.Conv2d(n_channels, c_out, kernel_size=1)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Condition embedding (from DINO embeddings)
        self.condition_embedding = nn.Sequential(
            nn.Linear(embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def pos_encoding(self, t, channels):
        """Position encoding for time steps"""
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, c):
        """
        Forward pass
        
        Args:
            x: Input image [B, C, H, W]
            t: Time steps [B, 1]
            c: Condition embedding [B, embed_dim]
        """
        # Time embedding
        t_emb = self.pos_encoding(t, self.time_dim)
        t_emb = self.time_embedding(t_emb)
        
        # Condition embedding
        c_emb = self.condition_embedding(c)
        
        # U-Net forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb, c_emb)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t_emb, c_emb)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t_emb, c_emb)
        x4 = self.sa3(x4)
        
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t_emb, c_emb)
        x = self.sa4(x)
        x = self.up2(x, x2, t_emb, c_emb)
        x = self.sa5(x)
        x = self.up3(x, x1, t_emb, c_emb)
        x = self.sa6(x)
        
        return self.outc(x)

class DiffusionModel:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=128, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        # Noise schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """Linear beta schedule"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """Add noise to images at time step t"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        ε = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * ε, ε

    def sample_timesteps(self, n):
        """Sample time steps uniformly"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)

    @torch.no_grad()
    def sample(self, model, n, c, cfg_scale=3.0):
        """
        Sample n images conditioned on embeddings c
        
        Args:
            model: UNet model
            n: Number of images to sample
            c: Condition embeddings [n, embed_dim]
            cfg_scale: Classifier-free guidance scale
        """
        model.eval()
        
        # Start from random noise
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        
        # Create unconditioned embedding for classifier-free guidance
        uncond_embed = torch.zeros_like(c).to(self.device)
        
        # Iterative denoising
        for i in tqdm(reversed(range(1, self.noise_steps)), desc="Sampling", position=0):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            
            # Predict noise with conditioning
            predicted_noise_cond = model(x, t, c)
            
            # Predict noise without conditioning (for classifier-free guidance)
            predicted_noise_uncond = model(x, t, uncond_embed)
            
            # Apply classifier-free guidance
            predicted_noise = predicted_noise_uncond + cfg_scale * (predicted_noise_cond - predicted_noise_uncond)
            
            # Get alpha and beta for current time step
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            
            # Only add noise if we're not at the last step
            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            
            # Update x using the formula from DDPM paper
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        
        model.train()
        
        # Clip and normalize to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        return x

class DiffusionTrainer:
    def __init__(self, data_dir, embedding_model_path, output_dir, batch_size=16, lr=1e-4, epochs=100):
        self.data_dir = data_dir
        self.embedding_model_path = embedding_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize embedding model
        self.embedding_model = self._load_embedding_model()
        
        # Initialize diffusion model
        self.diffusion = DiffusionModel(device=self.device)
        
        # Get embedding dimension
        with torch.no_grad():
            random_input = torch.randn(1, 3, 224, 224).to(self.device)
            embed_dim = self.embedding_model(random_input).shape[1]
        
        # Initialize U-Net model
        self.model = UNetConditional(embed_dim=embed_dim).to(self.device)
        
        # Define datasets and dataloaders
        train_transform, val_transform = get_transforms()
        
        # Create dataset with computed embeddings
        embedding_cache_path = os.path.join(self.output_dir, "embeddings_cache.pt")
        
        # Setup train dataset
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        
        # Extract and save embeddings if not already cached
        if not os.path.exists(embedding_cache_path):
            print("Computing and caching embeddings...")
            self._precompute_embeddings(train_dir, val_dir, embedding_cache_path)
        
        self.train_dataset = EmbeddingFaceDataset(
            train_dir, 
            embedding_path=os.path.join(self.output_dir, "train_embeddings.pt"),
            transform=train_transform
        )
        
        self.val_dataset = EmbeddingFaceDataset(
            val_dir, 
            embedding_path=os.path.join(self.output_dir, "val_embeddings.pt"),
            transform=val_transform
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Define optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Set up logging
        self.train_log = []
        self.val_log = []

    def _load_embedding_model(self):
        """Load the fine-tuned DINOv2 model"""
        # Initialize base model
        model = vits16_dinov2().to(self.device)
        
        # Load fine-tuned weights
        checkpoint = torch.load(self.embedding_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode and freeze parameters
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        return model

    def _precompute_embeddings(self, train_dir, val_dir, cache_path):
        """Precompute and cache embeddings for all images"""
        # Define basic transform for embedding extraction
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Process training images
        train_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir) 
        if f.endswith(('.png', '.jpg', '.jpeg'))]
                train_embeddings = []
                
                print(f"Computing embeddings for {len(train_paths)} training images...")
                
                for path in tqdm(train_paths):
                    img = Image.open(path).convert('RGB')
                    img = transform(img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.embedding_model(img).squeeze(0).cpu()
                        train_embeddings.append(embedding)
                
                train_embeddings = torch.stack(train_embeddings)
                torch.save(train_embeddings, os.path.join(self.output_dir, "train_embeddings.pt"))
                
                # Process validation images
                val_paths = [os.path.join(val_dir, f) for f in os.listdir(val_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
                val_embeddings = []
                
                print(f"Computing embeddings for {len(val_paths)} validation images...")
                
                for path in tqdm(val_paths):
                    img = Image.open(path).convert('RGB')
                    img = transform(img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.embedding_model(img).squeeze(0).cpu()
                        val_embeddings.append(embedding)
                
                val_embeddings = torch.stack(val_embeddings)
                torch.save(val_embeddings, os.path.join(self.output_dir, "val_embeddings.pt"))
                
                print("Embeddings computation completed and cached.")

            def train_epoch(self, epoch):
                self.model.train()
                epoch_loss = 0.0
                
                pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
                
                for step, (images, embeddings) in enumerate(pbar):
                    images = images.to(self.device)
                    embeddings = embeddings.to(self.device)
                    
                    # Sample random time steps
                    t = self.diffusion.sample_timesteps(images.shape[0])
                    
                    # Add noise to images according to the noise schedule
                    x_t, noise = self.diffusion.noise_images(images, t)
                    
                    # Predict the noise
                    predicted_noise = self.model(x_t, t, embeddings)
                    
                    # Calculate loss
                    loss = F.mse_loss(predicted_noise, noise)
                    
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                    # Generate samples every 500 steps
                    if step % 500 == 0:
                        self._generate_samples(epoch, step)
                
                # Return average loss for the epoch
                return epoch_loss / len(self.train_loader)

            def validate(self, epoch):
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for images, embeddings in tqdm(self.val_loader, desc='Validating'):
                        images = images.to(self.device)
                        embeddings = embeddings.to(self.device)
                        
                        t = self.diffusion.sample_timesteps(images.shape[0])
                        x_t, noise = self.diffusion.noise_images(images, t)
                        predicted_noise = self.model(x_t, t, embeddings)
                        
                        loss = F.mse_loss(predicted_noise, noise)
                        val_loss += loss.item()
                
                # Generate validation samples
                self._generate_samples(epoch, is_validation=True)
                
                return val_loss / len(self.val_loader)

            def _generate_samples(self, epoch, step=None, is_validation=False):
                # Generate samples from the model
                self.model.eval()
                
                # Get a batch of validation embeddings
                if is_validation:
                    batch = next(iter(self.val_loader))
                else:
                    batch = next(iter(self.train_loader))
                
                images, embeddings = batch
                images = images.to(self.device)
                embeddings = embeddings.to(self.device)
                
                # Sample only for the first 8 images to save time
                n_samples = min(8, images.shape[0])
                
                # Generate samples
                samples = self.diffusion.sample(
                    self.model, 
                    n=n_samples, 
                    c=embeddings[:n_samples]
                )
                
                # Make a grid of images
                grid = make_grid(samples, nrow=4, normalize=True)
                
                # Save generated images
                if is_validation:
                    save_path = self.output_dir / f"val_samples_epoch_{epoch+1}.png"
                else:
                    save_path = self.output_dir / f"samples_epoch_{epoch+1}_step_{step}.png"
                
                save_image(grid, save_path)
                
                # Also save the original images for comparison
                original_grid = make_grid(images[:n_samples], nrow=4, normalize=True)
                
                if is_validation:
                    orig_path = self.output_dir / f"val_original_epoch_{epoch+1}.png"
                else:
                    orig_path = self.output_dir / f"original_epoch_{epoch+1}_step_{step}.png"
                
                save_image(original_grid, orig_path)
                
                self.model.train()

            def train(self):
                print(f"Training on {self.device}")
                print(f"Training set size: {len(self.train_dataset)}")
                print(f"Validation set size: {len(self.val_dataset)}")
                
                best_val_loss = float('inf')
                start_time = time.time()
                
                for epoch in range(self.epochs):
                    train_loss = self.train_epoch(epoch)
                    val_loss = self.validate(epoch)
                    
                    self.train_log.append(train_loss)
                    self.val_log.append(val_loss)
                    
                    self.scheduler.step()
                    
                    print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                    
                    # Save model if validation loss improved
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': val_loss,
                        }, self.output_dir / 'best_model.pth')
                        print(f'Model saved at epoch {epoch+1}')
                    
                    # Save a checkpoint every 10 epochs
                    if (epoch + 1) % 10 == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': val_loss,
                        }, self.output_dir / f'checkpoint_epoch_{epoch+1}.pth')
                    
                    # Plot and save loss curves
                    self._plot_loss_curves()
                    
                    # Check if we're approaching time limit (5.5 hours)
                    elapsed_time = (time.time() - start_time) / 3600  # Convert to hours
                    if elapsed_time > 5.5:
                        print(f"Training stopped after {epoch+1} epochs due to time constraint (5.5 hours)")
                        break
                
                # Save final model
                torch.save({
                    'epoch': self.epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.output_dir / 'final_model.pth')
                
                print(f'Training completed in {(time.time() - start_time) / 3600:.2f} hours')
                
                # Final plot of loss curves
                self._plot_loss_curves()

            def _plot_loss_curves(self):
                """Plot and save loss curves"""
                plt.figure(figsize=(10, 5))
                plt.plot(self.train_log, label='Training Loss')
                plt.plot(self.val_log, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.savefig(self.output_dir / 'loss_curves.png')
                plt.close()

            @torch.no_grad()
            def evaluate_zero_shot(self, test_images_dir, test_output_dir):
                """Evaluate the model on unseen test images"""
                self.model.eval()
                test_output_dir = Path(test_output_dir)
                test_output_dir.mkdir(exist_ok=True, parents=True)
                
                # Define transform for test images
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # Get test images
                test_paths = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                # Limit to 50 images for evaluation
                test_paths = test_paths[:50]
                
                for i, path in enumerate(tqdm(test_paths, desc="Evaluating zero-shot performance")):
                    # Load image and compute embedding
                    img = Image.open(path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.embedding_model(img_tensor).squeeze(0)
                    
                    # Generate face from embedding
                    samples = self.diffusion.sample(
                        self.model, 
                        n=1, 
                        c=embedding.unsqueeze(0)
                    )
                    
                    # Save generated image
                    save_image(samples[0], test_output_dir / f"generated_{i}.png")
                    
                    # Save original for comparison
                    orig_img = transforms.Resize((128, 128))(img)
                    orig_img.save(test_output_dir / f"original_{i}.png")
                
                print(f"Evaluation completed. Results saved to {test_output_dir}")

        if __name__ == "__main__":
            parser = argparse.ArgumentParser(description='Train a conditional diffusion model for face generation')
            parser.add_argument('--data_dir', type=str, required=True, help='Directory with face images')
            parser.add_argument('--embedding_model_path', type=str, required=True, help='Path to fine-tuned DINO model')
            parser.add_argument('--output_dir', type=str, default='./diffusion_model', help='Output directory')
            parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
            parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
            parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
            parser.add_argument('--test', action='store_true', help='Evaluate on test set')
            parser.add_argument('--test_dir', type=str, help='Directory with test images')
            parser.add_argument('--test_output_dir', type=str, default='./test_results', help='Directory for test results')
            
            args = parser.parse_args()
            
            trainer = DiffusionTrainer(
                data_dir=args.data_dir,
                embedding_model_path=args.embedding_model_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs
            )
            
            if args.test:
                # Load best model
                checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.evaluate_zero_shot(args.test_dir, args.test_output_dir)
            else:
                trainer.train()