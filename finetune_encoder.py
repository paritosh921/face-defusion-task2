import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import time

# Import DINOv2 model
from torchvision.models import vits16_dinov2, ViT_S16_DINOv2_Weights

class FaceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, image  # Return the same image as input and target for contrastive learning

def get_transforms():
    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features):
        batch_size = features.size(0)
        
        # Normalize features to unit norm
        features = nn.functional.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.T) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        # For each row, the ground truth is the same row (identity mapping)
        labels = torch.arange(batch_size, device=features.device)
        
        return self.criterion(sim_matrix, labels)

class DINOv2FineTuner:
    def __init__(self, data_dir, output_dir, batch_size=32, lr=1e-4, epochs=20):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Define datasets and dataloaders
        train_transform, val_transform = get_transforms()
        
        # Split dataset into train/val
        all_files = os.listdir(data_dir)
        np.random.shuffle(all_files)
        split_idx = int(len(all_files) * 0.9)  # 90% for training
        
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        
        # Create train/val directories if they don't exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        self.train_dataset = FaceDataset(train_dir, transform=train_transform)
        self.val_dataset = FaceDataset(val_dir, transform=val_transform)
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Define optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = ContrastiveLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def _initialize_model(self):
        # Load pretrained DINOv2
        model = vits16_dinov2(weights=ViT_S16_DINOv2_Weights.PRETRAINED)
        
        # Freeze most of the model layers
        for name, param in model.named_parameters():
            if 'blocks.11' not in name:  # Only fine-tune the last block
                param.requires_grad = False
        
        model = model.to(self.device)
        return model

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Get embeddings
            embeddings = self.model(images)
            
            # Calculate contrastive loss
            loss = self.criterion(embeddings)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (batch_idx + 1)})
        
        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, _ in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                
                # Get embeddings
                embeddings = self.model(images)
                
                # Calculate contrastive loss
                loss = self.criterion(embeddings)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)

    def train(self):
        print(f"Training on {self.device}")
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            
            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.output_dir / 'best_model.pth')
                print(f'Model saved at epoch {epoch+1}')
            
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

    def extract_embeddings(self, data_loader):
        """Extract embeddings for all images in the dataset"""
        self.model.eval()
        all_embeddings = []
        all_paths = []
        
        with torch.no_grad():
            for images, paths in tqdm(data_loader, desc='Extracting embeddings'):
                images = images.to(self.device)
                embeddings = self.model(images)
                all_embeddings.append(embeddings.cpu())
                all_paths.extend(paths)
        
        return torch.cat(all_embeddings), all_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune DINOv2 on face dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with face images')
    parser.add_argument('--output_dir', type=str, default='./dinov2_finetuned', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    
    args = parser.parse_args()
    
    fine_tuner = DINOv2FineTuner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs
    )
    
    fine_tuner.train()