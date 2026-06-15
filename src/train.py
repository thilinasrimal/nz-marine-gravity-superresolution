import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

class GravityDataset(Dataset):
    """PyTorch Dataset for gravity super-resolution"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 transform=None, augment: bool = True):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        
        if self.augment and np.random.random() > 0.5:
            # Random horizontal flip
            if np.random.random() > 0.5:
                x = torch.flip(x, dims=[-1])
                y = torch.flip(y, dims=[-1])
            # Random vertical flip
            if np.random.random() > 0.5:
                x = torch.flip(x, dims=[-2])
                y = torch.flip(y, dims=[-2])
            # Random rotation
            k = np.random.randint(0, 4)
            x = torch.rot90(x, k, dims=[-2, -1])
            y = torch.rot90(y, k, dims=[-2, -1])
        
        return x, y


class GravityTrainer:
    """Trainer for gravity super-resolution models"""
    
    def __init__(self, model, config: dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['scheduler']['T_max']
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, n_epochs: int):
        early_stopping_patience = self.config['training']['early_stopping']['patience']
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Early stopping and model checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print(f"  New best model saved! (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, Path('outputs/models') / filename)
    
    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(Path('outputs/models') / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']


def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load and prepare data
    from data_loader import NZGravityDataLoader
    loader = NZGravityDataLoader(config)
    features, targets = loader.prepare_training_data()
    patches = loader.create_patches(features, targets)
    
    # Convert patches to numpy arrays
    X = np.array([p[0] for p in patches])
    y = np.array([p[1] for p in patches])
    
    # Train/val/test split
    n_samples = len(X)
    n_train = int(n_samples * config['data']['train_val_test_split']['train'])
    n_val = int(n_samples * config['data']['train_val_test_split']['val'])
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets and loaders
    train_dataset = GravityDataset(X_train, y_train, augment=True)
    val_dataset = GravityDataset(X_val, y_val, augment=False)
    test_dataset = GravityDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Create model
    from models.cnn_sr import CNN_SuperResolution
    model = CNN_SuperResolution(
        n_input_channels=4,
        n_output_channels=1,
        n_features=config['model']['cnn_params']['n_features'],
        n_residual_blocks=config['model']['cnn_params']['n_residual_blocks'],
        upscale_factor=config['model']['cnn_params']['upscale_factor']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer = GravityTrainer(model, config, device)
    train_losses, val_losses = trainer.train(train_loader, val_loader, config['training']['n_epochs'])
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('outputs/figures/training_curves.png', dpi=150)
    plt.show()
    
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")


if __name__ == "__main__":
    main()