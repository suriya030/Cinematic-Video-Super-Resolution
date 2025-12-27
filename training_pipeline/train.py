"""
Training script for VRT Video Super-Resolution
Version 1: Basic SR training loop (no gradient accumulation)
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import time

from model import build_vrt_sr_model
from dataset import create_sr_dataloaders
from config import SRConfig
from losses import CharbonnierLoss
from utils import AverageMeter, save_checkpoint, load_checkpoint

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, config):
    """Train for one epoch (no gradient accumulation)"""
    
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs}')
    for i, batch in enumerate(pbar):
        # Move data to device
        lq = batch['lq'].to(config.device)  # [B, T, C, H_LR, W_LR]
        gt = batch['gt'].to(config.device)  # [B, T, C, H_HR, W_HR]
        
        # Forward pass
        output = model(lq)
        
        # Calculate loss
        loss = criterion(output, gt)
        
        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), lq.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': losses.avg,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # No TensorBoard logging
    
    return losses.avg, None

def validate(model, dataloader, criterion, epoch, config):
    """Validate the model"""
    model.eval()

    losses = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for i, batch in enumerate(pbar):
            lq = batch['lq'].to(config.device)
            gt = batch['gt'].to(config.device)
            
            # Forward pass
            output = model(lq)
            
            # Calculate loss
            loss = criterion(output, gt)
            
            losses.update(loss.item(), lq.size(0))
            pbar.set_postfix({'val_loss': losses.avg})
    
    return losses.avg, None

def main():
    """Main training function for video super-resolution"""
    # Load config
    config = SRConfig()
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # Create model
    model = build_vrt_sr_model(config)
    model = model.to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params/1e6:.2f}M parameters ({trainable_params/1e6:.2f}M trainable)")
    
    # Create dataloaders
    train_loader, val_loader = create_sr_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Loss function
    criterion = CharbonnierLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Load checkpoint if exists
    start_epoch = 0
    if config.resume and os.path.exists(config.resume_path):
        start_epoch, _ = load_checkpoint(
            model, optimizer, None, config.resume_path
        )
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, config
        )
        print(f"Training - Loss: {train_loss:.6f}")
        
        # Validate
        if (epoch + 1) % config.val_interval == 0:
            val_loss, _ = validate(
                model, val_loader, criterion, epoch, config
            )
            print(f"Validation - Loss: {val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, None, epoch, 0.0,
                os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
            # Also save as latest
            save_checkpoint(
                model, optimizer, None, epoch, 0.0,
                os.path.join(config.checkpoint_dir, 'latest.pth')
            )
    
    print(f"\nTraining completed!")

if __name__ == '__main__':
    main()