"""
Loss functions for VRT Video Super-Resolution training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1) - More robust to outliers than L2"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()


class L1Loss(nn.Module):
    """Simple L1 Loss"""
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss_fn = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.loss_fn(pred, target)


class L2Loss(nn.Module):
    """Simple L2/MSE Loss"""
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss_fn = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.loss_fn(pred, target)


class PSNRLoss(nn.Module):
    """Loss based on PSNR (Peak Signal-to-Noise Ratio)"""
    def __init__(self):
        super(PSNRLoss, self).__init__()
        
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        return -10 * torch.log10(1 / (mse + 1e-8))


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel filters"""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
    def forward(self, pred, target):
        # Move Sobel filters to the same device as input
        self.sobel_x = self.sobel_x.to(pred.device)
        self.sobel_y = self.sobel_y.to(pred.device)
        
        # Reshape if needed (B, T, C, H, W) -> (B*T, C, H, W)
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred = pred.view(B*T, C, H, W)
            target = target.view(B*T, C, H, W)
        
        # Compute edges
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)
        
        return F.l1_loss(pred_edge, target_edge)


class CombinedSRLoss(nn.Module):
    """Combined loss for super-resolution (can add more components in v2)"""
    def __init__(self, loss_type='charbonnier', edge_weight=0.0):
        super(CombinedSRLoss, self).__init__()
        
        # Primary reconstruction loss
        if loss_type == 'charbonnier':
            self.pixel_loss = CharbonnierLoss()
        elif loss_type == 'l1':
            self.pixel_loss = L1Loss()
        elif loss_type == 'l2':
            self.pixel_loss = L2Loss()
        else:
            self.pixel_loss = CharbonnierLoss()
        
        # Optional edge loss
        self.edge_weight = edge_weight
        if edge_weight > 0:
            self.edge_loss = EdgeLoss()
        
    def forward(self, pred, target):
        # Pixel-wise loss
        loss = self.pixel_loss(pred, target)
        
        # Add edge loss if enabled
        if self.edge_weight > 0:
            loss = loss + self.edge_weight * self.edge_loss(pred, target)
        
        return loss