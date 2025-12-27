"""
Utility functions for VRT Video Super-Resolution
Version 1: Essential functions for SR training and testing
"""

import torch
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_metric': best_metric
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint.get('best_metric', 0)


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images
    Args:
        img1: First image (tensor or numpy array)
        img2: Second image (tensor or numpy array)
        max_val: Maximum pixel value (1.0 for normalized, 255 for uint8)
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure values are in correct range
    img1 = np.clip(img1, 0, max_val)
    img2 = np.clip(img2, 0, max_val)
    
    # Handle batch dimension
    if img1.ndim == 5:  # (B, T, C, H, W)
        psnr_vals = []
        for b in range(img1.shape[0]):
            for t in range(img1.shape[1]):
                psnr_vals.append(psnr(img1[b, t], img2[b, t], data_range=max_val))
        return np.mean(psnr_vals)
    elif img1.ndim == 4:  # (T, C, H, W) or (B, C, H, W)
        psnr_vals = []
        for i in range(img1.shape[0]):
            psnr_vals.append(psnr(img1[i], img2[i], data_range=max_val))
        return np.mean(psnr_vals)
    else:
        return psnr(img1, img2, data_range=max_val)


def calculate_ssim(img1, img2, max_val=1.0):
    """
    Calculate SSIM between two images
    Args:
        img1: First image (tensor or numpy array)
        img2: Second image (tensor or numpy array)
        max_val: Maximum pixel value
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure values are in correct range
    img1 = np.clip(img1, 0, max_val)
    img2 = np.clip(img2, 0, max_val)
    
    # Handle different dimensions
    if img1.ndim == 5:  # (B, T, C, H, W)
        ssim_vals = []
        for b in range(img1.shape[0]):
            for t in range(img1.shape[1]):
                # Transpose to (H, W, C) for skimage
                im1 = np.transpose(img1[b, t], (1, 2, 0))
                im2 = np.transpose(img2[b, t], (1, 2, 0))
                ssim_vals.append(ssim(im1, im2, data_range=max_val, channel_axis=2))
        return np.mean(ssim_vals)
    elif img1.ndim == 4:  # (T, C, H, W) or (B, C, H, W)
        ssim_vals = []
        for i in range(img1.shape[0]):
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            ssim_vals.append(ssim(im1, im2, data_range=max_val, channel_axis=2))
        return np.mean(ssim_vals)
    elif img1.ndim == 3:  # (C, H, W)
        im1 = np.transpose(img1, (1, 2, 0))
        im2 = np.transpose(img2, (1, 2, 0))
        return ssim(im1, im2, data_range=max_val, channel_axis=2)
    else:
        return ssim(img1, img2, data_range=max_val)


def tensor_to_img(tensor, to_bgr=False):
    """
    Convert tensor to image array for visualization
    Args:
        tensor: Input tensor [C, H, W] or [B, C, H, W]
        to_bgr: Convert RGB to BGR for OpenCV
    Returns:
        Image array [H, W, C] in uint8
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first in batch
    
    # Detach and convert to numpy
    img = tensor.detach().cpu().numpy()
    
    # Handle grayscale
    if img.shape[0] == 1:
        img = img[0]  # Remove channel dimension for grayscale
    else:
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        if to_bgr and img.shape[2] == 3:
            img = img[..., ::-1]  # RGB to BGR
    
    # Convert to uint8
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img


def save_video_frames(frames, output_path, fps=30):
    """
    Save tensor frames as video file
    Args:
        frames: Tensor [T, C, H, W]
        output_path: Output video path
        fps: Frames per second
    """
    if torch.is_tensor(frames):
        frames = frames.detach().cpu()
    
    T, C, H, W = frames.shape
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    # Write frames
    for t in range(T):
        frame = tensor_to_img(frames[t], to_bgr=True)
        if frame.ndim == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")


def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers
    Used for positional encoding in some SR methods
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def bicubic_upsample(x, scale):
    """
    Bicubic upsampling for comparison
    Args:
        x: Input tensor [B, C, H, W] or [C, H, W]
        scale: Upsampling scale factor
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    B, C, H, W = x.shape
    out_h, out_w = H * scale, W * scale
    
    x_up = torch.nn.functional.interpolate(
        x, size=(out_h, out_w), mode='bicubic', align_corners=False
    )
    
    if squeeze:
        x_up = x_up.squeeze(0)
    
    return x_up


def print_network_summary(model, input_size=(1, 7, 3, 64, 64)):
    """
    Print network architecture summary
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, T, C, H, W)
    """
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_num, trainable_num
    
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    total_params, trainable_params = get_parameter_number(model)
    model_size = get_model_size(model)
    
    print(f"{'='*60}")
    print(f"Network Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    print(f"{'='*60}")