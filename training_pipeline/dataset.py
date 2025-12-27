"""
Dataset loader for video super-resolution
Folder structure: data_type/movie/scene/frames
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset
import glob
import random

class VideoSRDataset(Dataset):
    def __init__(self, root_dir, num_frames=7, patch_size=64, scale=2, 
                                    is_training=True, frame_extension='tif', interpolation_mode='bicubic'):
        """
        Checked: Initialize the dataset
        Args:
            root_dir: Path to training_data/validation_data/testing_data
            num_frames: Number of consecutive frames to load
            patch_size: Size of GT patches (LQ will be patch_size//scale)
            scale: Super-resolution scale factor
            is_training: Whether this is training dataset
            frame_extension: File extension for frames (e.g., 'png', 'jpg', 'jpeg')
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.scale = scale
        self.is_training = is_training
        self.frame_extension = frame_extension
        self.interpolation_mode = interpolation_mode
        # Collect all scene directories
        self.scene_dirs = []
        self._scan_directory()
        
        print(f"Found {len(self.scene_dirs)} scenes in {root_dir}")
    
    def _scan_directory(self):
        """Checked: Scan directory structure: movie/scene"""
        for movie_dir in sorted(glob.glob(os.path.join(self.root_dir, "*"))):
                for scene_dir in sorted(glob.glob(os.path.join(movie_dir, "*"))):
                    # Check if scene has enough frames
                    frames = sorted(glob.glob(os.path.join(scene_dir, f"*.{self.frame_extension}")))
                    
                    if len(frames) >= self.num_frames:
                        self.scene_dirs.append(scene_dir)
    
    def __len__(self): 
        """Checked: Return the number of scenes"""
        return len(self.scene_dirs)
    
    def _load_gt_frames(self, scene_dir, start_idx=0):
        """Checked: Load consecutive frames from a scene
        Args:
            scene_dir: Path to scene directory
            start_idx: Index of first frame to load
        
        Returns:
            frame_list: List of frames/numpy arrays of shape (1716, 4096, 3)
        """
        frame_path = sorted(glob.glob(os.path.join(scene_dir, f"*.{self.frame_extension}")))
        
        frame_list = []
        for i in range(start_idx, start_idx + self.num_frames):
            img = tiff.imread(frame_path[i]) 
            # print( type(img) ) # <class 'numpy.ndarray'>
            img = img.astype(np.float32) / 4095.0 # 2^12 - 1 = 4095
            frame_list.append(img)
        
        return frame_list 
    
    def _get_lq(self, frames_gt):
        """Checked: Generate LQ frames from full GT frames
        Args:
            frames_gt: List of GT frames/numpy arrays of shape (1716, 4096, 3)
        Returns:
            frames_lq: List of LQ frames/numpy arrays of shape (1716, 2048, 3)
        """

        # Downsample GT to create LQ using scale factor
        frames_lq = []
        for img_gt in frames_gt:
            img_tensor = torch.from_numpy(img_gt).permute(2, 0, 1).unsqueeze(0).float()
            img_lq = F.interpolate(img_tensor, scale_factor=1.0 / self.scale, 
                                    mode= self.interpolation_mode, align_corners=False)
            img_lq = img_lq.squeeze(0).permute(1, 2, 0).numpy()
            frames_lq.append(img_lq)

        return frames_lq
    
    def __getitem__(self, idx):
        """Checked: Get item from dataset
        Args:
            idx: Index of scene
        Returns:
            dict: Dictionary containing 'lq' and 'gt' tensors [T, C, H, W]
        """
        # Get scene directory
        scene_dir = self.scene_dirs[idx]
        
        # Determine starting frame
        frames = sorted(glob.glob(os.path.join(scene_dir, f"*.{self.frame_extension}")))
        
        # Load frames
        frames_gt = self._load_gt_frames(scene_dir)
        
        # Get patches and generate LQ
        frames_lq = self._get_lq(frames_gt)
        
        # Convert to tensors and stack
        lq_tensor = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() for f in frames_lq], dim=0)  # [T, C, H, W]
        gt_tensor = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() for f in frames_gt], dim=0) # [T, C, H, W]
        
        return {
            'lq': lq_tensor,  # Low resolution input [T, C, H, W]
            'gt': gt_tensor,  # High resolution target [T, C, H, W]
        }

def create_sr_dataloaders(config):
    """Checked: Create training and validation dataloaders for super-resolution"""
    train_dataset = VideoSRDataset(
        root_dir=config.train_data_path,
        num_frames=config.num_frames,
        patch_size=config.patch_size,
        scale=config.scale,
        is_training=True,
        frame_extension=config.frame_extension,
        interpolation_mode=config.interpolation_mode
    )
    
    val_dataset = VideoSRDataset(
        root_dir=config.val_data_path,
        num_frames=config.num_frames,
        patch_size=config.patch_size,
        scale=config.scale,
        is_training=False,
        frame_extension=config.frame_extension,
        interpolation_mode=config.interpolation_mode
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader