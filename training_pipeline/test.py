"""
Testing script for VRT Video Super-Resolution
Version 1: Basic testing with sliding window for large videos
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image

from model import build_vrt_sr_model
from dataset import VideoSRDataset
from config import SRConfig
from utils import calculate_ssim, tensor_to_img, save_video_frames

def test_video_sr(model, lq_frames, config):
    """
    Process low-resolution video frames to super-resolution
    Args:
        model: VRT SR model
        lq_frames: Low-resolution frames [T, C, H, W]
        config: Configuration
    Returns:
        sr_frames: Super-resolved frames [T, C, H*scale, W*scale]
    """
    model.eval()
    
    T, C, H, W = lq_frames.shape
    
    # Process in chunks of num_frames
    num_frames = config.num_frames
    stride = num_frames // 2  # Overlap for temporal consistency
    
    sr_frames = []
    frame_count = torch.zeros(T, 1, 1, 1)
    output_frames = torch.zeros(T, C, H * config.scale, W * config.scale)
    
    # Sliding window processing
    positions = list(range(0, T - num_frames + 1, stride))
    if positions[-1] < T - num_frames:
        positions.append(T - num_frames)
    
    with torch.no_grad():
        for pos in tqdm(positions, desc='Processing frames'):
            # Extract chunk
            chunk = lq_frames[pos:pos+num_frames].unsqueeze(0)  # [1, T, C, H, W]
            chunk = chunk.to(config.device)
            
            # Super-resolve
            sr_chunk = model(chunk)
            sr_chunk = sr_chunk.squeeze(0).cpu()  # [T, C, H*s, W*s]
            
            # Accumulate with overlap handling
            for i in range(num_frames):
                if pos + i < T:
                    output_frames[pos + i] += sr_chunk[i]
                    frame_count[pos + i] += 1
    
    # Average overlapping regions
    output_frames = output_frames / frame_count
    
    return output_frames

def test_single_video_file(model, video_path, config, save_path=None):
    """Test on a single video file"""
    print(f"Processing video: {video_path}")
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB and normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        print("No frames found in video!")
        return
    
    # Stack frames
    lq_frames = torch.stack(frames[:min(len(frames), 100)], dim=0)  # Limit to 100 frames for memory
    
    # Downsample to create LQ (for testing on HR videos)
    T, C, H, W = lq_frames.shape
    lq_h, lq_w = H // config.scale, W // config.scale
    lq_frames_down = F.interpolate(
        lq_frames.view(T, C, H, W),
        size=(lq_h, lq_w),
        mode='bicubic',
        align_corners=False
    )
    
    # Super-resolve
    sr_frames = test_video_sr(model, lq_frames_down, config)
    
    # Save output
    if save_path:
        save_video_frames(sr_frames, save_path, fps)
        print(f"Saved SR video to {save_path}")
    
    # Calculate metrics (using original as GT)
    avg_ssim = calculate_ssim(sr_frames.numpy(), lq_frames.numpy())
    
    print(f"SSIM: {avg_ssim:.4f}")
    
    return sr_frames

def test_dataset(config):
    """Test on entire test dataset and calculate metrics"""
    # Create model
    model = build_vrt_sr_model(config)
    model = model.to(config.device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return
    
    model.eval()
    
    # Create test dataset
    test_dataset = VideoSRDataset(
        root_dir=config.test_data_path,
        num_frames=config.num_frames,
        patch_size=config.test_patch_size,
        scale=config.scale,
        is_training=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Metrics storage
    all_ssim = []
    
    # Output directory
    output_dir = config.result_dir
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Testing')):
            lq = batch['lq'].to(config.device)  # [1, T, C, H_LR, W_LR]
            gt = batch['gt'].to(config.device)  # [1, T, C, H_HR, W_HR]
            scene_name = batch['scene_name'][0]
            
            # Forward pass
            output = model(lq)  # [1, T, C, H_HR, W_HR]
            
            # Calculate metrics for center frame
            center_idx = output.shape[1] // 2
            output_center = output[0, center_idx].cpu().numpy()
            gt_center = gt[0, center_idx].cpu().numpy()
            
            ssim_val = calculate_ssim(output_center, gt_center)
            
            all_ssim.append(ssim_val)
            
            # Save sample outputs (first 5 scenes)
            if i < 5:
                # Save center frames as images
                output_img = tensor_to_img(output[0, center_idx])
                gt_img = tensor_to_img(gt[0, center_idx])
                lq_img = tensor_to_img(lq[0, center_idx])
                
                # Upscale LQ for comparison
                lq_img_up = cv2.resize(lq_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_CUBIC)
                
                # Create comparison image
                comparison = np.hstack([lq_img_up, output_img, gt_img])
                cv2.imwrite(
                    os.path.join(output_dir, f'{scene_name}_comparison.png'),
                    cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                )
                
                # Save video
                if output.shape[1] == config.num_frames:
                    output_path = os.path.join(output_dir, f'{scene_name}_sr.mp4')
                    save_video_frames(output[0], output_path)
    
    # Print average metrics
    avg_ssim = np.mean(all_ssim)
    std_ssim = np.std(all_ssim)
    
    print(f"\n{'='*50}")
    print(f"Test Results on {len(all_ssim)} scenes:")
    print(f"SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"{'='*50}")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Video Super-Resolution Test Results\n")
        f.write(f"Scale: {config.scale}x\n")
        f.write(f"Model: {checkpoint_path}\n")
        f.write(f"Number of test scenes: {len(all_ssim)}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
        f.write(f"{'='*50}\n\n")
        f.write("Per-scene results:\n")
        for i, s in enumerate(all_ssim):
            f.write(f"Scene {i+1}: SSIM={s:.4f}\n")
    
    print(f"Results saved to {metrics_file}")

def main():
    """Main testing function"""
    config = SRConfig()
    
    # Test on entire dataset
    test_dataset(config)
    
    # Optional: Test on a specific video file
    # model = build_vrt_sr_model(config)
    # model.load_state_dict(torch.load('checkpoints_sr/best_model.pth')['model_state_dict'])
    # model = model.to(config.device)
    # test_single_video_file(model, 'path/to/video.mp4', config, 'output_sr.mp4')

if __name__ == '__main__':
    main()