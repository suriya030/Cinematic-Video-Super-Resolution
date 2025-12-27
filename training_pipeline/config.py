"""
Configuration file for VRT Video Super-Resolution training
Version 1: SR-specific configuration
"""

import torch

class SRConfig:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Super-Resolution specific
    scale = 2  # 2x super-resolution
    
    # Data paths
    train_data_path = 'training_data'
    val_data_path = 'validation_data'   
    test_data_path = 'testing_data'
    
    # Data parameters
    num_frames = 7  # Number of input frames (should be odd for center frame)
    patch_size = 256  # GT patch size (LQ will be 64x64 for 4x SR)
    frame_extension = 'tif'  # File extension for frames (png, jpg, jpeg, etc.)
    interpolation_mode = 'bicubic'  # Interpolation mode for downsampling
    
    # VRT architecture parameters for SR
    window_size = [6, 8, 8]
    depths = [8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4]
    embed_dims = [120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180]
    num_heads = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    pa_frames = 2  # Parallel warping frames for alignment
    deformable_groups = 12
    
    # Memory optimization
    use_checkpoint_attn = False  # Set True to save memory during training
    use_checkpoint_ffn = False
    
    # Training parameters
    batch_size = 1  # Small batch due to video memory requirements
    accumulation_steps = 8  # Gradient accumulation to simulate larger batch
    num_epochs = 300
    learning_rate = 2e-4
    min_lr = 1e-7
    weight_decay = 0.01
    
    # Training settings
    val_interval = 5  # Validate every N epochs
    save_interval = 10  # Save checkpoint every N epochs
    
    # Directories
    checkpoint_dir = 'checkpoints_sr'
    result_dir = 'results_sr'
    
    # Resume training
    resume = False
    resume_path = 'checkpoints_sr/latest.pth'
    
    # Testing
    test_patch_size = 256  # Can be larger for testing
    test_overlap = 32  # Overlap for sliding window testing