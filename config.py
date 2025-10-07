# Configuration settings for MXF video analysis

# Video processing settings
VIDEO_CONVERSION = {
    'reduce_factor': 3
}

# Scene detection settings
SCENE_DETECTION = {
    'save_detected_scenes': False,
    'adaptive_threshold': 3.5
}

# Quality analysis settings
QUALITY_ANALYSIS = {
    'sequence_length': 15,
    'quality_metrics': ['niqe'], # 'musiq' , 'niqe
    'threshold': {
        'niqe': 6.0 # musiq: 35.0
    },
    'min_frame_variance': 10.0
}

# Directory settings
DIRECTORIES = {
    'analysis_results_dir': 'analysis_results',
    'extracted_frames_dir': 'extracted_frames',
    'tmp_dir_for_j2c': 'tmp/tmp_dir_for_j2c'
}

# Device settings
DEVICE = 'auto'  # 'auto', 'cuda', or 'cpu'