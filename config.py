# Configuration settings for MXF video analysis

# Video processing settings
VIDEO_CONVERSION = {
    'target_width': 1280,
    'target_height': 720,
    'format': 'yuv420p'
}

# Scene detection settings
SCENE_DETECTION = {
    'adaptive_threshold': 3.0
}

# Quality analysis settings
QUALITY_ANALYSIS = {
    'sequence_length': 15,
    'use_musiq': False,  # Set to False to use NIQE-only
    'musiq_threshold': 35.0,
    'niqe_threshold': 6.0,
    'min_frame_variance': 10.0
}

# Output settings
OUTPUT = {
    'json_indent': 2
}

# Directory settings
DIRECTORIES = {
    'analysis_results_dir': 'analysis_results',
    'extracted_frames_dir': 'extracted_frames',
    'tmp_dir_for_j2c': 'tmp/tmp_dir_for_j2c'
}

# Device settings
DEVICE = 'auto'  # 'auto', 'cuda', or 'cpu'