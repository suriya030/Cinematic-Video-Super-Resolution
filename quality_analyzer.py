"""
1. Batch IQA processing is possibe. ( try that later)
2. ISSUES: Not sure what is the input format( RBG or not etc ) for the IQA models.
"""

import cv2
import torch
import pyiqa
import time
import numpy as np
import os
import utils
from collections import deque
from tqdm import tqdm
from colorama import Fore
from config import QUALITY_ANALYSIS, DEVICE

def initialize_quality_metrics(device='auto'):
    """Initialize Image Quality Assessment (IQA) models
    args: device [str] 
    return: quality_metrics [dict of pyiqa.Metric],device [torch.device],
            execution_time [float]"""

    device = utils.get_device(device)
    quality_metrics = QUALITY_ANALYSIS['quality_metrics'] 
    quality_metric_models = {}
    for metric in quality_metrics:
        quality_metric_models[metric] = pyiqa.create_metric(metric, device=device)

    utils.print_success("Quality metrics models loaded successfully")
    return quality_metric_models, device

def find_sequence_per_scene(frames_ndarray, detected_scenes, base_video_name):
    """Find high-quality frame sequences in each detected scene using only NIQE threshold
    args: frames_ndarray [list of ndarrays], detected_scenes [list of dicts], base_video_name [str]
    return: scene_results [list of dicts] 
    {scene_id [int], sequence_found [bool], selected_frames [list of int]}, 
    and execution_time [float]"""
    start_time = time.time()

    # Step 0 : Initalize IQA models & get parameters
    sequence_length = QUALITY_ANALYSIS['sequence_length']
    min_frame_variance = QUALITY_ANALYSIS['min_frame_variance']
    quality_metrics = QUALITY_ANALYSIS['quality_metrics'] # ex: ['niqe', 'musiq']
    quality_metric_thresholds = QUALITY_ANALYSIS['threshold'] # ex: {'niqe': 6.0, 'musiq': 35.0}
    quality_metric_models, device = initialize_quality_metrics(DEVICE)

    # Step 1 : print starting message
    utils.print_processing("Starting quality analysis per scene...")
    for metric in quality_metrics:
        symbol = '<' if quality_metric_models[metric].lower_better else '>'
        print(f"{Fore.CYAN}   Threshold: {Fore.GREEN}{metric} {symbol} {QUALITY_ANALYSIS['threshold'][metric]}")
    print(f"{Fore.CYAN}   Target sequence length: {Fore.GREEN}{QUALITY_ANALYSIS['sequence_length']} frames ")
    
    # Step 2 : Processing each scene individually
    scene_results = []
    for scene in tqdm(detected_scenes, desc="Processing scenes", unit="scene", colour="magenta"):
        scene_id = scene['scene_id']
        start_frame = scene['start_frame']
        end_frame = scene['end_frame']
        
        frame_buffer = deque(maxlen=sequence_length)
        selected_frames = []
        # Step 2.1 : Process frames within current scene
        for frame_idx in range(start_frame - 1, end_frame):
            frame_ndarray = frames_ndarray[frame_idx]
            frame_number = frame_idx + 1
            # Step 2.2 : Quality Check 1: Skip low-variance frames (likely blank/black)
            if np.std(frame_ndarray) < min_frame_variance:
                frame_buffer.clear()
                continue
            # Step 2.3 : Quality Check 2: Calculate IQA-based quality scores
            flag = True
            frame_tensor = torch.tensor(frame_ndarray).permute(2, 0, 1).unsqueeze(0) / 255.0 # (H, W, 3) -> (1, 3, H, W)
            frame_tensor = frame_tensor.to(device)
            for metric in quality_metrics:
                quality_metric_model = quality_metric_models[metric]
                quality_score = quality_metric_model(frame_tensor).item()
                lower_better = quality_metric_model.lower_better
                if ((not lower_better and quality_score < quality_metric_thresholds[metric]) or 
                    (lower_better and quality_score > quality_metric_thresholds[metric])):
                    flag = False
                    break
            if flag:
                frame_buffer.append(frame_number)
            else:
                frame_buffer.clear()
            # Step 2.4 : Check if we found a complete sequence
            if len(frame_buffer) == sequence_length:
                selected_frames = list(frame_buffer)
                break
            
        # Step 2.5 : Store results for current scene
        success = len(selected_frames) == sequence_length
        scene_result = {
            'scene_id': scene_id,
            'sequence_found': success,
            'selected_frames': selected_frames }        
        # if success:
        #     utils.print_success(f"Scene {scene_id}: Found {sequence_length} quality frames")
        # else:
        #     utils.print_warning(f"Scene {scene_id}: Found no quality frames")
        scene_results.append(scene_result)
    
    # Step 3 : Return results & final print statements
    execution_time = time.time() - start_time
    print(f"{Fore.WHITE}⏱️  Total execution time: {execution_time:.2f} seconds")
    return scene_results, execution_time