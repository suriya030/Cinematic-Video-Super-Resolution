import os
import time
import subprocess
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector
from colorama import Fore
from config import SCENE_DETECTION
from utils import print_processing, print_success, print_warning

import subprocess
import os

import subprocess
import os

def save_detected_scenes(detected_scenes, video_path):
    """ args: detected_scenes [list of dicts], video_path [str]
    return: None """
    
    # Save to test_scenes directory
    output_dir = "tmp/test_scenes"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0] 
    print_processing(f"Extracting {len(detected_scenes)} scenes...")
    
    for scene in detected_scenes:
        output_file = os.path.join(output_dir, f"{base_name}-Scene-{scene['scene_id']:05d}.mp4")
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(scene['start_time_seconds']),
            '-to', str(scene['end_time_seconds']),
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-y',
            output_file
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"{Fore.WHITE}  ✓ Scene {scene['scene_id']}")
    
    print_success(f"Scenes saved to: {Fore.CYAN}{output_dir}")
    

def scene_detection(video_path, frame_rate):
    """ args: video file path [str], frame rate [int]
    return: detected_scenes [list of dicts] 
    {scene_id [int], start_time_seconds [float], end_time_seconds [float], 
    start_frame [int], end_frame [int], frame_count [int]} 
    and execution time [float]"""
    
    if SCENE_DETECTION['use_proxy_video']:
        video_path = video_path.replace('.mxf', '.mp4')
    print_processing("Analyzing video for scene changes...")
    st_time = time.time()
    
    # Step 1: Initialize & run scene detection
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(
        adaptive_threshold=SCENE_DETECTION['adaptive_threshold'] ))
    scene_manager.detect_scenes(video=video, show_progress=True)
    
    # Step 2: Process detected scenes
    scene_list = scene_manager.get_scene_list()
    detected_scenes = []
    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = start_time.get_frames() + 1
        end_frame = end_time.get_frames()
        detected_scenes.append({
            'scene_id': i + 1,
            'start_time_seconds': round(start_time.get_seconds(), 2),
            'end_time_seconds': round(end_time.get_seconds(), 2),
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frame_count': end_frame - start_frame + 1
        })
        
    execution_time = time.time() - st_time
    print_success(f"Found {Fore.YELLOW}{len(scene_list)}{Fore.GREEN} scenes")
    print(f"{Fore.WHITE}⏱️  Total execution time: {execution_time:.2f} seconds")

    if SCENE_DETECTION['save_detected_scenes']:
        save_detected_scenes(detected_scenes, video_path)
    
    return detected_scenes, execution_time
