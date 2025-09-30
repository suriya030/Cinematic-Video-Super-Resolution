"""
Saving format has been changed => update the v2_frame_extractor.py
"""

import json
import os
import time
import shutil   
import subprocess
import glymur
import tifffile as tiff
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style
from config import DIRECTORIES

def load_json(json_path):
    """ Part of Step 1.2
    Return: dict
        - 'video_name': str - basename of video file (including extension)
        - 'video_path': str - full path to video file  
        - 'selected_frames': list[dict] - list of frame range dictionaries, each containing:
            * 'start_frame': int - starting frame number of selected range
            * 'end_frame': int - ending frame number of selected range
    """
    with open(json_path, 'r') as f:
        analysis_data = json.load(f)
    
    video_path = analysis_data['video_information']['file_path']
    video_name = os.path.basename(video_path)
    selected_frames = []
    
    for scene in analysis_data['scene_detection']['scenes']:
        if scene['frames_selected'] == True:
            selected_frames.append({
                'scene_id': scene['scene_id'],
                'start_frame': scene['selected_frame_range']['start_frame'],
                'end_frame': scene['selected_frame_range']['end_frame']
            })
    
    analysis_data_dict = {
        'video_name': video_name,
        'video_path': video_path,
        'selected_frames': selected_frames
    }

    return analysis_data_dict

def video_to_frames(video_path, output_dir):
    """ Part of Step 2.1
    Return: None
        - Creates individual J2C frames in the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%06d.j2c")
    cmd = ['ffmpeg', '-loglevel', 'quiet', '-i', video_path, '-c:v', 'copy', output_pattern]
    subprocess.run(cmd, check=True)

def decode_j2c_frame( frame_path ,CPU_threading=True):
    """ Part of Step 2.2.1
    Return: numpy array
        - Decoded J2C frame
    """
    if CPU_threading is True:
        max_threads = os.cpu_count()
        glymur.set_option('lib.num_threads', max_threads)
    
    frame = glymur.Jp2k(frame_path).read()
    return frame

def extract_selected_frames(start_frame, end_frame, 
                                temp_dir, output_dir, CPU_threading=True):
    """ Part of Step 2.2
    Return: None
        - Extracts selected frames from .j2c & saves as TIFF files
            in output directory
    """
    frame_count = end_frame - start_frame + 1
    frame_paths = []
    for frame_num in range(start_frame, end_frame+1):
        frame_path = os.path.join(temp_dir, f"frame_{frame_num:06d}.j2c")
        frame_paths.append(frame_path)
    
    frame_num = start_frame
    with tqdm(total=frame_count, desc=f"Extracting frames {start_frame}-{end_frame}", unit="frame") as pbar:
        for frame_path in frame_paths:
            # STEP2.2.1: Decode J2C frame to numpy array
            frame_numpy = decode_j2c_frame(frame_path, CPU_threading=CPU_threading)

            # STEP2.2.2: Save frame as TIFF file
            output_path = os.path.join(output_dir, f"frame_{frame_num:06d}.tiff")
            tiff.imwrite(output_path, frame_numpy, compression='zlib') # lossless compression ( CHECK IT LATER )  
            frame_num += 1
            pbar.update(1)

def print_time_summary(step1_time, video_conversion_times, frame_extraction_times, cleanup_times, total_time):
    """Print a formatted time summary of all processing steps"""
    print(f"\n{'='*60}")
    print(f"TIME SUMMARY")
    print(f"{'='*60}")
    print(f"Step 1 (File Discovery): {step1_time:.2f}s")
    if video_conversion_times:
        print(f"Video Conversion Total: {sum(video_conversion_times):.2f}s (avg: {sum(video_conversion_times)/len(video_conversion_times):.2f}s)")
        print(f"Frame Extraction Total: {sum(frame_extraction_times):.2f}s (avg: {sum(frame_extraction_times)/len(frame_extraction_times):.2f}s)")
        print(f"Cleanup Total: {sum(cleanup_times):.2f}s (avg: {sum(cleanup_times)/len(cleanup_times):.2f}s)")
    print(f"TOTAL TIME: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Start total timing
    total_start_time = time.time()
    
    # Get folder paths from config
    analysis_results_dir = DIRECTORIES['analysis_results_dir']
    extracted_frames_dir = DIRECTORIES['extracted_frames_dir']
    temp_dir = DIRECTORIES['tmp_dir_for_j2c']

    # STEP1: Read unexplored JSON files in analysis_results directory
    step1_start_time = time.time()
    unexplored_json_files = []
    for json_filename in os.listdir(analysis_results_dir):
        # STEP1.1: Checking if the JSON file is already explored
        video_name = json_filename.replace('_analysis.json', '')
        expected_output_dir = os.path.join(extracted_frames_dir,
                                             video_name)
        
        if os.path.exists(expected_output_dir):
            print(' SKIPPING ', video_name)
            continue
        
        # STEP1.2: Exploring unexplored JSON files
        json_path = os.path.join(analysis_results_dir, json_filename)
        analysis_data_dict = load_json(json_path)
        unexplored_json_files.append(analysis_data_dict)
    
    step1_time = time.time() - step1_start_time
    print(f"STEP1 completed in {step1_time:.2f} seconds")
    
    # Time tracking for each step
    video_conversion_times = []
    frame_extraction_times = []
    cleanup_times = []
        
    # STEP2: Extract selected frames from unexplored JSON files
    for analysis_data_dict in unexplored_json_files:
        video_name = analysis_data_dict['video_name'].split('.')[0]
        video_path = analysis_data_dict['video_path']
        selected_frames_dict = analysis_data_dict['selected_frames']

        print(" Processing : ", video_name)
        
        # STEP2.1: Video to frames ( .j2c )
        conversion_start_time = time.time()
        video_to_frames(video_path ,output_dir= temp_dir)
        conversion_time = time.time() - conversion_start_time
        video_conversion_times.append(conversion_time)
        print(f"Video conversion completed in {conversion_time:.2f} seconds")

        # STEP2.2: Extract selected frames from .j2c & save as TIFF files
        extraction_start_time = time.time()
        for frame_data in selected_frames_dict:
            scene_id = frame_data['scene_id']
            start_frame = frame_data['start_frame']
            end_frame = frame_data['end_frame']
            output_dir = os.path.join(extracted_frames_dir, video_name,
                            f"scene_{scene_id:06d}")
            os.makedirs(output_dir, exist_ok=True)
            extract_selected_frames(start_frame, end_frame, 
                            temp_dir, output_dir, CPU_threading=True)
        extraction_time = time.time() - extraction_start_time
        frame_extraction_times.append(extraction_time)
        print(f"Frame extraction completed in {extraction_time:.2f} seconds")
        
        # Delete all files in temporary directory
        cleanup_start_time = time.time()
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            os.remove(file_path)
        cleanup_time = time.time() - cleanup_start_time
        cleanup_times.append(cleanup_time)
        print(f"Cleanup completed in {cleanup_time:.2f} seconds")
    
    # Calculate total time and print summary
    total_time = time.time() - total_start_time
    print_time_summary(step1_time, video_conversion_times, 
                frame_extraction_times, cleanup_times, total_time)