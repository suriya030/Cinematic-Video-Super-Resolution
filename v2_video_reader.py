import subprocess
import cv2
import os
import json
import time
import numpy as np
import tifffile
import shutil
from pymediainfo import MediaInfo
from tqdm import tqdm
from colorama import Fore, Style
from config import VIDEO_CONVERSION
from utils import print_processing, print_success

def get_metadata(file_path):
    """ args: video file path [str]
    return: video metadata [dict]
    """
    media_info = MediaInfo.parse(file_path)
    
    for track in media_info.tracks:
        if track.track_type == "Video":
            frame_rate = track.frame_rate
            duration = track.duration
            total_frames = track.frame_count 
            codec = track.format 
            aspect_ratio = track.display_aspect_ratio 
            bit_depth = track.bit_depth 
            color_space = track.color_space 
            chroma_subsampling = track.chroma_subsampling 
            
            return {
                'frame_rate': frame_rate,
                'total_frames': total_frames,
                'width': track.width,
                'height': track.height,
                'codec': codec,
                'aspect_ratio': aspect_ratio,
                'bit_depth': bit_depth,
                'color_space': color_space,
                'chroma_subsampling': chroma_subsampling
            }
    
    raise ValueError("No video track found in the file")

def read_mxf_video(file_path, reduce_factor=0):
    """args: video file path [str]
    return: frames ndarray [list], metadata [dict], execution time [float]
    """
    start_time = time.time()
    print_processing("Opening MXF video file...")
    
    # Step 0: Read metadata
    metadata = get_metadata(file_path)    
    
    # Step 1: Extract .mxf to .j2c frames
    tmp_dir = "tmp"
    tmp_j2c_frames_dir = os.path.join(tmp_dir, "j2c_frames")
    os.makedirs(tmp_j2c_frames_dir, exist_ok=True)
    output_pattern = os.path.join(tmp_j2c_frames_dir, "frame_%06d.j2c")
    cmd = f"ffmpeg -i \"{file_path}\" -c:v copy \"{output_pattern}\" -y"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    # Step 2: .j2c frames to .tif frames
    tif_output_dir = os.path.join(tmp_dir, "tif_frames")
    os.makedirs(tif_output_dir, exist_ok=True)
    reduce_factor = VIDEO_CONVERSION['reduce_factor']  # Fixed reduce factor    
    cmd = f"grk_decompress -y \"{tmp_j2c_frames_dir}\" -a \"{tif_output_dir}\" -O tif -r {reduce_factor} --force-rgb -H {os.cpu_count()}"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    # Step 3: Read .tif frames and convert to ndarray
    frames_ndarray = []
    tif_files = sorted([f for f in os.listdir(tif_output_dir) if f.endswith('.tif')])
    for filename in tif_files:
        file_path_tif = os.path.join(tif_output_dir, filename)
        frame = tifffile.imread(file_path_tif) # ndarray uint16 (H, W, 3)
        frame_8bit = (frame >> 4).astype(np.uint8)
        frames_ndarray.append(frame_8bit)
        
    # Step 4: Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print_success(f"Successfully processed {len(tif_files)} frames and converted to ndarray")
    print(f"{Fore.WHITE}⏱️  Total execution time: {execution_time:.2f} seconds")
    
    return frames_ndarray, metadata, execution_time
