import subprocess
import cv2
import os
import json
import time
import tifffile
import shutil
from tqdm import tqdm
from colorama import Fore, Style
from config import VIDEO_CONVERSION
from utils import print_processing, print_success

def print_video_info(file_path, total_frames):
    """Display basic video file information"""
    separator = "=" * 80
    
    # Header
    print(f"\n{Fore.CYAN}{separator}")
    print(f"{Fore.CYAN}{'VIDEO INFORMATION':^80}")
    print(f"{Fore.CYAN}{separator}")
    
    # Print only filename and frame count
    print(f"{Fore.WHITE}{'File':<20}{Fore.YELLOW}{os.path.basename(file_path):<58}")
    print(f"{Fore.WHITE}{'Total Frames':<20}{Fore.GREEN}{total_frames:<58}")
    
    # Footer
    print(f"{Fore.CYAN}{separator}{Style.RESET_ALL}")

def save_video_metadata(file_path, video_info):
    """Save detailed video metadata to logs/metadata folder"""
    metadata = {
        'file_path': file_path,
        'format': video_info['format'],
        'codec': video_info['codec'],
        'resolution': video_info['original_resolution'],
        'frame_rate': video_info['frame_rate'],
        'total_frames': video_info['total_frames']
    }
    
    # Create metadata filename based on input video name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    metadata_dir = os.path.join('logs', 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, f"{base_name}_metadata.json")
    
    # Save metadata as JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def read_mxf_video(file_path):
    """Read 4K .mxf video file using grok decoder and keep frames at original resolution"""
    start_time = time.time()
    print_processing("Opening MXF video file...")
    
    # Get accurate frame info using OpenCV
    cap = cv2.VideoCapture(file_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print_video_info(file_path, total_frames)
    
    # Extract .j2c frames
    temp_dir = "tmp"
    j2c_frames_dir = os.path.join(temp_dir, "j2c_frames")
    os.makedirs(j2c_frames_dir, exist_ok=True)
    
    output_pattern = os.path.join(j2c_frames_dir, "frame_%06d.j2c")
    cmd = f"ffmpeg -i \"{file_path}\" -c:v copy \"{output_pattern}\" -y"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    # Batch decode using grok
    tif_output_dir = os.path.join(temp_dir, "tif_frames")
    os.makedirs(tif_output_dir, exist_ok=True)
    
    reduce_factor = 3  # Fixed reduce factor
    cmd = f"grk_decompress -y \"{j2c_frames_dir}\" -a \"{tif_output_dir}\" -O tif -r {reduce_factor} -H {os.cpu_count()}"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    # Read .tif frames and convert to 720p
    print_processing("Converting frames to 720p...")
    frame_count = 0
    frames_720p = []
    
    tif_files = sorted([f for f in os.listdir(tif_output_dir) if f.endswith('.tif')])
    
    # Process each frame with progress tracking
    with tqdm(total=len(tif_files), desc="Processing frames", unit="frame", colour="blue") as pbar:
        for filename in tif_files:
            file_path_tif = os.path.join(tif_output_dir, filename)
            frame = tifffile.imread(file_path_tif)
            frames_720p.append(frame)
            frame_count += 1
            pbar.update(1)
    
    # Prepare video metadata
    video_info = {
        'file_path': file_path,
        'format': 'mxf',
        'codec': 'j2c',
        'original_resolution': f"{width}x{height}",
        'frame_rate': frame_rate,
        'total_frames': frame_count
    }
    
    save_video_metadata(file_path, video_info)
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print_success(f"Successfully processed {frame_count} frames and converted to 720p")
    print(f"{Fore.WHITE}⏱️  Total execution time: {execution_time:.2f} seconds")
    
    # Clean up temporary directory and all its contents
    # if os.path.exists(temp_dir):
    #     shutil.rmtree(temp_dir)
    
    return frames_720p, video_info, execution_time
