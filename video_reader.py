import av
import cv2
import os
import json
import time
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

def save_video_metadata(file_path, container, video_stream, frame_rate, total_frames):
    """Save detailed video metadata to logs/metadata folder"""
    metadata = {
        'file_path': file_path,
        'format': container.format.name,
        'duration': f"{container.duration / av.time_base:.2f}",
        'codec': video_stream.codec.name,
        'resolution': f"{video_stream.width}x{video_stream.height}",
        'frame_rate': f"{frame_rate}",
        'pixel_format': video_stream.pix_fmt,
        'bitrate': container.bit_rate,
        'total_frames': total_frames
    }
    
    # Create metadata filename based on input video name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    metadata_dir = os.path.join('logs', 'metadata')
    metadata_path = os.path.join(metadata_dir, f"{base_name}_metadata.json")
    
    # Save metadata as JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def convert_frame_to_720p(frame):
    """Convert frame to 720p resolution at 8-bit depth"""
    frame_720p = frame.reformat(
        width=VIDEO_CONVERSION['target_width'], 
        height=VIDEO_CONVERSION['target_height'], 
        format=VIDEO_CONVERSION['format']
    )
    return frame_720p.to_ndarray()

def read_mxf_video(file_path):
    """Read 4K .mxf video file and convert frames to 720p list of ndarrays"""
    start_time = time.time()
    print_processing("Opening MXF video file...")
    
    # Open video containers
    container = av.open(file_path)
    video_stream = container.streams.video[0]
    
    # Get accurate frame info using OpenCV
    cap = cv2.VideoCapture(file_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print_video_info(file_path, total_frames)
    save_video_metadata(file_path, container, video_stream, frame_rate, total_frames)
    
    # Prepare video metadata
    video_info = {
        'file_path': file_path,
        'format': container.format.name,
        'codec': video_stream.codec.name,
        'original_resolution': f"{video_stream.width}x{video_stream.height}",
        'frame_rate': frame_rate,
        'pixel_format': video_stream.pix_fmt,
        'bitrate': container.bit_rate,
        'total_frames': total_frames
    }
    
    print_processing("Converting frames to 720p...")
    frame_count = 0
    frames_720p = []
    
    # Process each frame with progress tracking
    with tqdm(total=total_frames, desc="Processing frames", unit="frame", colour="blue") as pbar:
        for frame in container.decode(video_stream):
            frame_count += 1
            converted_frame = convert_frame_to_720p(frame)
            frames_720p.append(converted_frame)
            pbar.update(1)
    
    container.close()
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print_success(f"Successfully processed {frame_count} frames and converted to 720p")
    print(f"{Fore.WHITE}⏱️  Total execution time: {execution_time:.2f} seconds")
    
    return frames_720p, video_info, execution_time