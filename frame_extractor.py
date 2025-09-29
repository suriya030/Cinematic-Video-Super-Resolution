import json
import os
import av
import cv2
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style
from config import DIRECTORIES

def load_analysis_json(json_path):
    with open(json_path, 'r') as f:
        analysis_data = json.load(f)
    
    video_path = analysis_data['video_information']['file_path']
    scenes_with_frames = []
    
    for scene in analysis_data['scene_detection']['scenes']:
        if scene['frames_selected'] and scene['selected_frame_range']:
            scenes_with_frames.append({
                'scene_id': scene['scene_id'],
                'selected_frames': analysis_data['quality_analysis']['scene_results'][scene['scene_id']-1]['selected_frames']
            })
    
    return video_path, scenes_with_frames

def get_all_selected_frames(scenes_with_frames):
    all_frames = {}
    for scene_data in scenes_with_frames:
        scene_id = scene_data['scene_id']
        for frame_number in scene_data['selected_frames']:
            all_frames[frame_number] = scene_id
    return all_frames

def read_selected_frames_once(video_path, selected_frames_dict):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    
    if total_frames == 0:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    extracted_frames = {}
    current_frame = 0
    
    with tqdm(total=total_frames, desc="Reading video frames", unit="frame") as pbar:
        for frame in container.decode(video_stream):
            current_frame += 1
            pbar.update(1)
            
            if current_frame in selected_frames_dict:
                extracted_frames[current_frame] = frame
    
    container.close()
    return extracted_frames

def save_frames_to_png(extracted_frames, scenes_with_frames, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for scene_data in scenes_with_frames:
        scene_id = scene_data['scene_id']
        selected_frames = scene_data['selected_frames']
        
        scene_folder = os.path.join(output_folder, f"scene_{scene_id:02d}")
        os.makedirs(scene_folder, exist_ok=True)
        
        for frame_number in tqdm(selected_frames, desc=f"Saving Scene {scene_id}"):
            if frame_number in extracted_frames:
                frame = extracted_frames[frame_number]
                output_filename = f"frame_{frame_number:06d}.png"
                output_path = os.path.join(scene_folder, output_filename)
                
                # Convert xyz12le to rgb48le (16-bit RGB) using PyAV
                rgb_frame = frame.reformat(format='rgb48le')
                frame_array = rgb_frame.to_ndarray()
                
                # OpenCV expects BGR format for 16-bit PNG
                bgr_16bit = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                
                # Save as 16-bit PNG using OpenCV
                cv2.imwrite(output_path, bgr_16bit)

def extract_movie_name_from_json(json_filename):
    """Extract movie name from JSON filename by removing '_analysis.json' suffix"""
    if json_filename.endswith('_analysis.json'):
        return json_filename[:-13]  # Remove '_analysis.json'
    else:
        return os.path.splitext(json_filename)[0]

def get_json_files_to_process(analysis_results_folder, extracted_frames_folder):
    """Get list of JSON files that need processing"""
    if not os.path.exists(analysis_results_folder):
        print(f"{Fore.RED}❌ Analysis results folder '{analysis_results_folder}' not found!")
        return []
    
    json_files = [f for f in os.listdir(analysis_results_folder) if f.endswith('.json')]
    
    if not json_files:
        print(f"{Fore.YELLOW}⚠️ No JSON files found in '{analysis_results_folder}'")
        return []
    
    files_to_process = []
    
    for json_file in json_files:
        movie_name = extract_movie_name_from_json(json_file)
        movie_output_folder = os.path.join(extracted_frames_folder, movie_name)
        
        # Check if this movie has already been processed
        if os.path.exists(movie_output_folder) and os.listdir(movie_output_folder):
            print(f"{Fore.YELLOW}⏩ Skipping '{json_file}': Frames already extracted for '{movie_name}'")
            continue
        
        files_to_process.append(json_file)
    
    return files_to_process

def extract_selected_frames(json_path, output_folder):
    """Extract selected frames from a single JSON analysis file"""
    video_path, scenes_with_frames = load_analysis_json(json_path)
    
    if not scenes_with_frames:
        print(f"{Fore.YELLOW}⚠️ No scenes with selected frames found in {os.path.basename(json_path)}")
        return
    
    selected_frames_dict = get_all_selected_frames(scenes_with_frames)
    
    if not selected_frames_dict:
        print(f"{Fore.YELLOW}⚠️ No frames to extract from {os.path.basename(json_path)}")
        return
    
    print(f"{Fore.CYAN}📹 Processing: {os.path.basename(json_path)}")
    print(f"{Fore.CYAN}🎬 Video: {os.path.basename(video_path)}")
    print(f"{Fore.CYAN}📁 Output: {output_folder}")
    print(f"{Fore.CYAN}🎯 Total frames to extract: {len(selected_frames_dict)}")
    
    extracted_frames = read_selected_frames_once(video_path, selected_frames_dict)
    save_frames_to_png(extracted_frames, scenes_with_frames, output_folder)
    
    print(f"{Fore.GREEN}✅ Successfully extracted frames for {os.path.basename(json_path)}")

def process_all_analysis_files(analysis_results_folder, extracted_frames_folder):
    """Process all JSON files in the analysis_results folder"""
    
    print(f"{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}🚀 FRAME EXTRACTION PIPELINE")
    print(f"{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.CYAN}📂 Analysis folder: {analysis_results_folder}")
    print(f"{Fore.CYAN}📁 Output folder: {extracted_frames_folder}")
    
    # Get list of JSON files to process
    json_files_to_process = get_json_files_to_process(analysis_results_folder, extracted_frames_folder)
    
    if not json_files_to_process:
        print(f"{Fore.GREEN}✅ All files have been processed or no files found to process.")
        return
    
    print(f"{Fore.GREEN}📊 Found {len(json_files_to_process)} JSON files to process:")
    for json_file in json_files_to_process:
        movie_name = extract_movie_name_from_json(json_file)
        print(f"{Fore.WHITE}   • {json_file} → {movie_name}/")
    
    print(f"\n{Fore.CYAN}🎬 Starting frame extraction...")
    
    # Process each JSON file
    for i, json_file in enumerate(json_files_to_process, 1):
        print(f"\n{Fore.MAGENTA}{'─'*80}")
        print(f"{Fore.MAGENTA}📋 Processing file {i}/{len(json_files_to_process)}: {json_file}")
        print(f"{Fore.MAGENTA}{'─'*80}")
        
        try:
            # Create movie-specific output folder
            movie_name = extract_movie_name_from_json(json_file)
            movie_output_folder = os.path.join(extracted_frames_folder, movie_name)
            
            # Extract frames for this JSON file
            json_path = os.path.join(analysis_results_folder, json_file)
            extract_selected_frames(json_path, movie_output_folder)
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error processing {json_file}: {str(e)}")
            continue
    
    print(f"\n{Fore.GREEN}{'='*80}")
    print(f"{Fore.GREEN}🎉 FRAME EXTRACTION COMPLETED!")
    print(f"{Fore.GREEN}📊 Processed {len(json_files_to_process)} files")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")

if __name__ == "__main__":
    # Get folder paths from config
    analysis_results_folder = DIRECTORIES['analysis_results_dir']
    extracted_frames_folder = DIRECTORIES['extracted_frames_dir']
    
    # Process all JSON files in the analysis_results folder
    process_all_analysis_files(analysis_results_folder, extracted_frames_folder)