import os
from colorama import Fore, Style

# Import all modules
from v2_video_reader import read_mxf_video
# from video_reader import read_mxf_video
from scene_detector import scene_detection
from quality_analyzer import initialize_quality_metrics, find_sequence_per_scene
from data_exporter import save_analysis_json
from utils import print_header, print_step, print_success, get_base_filename
from config import DEVICE,QUALITY_ANALYSIS

def mxf_pipeline(mxf_file_path, output_folder, is_timetaken=True):
    """Complete MXF video pipeline: 1. read video, 2. detect scenes, 3. find quality frames per scene 
        and 4. save results to json"""
    
    # Step 1: Read and convert 4K .mxf video into 720p list of ndarrays
    print_step(1, "Reading MXF file and converting to frames list")
    frames_ndarray, metadata, execution_time1 = read_mxf_video(mxf_file_path)
    
    # Step 2: Detect video scenes
    print_step(2, "Scene detection")
    detected_scenes, execution_time2 = scene_detection(mxf_file_path, metadata['frame_rate'])
    
    # Step 3: Find quality frame sequence per scene
    print_step(3, "Finding quality frame sequence per scene")
    base_video_name = get_base_filename(mxf_file_path)
    scene_results, execution_time3 = find_sequence_per_scene(frames_ndarray, detected_scenes, base_video_name)
    
    # Step 4: Save results to json
    print_step(4, "Saving Selected Frames info per Scene to json")
    json_output_path = os.path.join(output_folder, f"{base_video_name}_analysis.json")
    analysis_data = save_analysis_json(metadata, detected_scenes, scene_results, json_output_path)

    # Time taken for the pipeline
    total_execution_time = execution_time1 + execution_time2 + execution_time3
    if is_timetaken:    
        print(f"\n\n{Fore.WHITE}  Time taken breakdown:")
        print(f"{Fore.WHITE}  Step 1: {execution_time1:.2f} seconds (reading .mxf file)")
        print(f"{Fore.WHITE}  Step 2: {execution_time2:.2f} seconds (scene detection)")
        print(f"{Fore.WHITE}  Step 3: {execution_time3:.2f} seconds (iqa analysis)")
    print(f"{Fore.WHITE}⏱️  Total execution time: {total_execution_time:.2f} seconds ({len(frames_ndarray)} frames processed from .mxf file)")

    # Final summary
    sequences_found = sum(1 for sr in scene_results if sr['sequence_found'])
    total_scenes = len(detected_scenes)
    
    print(f"\n{Fore.GREEN}{'='*80}")
    print(f"{Fore.GREEN}🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{Fore.GREEN}📊 Results: Found quality sequences in {Fore.YELLOW}{sequences_found}/{total_scenes}{Fore.GREEN} scenes")
    print(f"{Fore.WHITE}Average FPS of the pipeline: {len(frames_ndarray)/total_execution_time:.2f} fps")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    
    return analysis_data

def check_mxf_files(mxf_files,output_folder):

    processed_mxf_files = []
    for filename in mxf_files:
        base_name = os.path.splitext(filename)[0]
        expected_output_file = os.path.join(output_folder, f"{base_name}_analysis.json")
        
        if os.path.exists(expected_output_file):
            processed_mxf_files.append(filename)

    return processed_mxf_files

if __name__ == "__main__":
    
    print_header("MXF Data Pipeline")

    input_folder = "movie"
    output_folder = "analysis_results"
        
    mxf_files = [f for f in os.listdir(input_folder) if f.endswith('.mxf') ]
    processed_mxf_files = check_mxf_files( mxf_files, output_folder)
    
    for i,filename in enumerate(mxf_files):
        if filename in processed_mxf_files:
            continue
        print(f"{Fore.WHITE}Processing file {i+1}/{len(mxf_files)-len(processed_mxf_files)}: {filename}")

        mxf_file_path = os.path.join(input_folder, filename)
        analysis_data = mxf_pipeline(mxf_file_path, output_folder, is_timetaken=True)

    print_header(" ALL FILES PROCESSED")