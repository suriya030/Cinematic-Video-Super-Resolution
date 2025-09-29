import os
from colorama import Fore, Style

# Import all modules
from v2_video_reader import read_mxf_video
from scene_detector import find_and_split_scenes
from quality_analyzer import initialize_quality_metrics, find_sequence_per_scene,find_sequences_per_scene_niqe_only
from data_exporter import save_analysis_json
from utils import print_header, print_step, print_success, get_base_filename
from config import DEVICE,QUALITY_ANALYSIS

def mxf_pipeline(mxf_file_path, output_folder, is_timetaken=True):
    """Complete MXF video pipeline: 1. read video, 2. detect scenes, 3. find quality frames per scene 
        and 4. save results to json"""
    
    # Step 1: Read and convert 4K .mxf video into 720p list of ndarrays
    print_step(1, "Reading MXF file and converting to frames list")
    frames_list, video_info, execution_time1 = read_mxf_video(mxf_file_path)
    
    # Step 2: Detect video scenes
    print_step(2, "Scene detection")
    scenes_info, execution_time2 = find_and_split_scenes(mxf_file_path, video_info['frame_rate'])
    
    # Step 3A: Initialize quality analysis models 
    print_step('3A', "Initializing IQA models")
    musiq_metric, niqe_metric, device, execution_time3 = initialize_quality_metrics(DEVICE, 
                                                        QUALITY_ANALYSIS['use_musiq'])
    
    # Step 3B: Analyze frame quality per scene
    print_step('3B', "Finding quality frames per scene")
    base_video_name = get_base_filename(mxf_file_path)
    if QUALITY_ANALYSIS['use_musiq']:
        scene_results = find_sequence_per_scene(frames_list, scenes_info, base_video_name, musiq_metric, niqe_metric, device)
    else:
        scene_results, execution_time4 = find_sequences_per_scene_niqe_only(frames_list, scenes_info, base_video_name, niqe_metric, device)
    
    # Step 4: Save results 
    print_step(4, "Saving Selected Frames info per Scene")
    json_output_path = os.path.join(output_folder, f"{base_video_name}_analysis.json")
    analysis_data = save_analysis_json(video_info, scenes_info, scene_results, json_output_path)

    # Time taken for the pipeline
    total_execution_time = execution_time1 + execution_time2 + execution_time3 + execution_time4
    if is_timetaken:    
        print(f"\n\n{Fore.WHITE}  Time taken breakdown:")
        print(f"{Fore.WHITE}  Step 1: {execution_time1:.2f} seconds")
        print(f"{Fore.WHITE}  Step 2: {execution_time2:.2f} seconds")
        print(f"{Fore.WHITE}  Step 3A: {execution_time3:.2f} seconds")
        print(f"{Fore.WHITE}  Step 3B: {execution_time4:.2f} seconds")
    print(f"{Fore.WHITE}⏱️  Total execution time: {total_execution_time:.2f} seconds ({len(frames_list)} frames processed from .mxf file)")

    # Final summary
    sequences_found = sum(1 for sr in scene_results if sr['sequence_found'])
    total_scenes = len(scenes_info)
    
    print(f"\n{Fore.GREEN}{'='*80}")
    print(f"{Fore.GREEN}🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{Fore.GREEN}📊 Results: Found quality sequences in {Fore.YELLOW}{sequences_found}/{total_scenes}{Fore.GREEN} scenes")
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