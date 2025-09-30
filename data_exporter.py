import json
from colorama import Fore
from utils import ensure_directory, print_processing, print_success

def save_analysis_json(metadata, detected_scenes, scene_results, output_path):
    """ args: metadata [dict], detected_scenes [list of dicts], scene_results [list of dicts], output_path [str]
    return: analysis_data [dict] """

    print_processing("Preparing analysis data for export...")
    # Step 1: Save metadata first
    analysis_data = {
        'metadata': metadata
    }
    
    # Step 2: Save detected scenes and add quality sequence info if available
    scenes_with_quality_info = []
    
    for scene in detected_scenes:
        scene_id = scene['scene_id']
        scene_data = scene.copy()
        scene_result = scene_results[scene_id - 1]
        
        if scene_result['sequence_found']:
            scene_data['frames_selected'] = True
            scene_data['selected_frame_range'] = {
                'start_frame': scene_result['selected_frames'][0],
                'end_frame': scene_result['selected_frames'][-1]
            }
        else:
            scene_data['frames_selected'] = False
            scene_data['selected_frame_range'] = None
        
        scenes_with_quality_info.append(scene_data)
    
    # Calculate summary statistics
    total_sequences_found = sum(1 for scene in scenes_with_quality_info if scene['frames_selected'])
    
    # Add scene detection data
    analysis_data['scene_detection'] = {
        'total_scenes_detected': len(detected_scenes),
        'total_scenes_with_sequences': total_sequences_found,
        'scenes': scenes_with_quality_info
    }
    
    # Step 3: Save to JSON file
    ensure_directory(output_path)
    with open(output_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print_success(f"Analysis data saved to: {Fore.YELLOW}{output_path}")
    return analysis_data