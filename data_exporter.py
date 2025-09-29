import json
from colorama import Fore
from config import OUTPUT
from utils import ensure_directory, print_processing, print_success

def save_analysis_json(video_info, scenes_info, scene_results, output_path):
    """Save complete analysis results to JSON file"""
    print_processing("Preparing analysis data for export...")
    
    # Add frame selection info to each scene
    for scene in scenes_info:
        scene_id = scene['scene_id']
        scene_result = next((sr for sr in scene_results if sr['scene_id'] == scene_id), None)
        
        if scene_result:
            scene['frames_selected'] = scene_result['sequence_found']
            if scene_result['sequence_found']:
                scene['selected_frame_range'] = {
                    'start_frame': scene_result['selected_frames'][0],
                    'end_frame': scene_result['selected_frames'][-1],
                    'total_selected': len(scene_result['selected_frames'])
                }
            else:
                scene['selected_frame_range'] = None
        else:
            scene['frames_selected'] = False
            scene['selected_frame_range'] = None
    
    # Calculate summary statistics
    total_sequences_found = sum(1 for sr in scene_results if sr['sequence_found'])
    total_frames_selected = sum(sr['total_frames_selected'] for sr in scene_results)
    
    # Prepare final analysis data
    analysis_data = {
        'video_information': video_info,
        'scene_detection': {
            'total_scenes_detected': len(scenes_info),
            'scenes': scenes_info
        },
        'quality_analysis': {
            'total_scenes_with_sequences': total_sequences_found,
            'total_frames_selected': total_frames_selected,
            'scene_results': scene_results
        }
    }
    
    # Save to JSON file
    ensure_directory(output_path)
    with open(output_path, 'w') as f:
        json.dump(analysis_data, f, indent=OUTPUT['json_indent'])
    
    print_success(f"Analysis data saved to: {Fore.YELLOW}{output_path}")
    return analysis_data