import json
import os
import av
import cv2
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    # Configuration
    analysis_results_folder = "analysis_results"
    extracted_frames_folder = "extracted_frames"
    
    # Process all JSON files in the analysis_results folder
    process_all_analysis_files(analysis_results_folder, extracted_frames_folder)