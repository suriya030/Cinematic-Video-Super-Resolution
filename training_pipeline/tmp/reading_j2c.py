# Comparing default reading and reading with CPU multi-threading

import glymur
import numpy as np
import os
import time

def read_j2c_frames_with_timing(folder_path, output_folder, use_multithreading=False):

    if use_multithreading:
        max_threads = os.cpu_count()
        print(f"Using {max_threads} threads")
        glymur.set_option('lib.num_threads', max_threads)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    times = []
    frame_array = None
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.j2c'):
            filepath = os.path.join(folder_path, filename)
            start_time = time.time()
            frame_array = glymur.Jp2k(filepath)[:]
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Save the numpy array
            output_filename = os.path.splitext(filename)[0] + '.npy'
            output_path = os.path.join(output_folder, output_filename)
            np.save(output_path, frame_array)
    
    avg_time = np.mean(times)
    avg_fps = 1 / avg_time
    
    return avg_time, avg_fps, frame_array

# --- 1. Load All Frames ---
folder_path = r"tmp\j2c_images"
output_folder = r"tmp\raw_images"

print("\nJ2c to numpy array [Default]")
avg_time_default, avg_fps_default, frame_array = read_j2c_frames_with_timing(folder_path, output_folder, use_multithreading=False)
print(f"Average Time: {avg_time_default:.2f} sec") # 2 decimal places
print(f"Average fps : {avg_fps_default:.2f}") # 2 decimal places

print("\nJ2c to numpy array [CPU multi-threading]")
avg_time_multithreaded, avg_fps_multithreaded, _ = read_j2c_frames_with_timing(folder_path, output_folder, use_multithreading=True)
print(f"Average Time: {avg_time_multithreaded:.2f} sec") # 2 decimal places
print(f"Average fps : {avg_fps_multithreaded:.2f}") # 2 decimal places


