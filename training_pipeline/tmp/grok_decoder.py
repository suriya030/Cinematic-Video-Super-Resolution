import subprocess
import time
import os
import tqdm

# Paths
input_directory = r"A:\testing\frames" # 4k frames
output_directory_single = r"A:\testing\frames_grok_single"
output_directory_batch = r"A:\testing\frames_grok_batch"
# input_directory = r"tmp\j2c_images"
# output_directory_single = r"tmp\grok_output_single"
# output_directory_batch = r"tmp\grok_output_batch"

def decode_j2c_to_tiff(input_file, output_file, reduce_factor=2):
    cmd = f"grk_decompress -i {input_file} -o {output_file} -r {reduce_factor} -H {os.cpu_count()}"
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, time.time() - start_time

def batch_decode_to_tiff(input_directory, output_directory, output_format="tif", reduce_factor=2):
    cmd = f"grk_decompress -y {input_directory} -a {output_directory} -O {output_format} -r {reduce_factor} -H {4}"
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, time.time() - start_time

# Usage - Single file processing (loop through directory)
# os.makedirs(output_directory_single, exist_ok=True)
# files = [ f for f in os.listdir(input_directory) if f.endswith(('.j2c'))]
# print( " Total frames to process: ", len(files))
# total_time = 0
# for filename in tqdm.tqdm(files):
#     input_file = os.path.join(input_directory, filename) #;print(input_file)
#     output_file = os.path.join(output_directory_single, filename.replace('.j2c', '.tif')) #;print(output_file)
#     success, time_taken = decode_j2c_to_tiff(input_file, output_file, reduce_factor=3)
#     total_time += time_taken

# print(f"Sequential file processing performance: {len(files)/total_time:.2f} FPS (frames per second)")

# Usage - Batch processing
os.makedirs(output_directory_batch, exist_ok=True)
success, time_taken = batch_decode_to_tiff(input_directory,output_directory_batch, reduce_factor=1)
file_count = len([f for f in os.listdir(input_directory) if f.endswith(('.j2c'))])
print(f"Batch processing performance: {file_count/time_taken:.2f} FPS (frames per second)")
print(f"Total processing time: {time_taken:.2f} seconds")

# Just now, I have tested the grok library in the local system (CPU cores = 24 ) ( their instance does not have the internet right now )
# 1. For reading frames at 2048x858: 32 fps 
# 2. For reading frames at 1024x429: 85.91 fps
# 3. For reading frames at 512x215: 150 fps
