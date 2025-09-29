import subprocess
import time
import os
import tqdm
import math

input_movie = r"../movie/Empuraan_trailer.mxf"
input_directory = r"j2c_images"
output_directory = r"grok_output"
reduce_factor = [3,2,1,0]

os.makedirs(input_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)


def mxf_to_j2c(input_movie, output_directory):
    start_time = time.time()
    cmd = f"ffmpeg -i {input_movie} -c:v copy {output_directory}/frame_%06d.j2c -y"
    subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return time.time() - start_time

def batch_decode_to_tiff(input_directory, output_directory, output_format="tif", reduce_factor=reduce_factor):
    cmd = f"grk_decompress -y {input_directory} -a {output_directory} -O {output_format} -r {reduce_factor} -H {4}"
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, time.time() - start_time

#Step 1: MXF to J2C
time_taken = mxf_to_j2c(input_movie, input_directory)
file_count = len([f for f in os.listdir(input_directory) if f.endswith(('.j2c'))])
print(f"MXF to J2C : {file_count/time_taken:.2f} FPS\n")   

h,w = 1716,4096
#Step 2: J2C to TIFF
for reduce_factor in reduce_factor:
    success, time_taken = batch_decode_to_tiff(input_directory,output_directory, reduce_factor=reduce_factor)    
    print(f"J2C to TIFF for reduce factor {reduce_factor} : {file_count/time_taken:.2f} FPS, {math.ceil(h/2**reduce_factor)}x{math.ceil(w/2**reduce_factor)} resolution")



