import subprocess
import time
import os
import tqdm

# Paths
# input_directory = r"A:\testing\frames" # 4k frames
# output_directory_single = r"A:\testing\frames_grok_single"
# output_directory_batch = r"A:\testing\frames_grok_batch"
input_directory = r"tmp\j2c_images"
output_directory_single = r"tmp\grok_output_single"
output_directory_batch = r"tmp\grok_output_batch"


def batch_decode_to_tiff(input_directory, output_directory, output_format="tif", reduce_factor=2):
    cmd = f"grk_decompress -y {input_directory} -a {output_directory} -O {output_format} -r {reduce_factor} -H {4}"
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, time.time() - start_time

# Usage - Batch processing
os.makedirs(output_directory_batch, exist_ok=True)
success, time_taken = batch_decode_to_tiff(input_directory,output_directory_batch, reduce_factor=1)
file_count = len([f for f in os.listdir(input_directory) if f.endswith(('.j2c'))])
print(f"Batch processing performance: {file_count/time_taken:.2f} FPS (frames per second)")
print(f"Total processing time: {time_taken:.2f} seconds")


import tifffile
import numpy as np

tiff_files = [f for f in os.listdir(output_directory_batch) if f.lower().endswith(('.tif'))]
sample_image_path = os.path.join(output_directory_batch, tiff_files[0])

tif = tifffile.TiffFile(sample_image_path)
image = tif.asarray()

print(f"\n=== TIFF Image Analysis: {os.path.basename(sample_image_path)} ===")
print(f"Resolution: {image.shape[1]} x {image.shape[0]} pixels")
print(f"Channels: {image.shape[2] if len(image.shape) > 2 else 1}")
print(f"Data type: {image.dtype}")
print(f"Bit depth: {image.dtype.itemsize * 8} bits")
print(f"Image shape: {image.shape}")

page = tif.pages[0]
print(f"Compression: {page.compression}")
print(f"Photometric: {page.photometric}")
print(f"Planar config: {page.planarconfig}")

if hasattr(page, 'colormap') and page.colormap is not None:
    print("Color space: Indexed color (with colormap)")
elif len(image.shape) == 3:
    if image.shape[2] == 3:
        print("Color space: RGB")
    elif image.shape[2] == 4:
        print("Color space: RGBA")
    else:
        print(f"Color space: Multi-channel ({image.shape[2]} channels)")
else:
    print("Color space: Grayscale")

print(f"Min value: {np.min(image)}")
print(f"Max value: {np.max(image)}")
print(f"Mean value: {np.mean(image):.2f}")
print(f"Standard deviation: {np.std(image):.2f}")

tif.close()

