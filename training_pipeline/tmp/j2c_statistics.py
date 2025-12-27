# Verifying 
# 1. pixel depth is 12 bits stored in uint16
# 2. resolution is (1716,4096,3)

import numpy as np
import glymur
import os
import random

# --- 1. Load the Frame ---
folder_path = r"A:\testing\frames"
# folder_path = r"tmp\j2c_images"
random_frame = random.choice(os.listdir(folder_path))
random_frame_path = os.path.join(folder_path, random_frame)
frame_array = glymur.Jp2k(random_frame_path)[:]

# --- 2. Calculate Statistics ---
print("\n--- Frame Properties ---")
print(f"Shape (Height, Width, Channels): {frame_array.shape}")
print(f"Data Type:                       {frame_array.dtype}")

print("\n--- Pixel Value Statistics ---")
min_value = np.min(frame_array)
max_value = np.max(frame_array)
mean_value = np.mean(frame_array)

# --- 3. Print the Results ---
print(f"Minimum Pixel Value: {min_value}")
print(f"Maximum Pixel Value: {max_value}")
print(f"Mean Pixel Value:    {mean_value:.2f}")