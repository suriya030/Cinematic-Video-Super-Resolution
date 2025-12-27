import tifffile
import numpy as np

# Replace with the path to your TIFF file
file_path = r'A:\testing\frames_grok_batch\frame_000001.tif'

# Read the TIFF file into a NumPy array
image_array = tifffile.imread(file_path)

# Calculate and print the min, max, and mean pixel values
print(f"Minimum Pixel Value: {np.min(image_array)}")
print(f"Maximum Pixel Value: {np.max(image_array)}")
print(f"Mean Pixel Value: {np.mean(image_array):.2f}")