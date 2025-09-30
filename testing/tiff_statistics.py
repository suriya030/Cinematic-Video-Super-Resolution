import tifffile
import numpy as np
import matplotlib.pyplot as plt

def analyze_tiff(file_path):
    """Analyzes a TIFF file and prints key statistics."""
    image_data = tifffile.imread(file_path)
    
    with tifffile.TiffFile(file_path) as tif:
        page = tif.pages[0]

        # Extract statistics
        colorspace = page.tags['PhotometricInterpretation'].value.name
        bit_depth = image_data.dtype
        height, width = image_data.shape[:2]
        min_val, max_val, avg_val = np.min(image_data), np.max(image_data), np.mean(image_data)

        # Print results
        print(f"--- Analysis for: {file_path} ---")
        print(f"  Resolution : {width}x{height}")
        print(f"  Colorspace : {colorspace} ")
        print(f"  Bit Depth  : {bit_depth}")
        print(f"  Pixel Stats: Min={min_val}, Max={max_val}, Avg={avg_val:.2f}")

# --- Set your file path and run the analysis ---
tiff_file_to_analyze = "tmp/tif_frames/frame_001976.tif" 
analyze_tiff(tiff_file_to_analyze)

# check the order RGB or BGR when using tifffile.imread
image_data = tifffile.imread(tiff_file_to_analyze)
image_data = (image_data >> 4).astype(np.uint8)
plt.imshow(image_data)
plt.show()