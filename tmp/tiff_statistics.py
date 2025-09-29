import tifffile
import numpy as np
import os

output_directory_batch = r"tmp\grok_output_batch"
    
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