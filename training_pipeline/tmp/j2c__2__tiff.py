# Here we are testing the lossless compression of J2C files to TIFF files using zlib compression algorithm
# and vice especially the line tiff.imwrite(output_path, original_array, compression='zlib')

import glymur
import numpy as np
import tifffile as tiff
import os

input_folder = r"tmp\j2c_images"
output_folder = r"tmp\tiff_images"

all_conversions_successful = True

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_filename = os.path.splitext(filename)[0] + '.tif'
    output_path = os.path.join(output_folder, output_filename)
    
    original_array = glymur.Jp2k(input_path)[:]
    tiff.imwrite(output_path, original_array, compression='zlib')
    reloaded_array = tiff.imread(output_path)
    # print( type(reloaded_array)) # <class 'numpy.ndarray'>
    
    are_arrays_equal = np.array_equal(original_array, reloaded_array)
    
    if not are_arrays_equal:
        all_conversions_successful = False

if all_conversions_successful:
    print("✅ Verification Successful: All J2C files were converted to TIFF losslessly.")
