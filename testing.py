import os
import shutil
# This will work great on Linux
def cleanup_tmp(tmp_directories):
    
    for tmp_dir in tmp_directories:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

tmp_directories = [
    "tmp/j2c_frames", 
    "tmp/tif_frames",
    "tmp/test_scenes"
]
cleanup_tmp(tmp_directories)