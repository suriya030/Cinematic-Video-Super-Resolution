from v2_video_reader import read_mxf_video
import math

reduce_factor = [0, 1, 2, 3]
h,w = 1716,4096
input_movie = r"movie/Empuraan_trailer.mxf" 
for reduce_factor in reduce_factor:
    frames_ndarray, metadata, execution_time = read_mxf_video(input_movie, reduce_factor)
    print(f" Resolution {math.ceil(h/2**reduce_factor)}x{math.ceil(w/2**reduce_factor)} FPS {int(metadata['total_frames'])/execution_time:.2f}")