import cv2
import time
from v2_video_reader import read_mxf_video

def frames_to_mp4(frames, output_path, fps=24):
    start_time = time.time()
    
    height, width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    return time.time() - start_time

def main():
    input_video = "movie/Empuraan_trailer.mxf"
    output_video = "output_24fps.mp4"
    
    # Process 1: Read frames using v2_frame_reader
    frames, metadata, reading_time = read_mxf_video(input_video, reduce_factor=0)
    
    # Process 2: Convert frames to 24 fps MP4
    conversion_time = frames_to_mp4(frames, output_video, fps=24)
    
    # Results
    print(f"Frame reading: {reading_time:.2f} seconds ({len(frames)} frames)")
    print(f"MP4 conversion: {conversion_time:.2f} seconds")
    print(f"Total time: {reading_time + conversion_time:.2f} seconds")

if __name__ == "__main__":
    main()