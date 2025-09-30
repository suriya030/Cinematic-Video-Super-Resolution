import scene_detector
video_path = r'movie\Empuraan_trailer.mp4'

# frame_rate = 24.0
# detected_scenes, execution_time = scene_detection(video_path, frame_rate)
# print(detected_scenes)
# print(execution_time)


import quality_analyzer
quality_metrics, device = quality_analyzer.initialize_quality_metrics()
# print(quality_metrics)
# print(device)

print(not quality_metrics['niqe'])
