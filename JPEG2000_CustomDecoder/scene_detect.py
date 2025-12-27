"""Scene detection using xp_python frame extractor and PySceneDetect."""

import argparse
import os
import re
import sys
import time
from typing import List, Optional

import numpy as np

# Add vcpkg DLLs to search path (Windows only, required for native dependencies)
if sys.platform == 'win32':
    vcpkg_root = os.environ.get('VCPKG_ROOT', r'C:\dev\vcpkg')
    dll_path = os.path.join(vcpkg_root, 'installed', 'x64-windows', 'bin')
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)


OUTPUT_DIR = "scene_cuts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect scene cuts in video files using PySceneDetect."
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Path to the MXF video file",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Starting frame number (optional, uses decoder default if not specified)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Number of frames to process (optional, uses entire file if not specified)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="Scene detection threshold (default: 27.0, higher = less sensitive)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for scene cut frames (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--drop-levels",
        type=int,
        default=2,
        help="JPEG2K resolution drop levels for faster processing (default: 2)",
    )
    return parser.parse_args()


def save_frame(frame_data: np.ndarray, path: str, silent: bool = False) -> None:
    """Save a BGR frame to disk as PNG (converts to RGB)."""
    try:
        from PIL import Image
        # Convert BGR to RGB for correct color display
        rgb_frame = frame_data[:, :, ::-1]
        img = Image.fromarray(rgb_frame)
        img.save(path)
        if not silent:
            print(f"Saved frame to: {os.path.abspath(path)}")
    except ImportError:
        try:
            import cv2
            cv2.imwrite(path, frame_data)  # OpenCV expects BGR, so no conversion
            if not silent:
                print(f"Saved frame to: {os.path.abspath(path)}")
        except ImportError:
            print("Neither Pillow nor OpenCV available to save frame", file=sys.stderr)


def display_frame(frame_data: np.ndarray) -> None:
    """Display a BGR frame on screen."""
    try:
        import cv2
        cv2.imshow("Last Extracted Frame", frame_data)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except ImportError:
        try:
            from PIL import Image
            # Convert BGR to RGB for correct display
            rgb_frame = frame_data[:, :, ::-1]
            img = Image.fromarray(rgb_frame)
            img.show()
        except ImportError:
            print("Neither OpenCV nor Pillow available to display frame", file=sys.stderr)


def main() -> int:
    args = parse_args()
    
    try:
        import xp_python
    except Exception as exc:  # pragma: no cover - failure path is the point of this script
        print(f"Failed to import xp_python: {exc}", file=sys.stderr)
        return 1

    try:
        import scenedetect
        from scenedetect.detectors import ContentDetector
        from scenedetect.frame_timecode import FrameTimecode
    except Exception as exc:  # pragma: no cover - runtime dependency probe
        print(f"Failed to import PySceneDetect: {exc}", file=sys.stderr)
        return 5

    doc = getattr(xp_python, "__doc__", "<missing>")
    print("Successfully imported xp_python")
    print(f"Processing: {args.filename}")
    start_info = f"from frame {args.start_frame}" if args.start_frame is not None else "from start"
    duration_info = f"for {args.duration} frames" if args.duration is not None else "until end"
    print(f"Processing {start_info} {duration_info} (threshold: {args.threshold})")

    detected: List[str] = []
    # Higher threshold = less sensitive (fewer false positives)
    detector = ContentDetector(threshold=args.threshold, min_scene_len=12, luma_only=False)
    last_frame_data: Optional[np.ndarray] = None
    scene_count = 0
    
    # Create output directory for scene cut frames
    os.makedirs(args.output_dir, exist_ok=True)

    if hasattr(xp_python, "FrameExtractor"):
        # Build extractor kwargs, only including optional params if specified
        extractor_kwargs = {
            "filename": args.filename,
            "drop_levels": args.drop_levels,
        }
        if args.start_frame is not None:
            extractor_kwargs["start_frame"] = args.start_frame
        if args.duration is not None:
            extractor_kwargs["duration"] = args.duration
        
        extractor = xp_python.FrameExtractor(**extractor_kwargs)
        timecode_pattern = re.compile(r"^\d{1,}:\d{2}:\d{2}\.\d{3}$")
        count = 0
        save_time_total = 0.0  # Track time spent saving frames (excluded from benchmark)
        start_time = time.perf_counter()  # Start benchmark timer
        
        # Process each frame immediately as it's extracted
        for frame in extractor:
            tc = frame["timecode"]
            if not isinstance(tc, str) or not timecode_pattern.fullmatch(tc):
                print("frame timecode has unexpected format", file=sys.stderr)
                return 2
            
            frame_number = int(frame["frame_number"])
            frame_data = frame.get("data")

            if frame_data is None:
                print(f"Warning: frame {frame_number} has no data", file=sys.stderr)
            else:
                # Keep reference to last frame for saving/display
                last_frame_data = frame_data
                frame_width = frame.get("width", 0)
                frame_height = frame.get("height", 0)
                # Run scene detection on each frame immediately
                for cut in detector.process_frame(frame_number, frame_data):
                    scene_count += 1
                    cut_tc = FrameTimecode(cut, fps=frame["fps"]).get_timecode()
                    detected.append(f"frame {cut} ({cut_tc})")
                    # Save the frame where the scene cut was detected
                    output_path = os.path.join(args.output_dir, f"scene_{scene_count:04d}_frame_{cut}.png")
                    save_start = time.perf_counter()
                    save_frame(frame_data, output_path)
                    save_time_total += time.perf_counter() - save_start
                    print(f"Scene cut #{scene_count} at frame {cut} ({cut_tc} {frame_width}x{frame_height})")
            
            count += 1
        
        end_time = time.perf_counter()
        
        # Calculate benchmark (excluding save time)
        total_time = end_time - start_time
        processing_time = total_time - save_time_total
        fps = count / processing_time if processing_time > 0 else 0
        
        print(f"\n--- Benchmark ---")
        print(f"Frames processed: {count}")
        print(f"Processing time: {processing_time:.2f}s (excludes {save_time_total:.2f}s saving)")
        print(f"Processing FPS: {fps:.2f}")
        print(f"Total wall time: {total_time:.2f}s")
        
        if args.duration is not None and count != args.duration:
            print(
                f"expected {args.duration} frames from iterator, got {count}",
                file=sys.stderr,
            )
            return 3
        if extractor.extract() is not None:
            print("extract() should return None after iterator is exhausted", file=sys.stderr)
            return 4
    else:
        print("Warning: FrameExtractor class is missing")
        return 1

    print(f"PySceneDetect available (version {scenedetect.__version__})")
    if detected:
        print(f"PySceneDetect detected {len(detected)} scene cuts")
        print(f"Scene cut frames saved to: {os.path.abspath(args.output_dir)}/")
    else:
        print("PySceneDetect did not find any cuts in the sequence")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
