"""Extract a single frame with linear RGB output and save as TIFF."""

import argparse
import os
import sys

import numpy as np

# Add vcpkg DLLs to search path (Windows only, required for native dependencies)
if sys.platform == 'win32':
    vcpkg_root = os.environ.get('VCPKG_ROOT', r'C:\dev\vcpkg')
    dll_path = os.path.join(vcpkg_root, 'installed', 'x64-windows', 'bin')
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a single frame with linear RGB output and save as TIFF."
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
        help="Frame number to extract (optional, uses decoder default if not specified)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Number of frames to extract (optional, extracts all if not specified)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output TIFF path (default: frame_<start_frame>_linear.tiff)",
    )
    return parser.parse_args()


def apply_inverse_gamma(linear_data: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply inverse gamma correction to convert linear RGB to gamma-corrected RGB.
    
    Args:
        linear_data: Linear RGB float32 array with values in [0.0, 1.0]
        gamma: Gamma value (default: 2.2 for sRGB-like curve)
    
    Returns:
        Gamma-corrected RGB array with values in [0.0, 1.0]
    """
    # Clamp to valid range and apply inverse gamma: output = input^(1/gamma)
    clamped = np.clip(linear_data, 0.0, 1.0)
    return np.power(clamped, 1.0 / gamma)


def save_linear_tiff(frame_data: np.ndarray, path: str, apply_gamma: bool = True) -> None:
    """Save a linear RGB float32 frame to TIFF.
    
    Args:
        frame_data: Linear RGB float32 array
        path: Output file path
        apply_gamma: If True, apply inverse gamma (1/2.2) before saving
    """
    if apply_gamma and frame_data.dtype == np.float32:
        frame_data = apply_inverse_gamma(frame_data)
        print("Applied inverse gamma (1/2.2) correction")
    
    # Try tifffile first - it handles float32 and 16-bit RGB natively
    try:
        import tifffile
        if frame_data.dtype == np.float32:
            # Save as float32 TIFF (preserves full precision)
            tifffile.imwrite(path, frame_data, photometric='rgb')
            print(f"Saved float32 TIFF to: {os.path.abspath(path)}")
        else:
            tifffile.imwrite(path, frame_data, photometric='rgb')
            print(f"Saved TIFF to: {os.path.abspath(path)}")
        return
    except ImportError:
        pass


def main() -> int:
    args = parse_args()
    
    try:
        import xp_python
    except Exception as exc:
        print(f"Failed to import xp_python: {exc}", file=sys.stderr)
        return 1

    print("Successfully imported xp_python")

    if not hasattr(xp_python, "FrameExtractor"):
        print("Error: FrameExtractor class is missing", file=sys.stderr)
        return 1

    frame_label = args.start_frame if args.start_frame is not None else "first"
    output_path = args.output or f"frame_{frame_label}_linear.tiff"

    # Build extractor kwargs, only including optional params if specified
    extractor_kwargs = {
        "filename": args.filename,
        "pixel_format": xp_python.PixelFormat.RgbLinearF32,
    }
    if args.start_frame is not None:
        extractor_kwargs["start_frame"] = args.start_frame
    if args.duration is not None:
        extractor_kwargs["duration"] = args.duration

    # Extract a single frame with RgbLinearF32 pixel format
    extractor = xp_python.FrameExtractor(**extractor_kwargs)

    frame = extractor.extract()
    
    if frame is None:
        frame_desc = f"frame {args.start_frame}" if args.start_frame is not None else "first frame"
        print(f"Error: Failed to extract {frame_desc}", file=sys.stderr)
        return 2

    frame_number = frame["frame_number"]
    frame_data = frame.get("data")
    timecode = frame.get("timecode", "N/A")
    width = frame.get("width", 0)
    height = frame.get("height", 0)

    print(f"Extracted frame {frame_number} at {timecode}")
    print(f"Resolution: {width}x{height}")
    
    if frame_data is None:
        print("Error: Frame has no data", file=sys.stderr)
        return 3

    print(f"Data shape: {frame_data.shape}")
    print(f"Data dtype: {frame_data.dtype}")
    print(f"Value range: [{frame_data.min():.4f}, {frame_data.max():.4f}]")

    save_linear_tiff(frame_data, output_path)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
