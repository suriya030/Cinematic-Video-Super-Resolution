# xp-python

A Python extension for extracting and processing frames from DCI (Digital Cinema Initiative) MXF video files. Built with PyO3/maturin.

## Prerequisites

- Python 3.10 or later
- Rust toolchain (rustup)
- [maturin](https://github.com/PyO3/maturin) for building the extension
- vcpkg dependencies (Windows: set `VCPKG_ROOT` environment variable)

## Setup

### 1. Create a Python Virtual Environment

```bash
# From the repository root
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Windows CMD:
.venv\Scripts\activate.bat

# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install -r application/xp-python/requirements.txt
```

### 3. Install maturin

```bash
pip install maturin
```

## Building

### Development Build

For development, use `maturin develop` which builds and installs the extension directly into your virtual environment:

```bash
cd application/xp-python
maturin develop --release
```

The `--release` flag is recommended for performance, as debug builds are significantly slower.

### Building a Wheel

To create a distributable wheel file:

```bash
cd application/xp-python
maturin build --release --strip
```

The wheel will be created in `target/wheels/`. You can install it with:

```bash
pip install ../../target/wheels/xp_python-*.whl
```

#### Build Options

| Option | Description |
|--------|-------------|
| `--release` | Optimized release build |
| `--strip` | Strip debug symbols for smaller binary |
| `--interpreter python3.11 python3.12` | Build for multiple Python versions |

## Using the Scripts

### Scene Detection

Detect scene cuts in a video file using PySceneDetect:

```bash
cd application/xp-python
python scripts/scene_detect.py <path-to-mxf-file>

# With options:
python scripts/scene_detect.py video.mxf --start-frame 100 --duration 1000 --threshold 30.0
```

**Arguments:**

- `filename` - Path to the MXF video file (required)
- `--start-frame` - Starting frame number (optional, default is the first frame if the file)
- `--duration` - Number of frames to process (optional, defaults to entire file)
- `--threshold` - Scene detection sensitivity (default: 27.0, higher = less sensitive)
- `--output-dir` - Directory for scene cut frame images (default: `scene_cuts`)
- `--drop-levels` - JPEG2K resolution drop levels for faster processing (default: 2)

### Extract Linear RGB Frame

Extract a single frame as a high-quality linear RGB TIFF:

```bash
cd application/xp-python
python scripts/extract_linear_frame.py <path-to-mxf-file>

# With options:
python scripts/extract_linear_frame.py video.mxf --start-frame 1500 --output frame.tiff
```

**Arguments:**

- `filename` - Path to the MXF video file (required)
- `--start-frame` - Frame number to extract (optional)
- `--duration` - Number of frames to extract (optional)
- `--output`, `-o` - Output TIFF path (default: `frame_<n>_linear.tiff`)

## Using the Library

### Basic Usage

```python
import xp_python

# Create a frame extractor
extractor = xp_python.FrameExtractor(
    filename="path/to/video.mxf",
    start_frame=0,           # Optional: starting frame
    duration=100,            # Optional: number of frames to extract
    drop_levels=2,           # Optional: JPEG2K resolution reduction (0=full res)
    pixel_format=xp_python.PixelFormat.Xyz8Bit,  # Optional: output format
)

# Extract frames one by one
frame = extractor.extract()
if frame is not None:
    print(f"Frame {frame['frame_number']} at {frame['timecode']}")
    print(f"Resolution: {frame['width']}x{frame['height']}")
    print(f"FPS: {frame['fps']}")
    data = frame['data']  # numpy array
```

### Iterating Over Frames

```python
import xp_python

extractor = xp_python.FrameExtractor(filename="video.mxf", duration=100)

for frame in extractor:
    frame_number = frame['frame_number']
    timecode = frame['timecode']
    data = frame['data']  # numpy array (H, W, 3)
    # Process frame...
```

### Pixel Formats

The library supports three output pixel formats:

| Format | Type | Description |
|--------|------|-------------|
| `PixelFormat.Xyz8Bit` | `uint8` | XYZ color space, 8-bit, BGR order (OpenCV compatible) |
| `PixelFormat.Xyz16Bit` | `uint16` | XYZ color space, 16-bit, BGR order |
| `PixelFormat.RgbLinearF32` | `float32` | Linear RGB, range [0.0, 1.0] |

```python
import xp_python

# For scene detection or display (fast, OpenCV compatible)
extractor = xp_python.FrameExtractor(
    filename="video.mxf",
    pixel_format=xp_python.PixelFormat.Xyz8Bit,
)

# For high-quality color grading (linear light)
extractor = xp_python.FrameExtractor(
    filename="video.mxf",
    pixel_format=xp_python.PixelFormat.RgbLinearF32,
)
```

### Frame Dictionary Contents

Each frame returned by `extract()` or iteration contains:

| Key | Type | Description |
|-----|------|-------------|
| `frame_number` | `int` | Frame index |
| `timecode` | `str` | SMPTE timecode (e.g., "0:01:23.456") |
| `time_seconds` | `float` | Time in seconds |
| `fps` | `float` | Frame rate |
| `width` | `int` | Frame width in pixels |
| `height` | `int` | Frame height in pixels |
| `data` | `numpy.ndarray` | Frame data as (H, W, 3) array |

### Windows DLL Loading

On Windows, if you encounter DLL loading errors, add the vcpkg bin directory before importing:

```python
import os
import sys

if sys.platform == 'win32':
    vcpkg_root = os.environ.get('VCPKG_ROOT', r'C:\dev\vcpkg')
    dll_path = os.path.join(vcpkg_root, 'installed', 'x64-windows', 'bin')
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)

import xp_python  # Now safe to import
```

## Troubleshooting

### ImportError: DLL load failed

Ensure vcpkg DLLs are in your PATH or use `os.add_dll_directory()` as shown above.

### Slow performance

- Use `--release` builds (debug builds are 10-50x slower)
- Increase `drop_levels` for faster JPEG2K decoding (at lower resolution)
- Use `Xyz8Bit` pixel format unless you need higher precision

### Build errors

Ensure all native dependencies are available:

- Windows: Check `VCPKG_ROOT` is set and vcpkg dependencies are installed
- Linux: Install `libxml2`, `ffmpeg`, `alsa` development packages
