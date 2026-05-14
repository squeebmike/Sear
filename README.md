# HandFrameRemover

Removes all frames containing visible hands from overhead drawing videos and exports a clean MP4 with only hand-free frames.

---

## What It Does

1. Reads your drawing video frame by frame
2. Uses Google MediaPipe to detect hands in each frame
3. Removes every frame where a hand is visible
4. Adds a configurable buffer of extra frames around each hand moment (so no partial-hand edges sneak through)
5. Exports a clean MP4 at the original resolution and frame rate
6. Saves a CSV report showing what happened to every frame

---

## Requirements

- **macOS** (also works on Linux/Windows)
- **Python 3.9+**
- **Homebrew** (for ffmpeg — needed by OpenCV under the hood)
- **pip** packages listed in `requirements.txt`

---

## Setup (One Time)

### 1. Install Homebrew (if you don't have it)

Open Terminal and paste:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install ffmpeg

```bash
brew install ffmpeg
```

### 3. Install Python packages

Navigate to the folder containing this project, then run:

```bash
pip3 install -r requirements.txt
```

---

## Usage

```bash
python3 remove_hand_frames.py input.mp4 output_clean.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--buffer N` | `5` | Remove N extra frames before and after each hand frame |
| `--confidence X` | `0.5` | Hand detection sensitivity (0.0 = loose, 1.0 = strict) |

### Examples

```bash
# Basic — remove hand frames with default 5-frame buffer
python3 remove_hand_frames.py drawing.mp4 drawing_clean.mp4

# Larger buffer for smoother transitions (removes more frames)
python3 remove_hand_frames.py drawing.mp4 drawing_clean.mp4 --buffer 10

# Stricter detection (fewer false positives, may miss some hands)
python3 remove_hand_frames.py drawing.mp4 drawing_clean.mp4 --confidence 0.7
```

---

## Output Files

| File | Description |
|------|-------------|
| `output_clean.mp4` | The cleaned video — only frames with no hands |
| `output_clean.csv` | Frame-by-frame report (see columns below) |

### CSV Columns

| Column | Description |
|--------|-------------|
| `frame_number` | Index of the frame (starts at 0) |
| `timestamp_seconds` | Time position in the original video |
| `hand_detected` | `True` if MediaPipe found a hand |
| `removed_by_buffer` | `True` if removed only because it was near a hand frame |
| `kept` | `True` if this frame is in the output video |

### Terminal Summary Example

```
====================================================
  SUMMARY
====================================================
  Total frames           : 3600
  Clean frames kept      : 2980
  Hand frames removed    : 412
  Buffer frames removed  : 208
  Percent kept           : 82.8%
  Original duration      : 120.00s
  Final duration         : 99.33s
====================================================
```

---

## Troubleshooting

**"Cannot open video file"** — Double-check the path to your input file. Drag the file into Terminal to get its exact path.

**Output is choppy** — Try increasing `--buffer` (e.g. `--buffer 10`) so the transitions between hand-in/hand-out sections are smoother.

**Too many frames removed** — Lower `--confidence` (e.g. `--confidence 0.3`) to make detection less aggressive.

**Too few hands detected** — Raise `--confidence` closer to `0.7` or `0.8`, or make sure your video has good lighting.

**MediaPipe install fails** — Make sure you're using Python 3.9–3.12. MediaPipe does not yet support Python 3.13+.

---

## How It Works (Simple Version)

The script runs two passes over your video:

- **Pass 1** — reads every frame and asks MediaPipe "is there a hand here?" It logs the answer for every frame.
- **Pass 2** — reads the video again and writes only the clean frames to a new file.

Between the two passes it expands each detected hand frame by the buffer amount so you don't get sudden jump cuts where a hand was just leaving the frame.
