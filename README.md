# Aerial Guardian – Drone Person Tracker

Detects and tracks people in drone footage using YOLOv8n + ByteTrack.

## Requirements

```bash
pip install ultralytics opencv-python numpy
```

## Usage

```bash
python tracker.py --source path/to/sequence_folder --output result.mp4 --hw "CPU"
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | required | Path to image folder or video file |
| `--output` | output.mp4 | Output video path |
| `--model` | yolov8n.pt | Path to YOLO weights |
| `--no-sahi` | off | Disable sliced inference (faster) |
| `--hw` | CPU | Hardware label shown on HUD |

## Example

```bash
python tracker.py --source VisDrone2019-MOT-val\sequences\uav0000086_00000_v --output result.mp4 --hw "CPU"
```

## Output

The output video shows:
- Bounding box per detected person
- Unique colour-coded ID label per track
- Fading trajectory tail (last 30 frames)
- HUD overlay with FPS, person count, and hardware label

## Results on VisDrone Val Set

| Metric | Value |
|---|---|
| Average FPS | 18.5 |
| Hardware | CPU  |
| Frames processed | 464 |
| Active tracks per frame | 12–21 |

## Architecture

- **Detector**: YOLOv8n (~6 MB) — well within 300 MB limit
- **Tracker**: ByteTrack via Ultralytics
- **Ego-motion compensation**: Lucas-Kanade optical flow to reduce ID switches caused by drone movement
- **SAHI**: Frame sliced into overlapping 640×640 tiles to detect small persons at altitude
