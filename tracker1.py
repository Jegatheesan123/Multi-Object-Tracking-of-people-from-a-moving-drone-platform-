import argparse
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
CONF_THRESH     = 0.25
IOU_THRESH      = 0.45
TAIL_LENGTH     = 30
SLICE_H         = 640
SLICE_W         = 640
SLICE_OVERLAP   = 0.2
USE_SAHI        = True

_PALETTE = [
    (0, 200, 255), (0, 255, 128), (255, 128, 0),
    (200, 0, 255), (255, 0, 100), (0, 255, 200),
]


def id_color(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


def sahi_detect(model, frame, slice_h=SLICE_H, slice_w=SLICE_W,
                overlap=SLICE_OVERLAP, conf=CONF_THRESH, iou=IOU_THRESH):
    H, W = frame.shape[:2]
    stride_h = int(slice_h * (1 - overlap))
    stride_w = int(slice_w * (1 - overlap))
    raw_boxes, raw_scores = [], []

    for y0 in range(0, H, stride_h):
        for x0 in range(0, W, stride_w):
            x1 = min(x0 + slice_w, W)
            y1 = min(y0 + slice_h, H)
            tile = frame[y0:y1, x0:x1]
            results = model.predict(tile, classes=[PERSON_CLASS_ID],
                                    conf=conf, iou=iou,
                                    verbose=False, stream=False)
            for r in results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    raw_boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
                    raw_scores.append(float(box.conf[0]))

    if not raw_boxes:
        return np.empty((0, 4)), np.empty(0)

    boxes_arr  = np.array(raw_boxes,  dtype=np.float32)
    scores_arr = np.array(raw_scores, dtype=np.float32)
    xywh = boxes_arr.copy()
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores_arr.tolist(), conf, iou)

    if len(keep) == 0:
        return np.empty((0, 4)), np.empty(0)

    keep = keep.flatten()
    return boxes_arr[keep], scores_arr[keep]


class EgoMotionCompensator:
    def __init__(self):
        self.prev_gray   = None
        self.prev_pts    = None
        self.lk_params   = dict(winSize=(15, 15), maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS |
                                          cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feat_params = dict(maxCorners=200, qualityLevel=0.01,
                                minDistance=7, blockSize=7)

    def estimate(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts  = cv2.goodFeaturesToTrack(gray, mask=None, **self.feat_params)
            return 0.0, 0.0

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feat_params)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        if curr_pts is None or status is None:
            self.prev_gray = gray
            return 0.0, 0.0

        good_old = self.prev_pts[status == 1]
        good_new = curr_pts[status == 1]

        if len(good_new) < 5:
            self.prev_gray = gray
            return 0.0, 0.0

        shift = (good_new - good_old).mean(axis=0)
        self.prev_gray = gray
        self.prev_pts  = good_new.reshape(-1, 1, 2)
        return float(shift[0]), float(shift[1])


def draw_tail(frame, history, color):
    n = len(history)
    for i in range(1, n):
        alpha = i / n
        thickness = max(1, int(alpha * 3))
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, history[i - 1], history[i], c, thickness)


def draw_box(frame, x1, y1, x2, y2, track_id, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, fps, n_tracks, hw_label):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"FPS : {fps:5.1f}",      (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1)
    cv2.putText(frame, f"Persons : {n_tracks}",  (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    cv2.putText(frame, hw_label,                 (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def run(source, output, model_path, use_sahi, hw_label):
    model = YOLO(model_path)

    source_path = Path(source)
    image_files = []
    if source_path.is_dir():
        image_files = sorted(source_path.glob("*.jpg"))
        if not image_files:
            image_files = sorted(source_path.glob("*.png"))
        if not image_files:
            raise FileNotFoundError(f"No jpg/png images found in: {source}")
        first   = cv2.imread(str(image_files[0]))
        H, W    = first.shape[:2]
        fps_src = 25.0
        cap     = None
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open source: {source}")
        W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (W, H))

    compensator = EgoMotionCompensator()
    histories   = defaultdict(list)
    frame_times = []
    frame_idx   = 0

    print(f"[Aerial Guardian] Source: {source}")
    print(f"[Aerial Guardian] Resolution: {W}x{H}  |  SAHI: {use_sahi}")

    def frame_iter():
        if image_files:
            for img_path in image_files:
                fr = cv2.imread(str(img_path))
                if fr is not None:
                    yield fr
        else:
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                yield fr

    for frame in frame_iter():
        t0 = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy = compensator.estimate(gray)

        results = model.track(frame, classes=[PERSON_CLASS_ID],
                              conf=CONF_THRESH, iou=IOU_THRESH,
                              tracker="bytetrack.yaml",
                              persist=True, verbose=False)

        n_tracks = 0
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                tid            = int(box.id[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                cx, cy         = (x1 + x2) // 2, (y1 + y2) // 2

                histories[tid].append((cx, cy))
                if len(histories[tid]) > TAIL_LENGTH:
                    histories[tid].pop(0)

                color = id_color(tid)
                draw_tail(frame, histories[tid], color)
                draw_box(frame, x1, y1, x2, y2, tid, color)
                n_tracks += 1

        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        if len(frame_times) > 60:
            frame_times.pop(0)
        fps_display = 1.0 / (sum(frame_times) / len(frame_times))

        draw_hud(frame, fps_display, n_tracks, hw_label)
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx:5d}  |  FPS {fps_display:5.1f}  |  Active tracks: {n_tracks}")

    if cap:
        cap.release()
    writer.release()
    avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0
    print(f"\n[Done]  {frame_idx} frames processed.")
    print(f"[Perf]  Average FPS: {avg_fps:.1f}  |  Hardware: {hw_label}")
    print(f"[Out]   Saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial Guardian - Drone MOT")
    parser.add_argument("--source",  required=True)
    parser.add_argument("--output",  default="output.mp4")
    parser.add_argument("--model",   default="yolov8n.pt")
    parser.add_argument("--no-sahi", action="store_true")
    parser.add_argument("--hw",      default="CPU")
    args = parser.parse_args()

    run(
        source     = args.source,
        output     = args.output,
        model_path = args.model,
        use_sahi   = not args.no_sahi,
        hw_label   = args.hw,
    )