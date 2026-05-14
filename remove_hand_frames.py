#!/usr/bin/env python3
"""
HandFrameRemover — strips every frame containing visible hands from an overhead
drawing video and exports a clean MP4 with only hand-free frames.

Usage:
    python3 remove_hand_frames.py input.mp4 output_clean.mp4 [--buffer 5]
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image, ImageFormat

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"  Downloading hand detection model (~6 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("  Model downloaded.\n")
    return model_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove frames containing hands from a drawing video."
    )
    parser.add_argument("input", help="Path to the input video")
    parser.add_argument("output", help="Path for the cleaned output MP4")
    parser.add_argument(
        "--buffer",
        type=int,
        default=5,
        help="Extra frames to remove before and after each hand frame (default: 5)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum hand detection confidence 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional custom path for the CSV report",
    )
    return parser.parse_args()


def build_removal_set(hand_frames: set, total: int, buffer: int) -> set:
    removal = set()
    for f in hand_frames:
        for i in range(max(0, f - buffer), min(total - 1, f + buffer) + 1):
            removal.add(i)
    return removal


def process_video(input_path: str, output_path: str, buffer: int, confidence: float, csv_path_override):
    model_path = ensure_model()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  FPS    : {fps:.2f}")
    print(f"  Size   : {width}x{height}")
    print(f"  Frames : {total_frames}")
    print(f"  Buffer : +/-{buffer} frames")
    print(f"  Confidence: {confidence}\n")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=2,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )

    frame_records = []
    hand_frame_indices = set()

    print("Pass 1/2 — Scanning for hands...")
    start_time = time.time()

    with HandLandmarker.create_from_options(options) as detector:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            has_hand = len(result.hand_landmarks) > 0

            timestamp = idx / fps if fps > 0 else 0.0
            frame_records.append({
                "frame_number": idx,
                "timestamp_seconds": round(timestamp, 4),
                "hand_detected": has_hand,
                "removed_by_buffer": False,
                "kept": None,
            })

            if has_hand:
                hand_frame_indices.add(idx)

            idx += 1
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                pct = idx / total_frames * 100 if total_frames else 0
                print(f"  Scanned {idx}/{total_frames} frames ({pct:.1f}%) — {elapsed:.1f}s")

    cap.release()

    removal_set = build_removal_set(hand_frame_indices, total_frames, buffer)
    buffer_only = removal_set - hand_frame_indices

    for rec in frame_records:
        f = rec["frame_number"]
        rec["removed_by_buffer"] = f in buffer_only
        rec["kept"] = f not in removal_set

    # --- Pass 2: write clean frames ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap2 = cv2.VideoCapture(input_path)
    kept_count = 0

    print("\nPass 2/2 — Writing clean frames...")
    idx = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        if idx not in removal_set:
            out.write(frame)
            kept_count += 1
        idx += 1
        if idx % 100 == 0:
            pct = idx / total_frames * 100 if total_frames else 0
            print(f"  Written {idx}/{total_frames} frames ({pct:.1f}%)")

    cap2.release()
    out.release()

    # --- CSV report ---
    csv_path = Path(csv_path_override) if csv_path_override else Path(output_path).with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_number", "timestamp_seconds", "hand_detected",
            "removed_by_buffer", "kept",
        ])
        writer.writeheader()
        writer.writerows(frame_records)

    hand_removed = len(hand_frame_indices)
    buffer_removed = len(buffer_only)
    pct_kept = kept_count / total_frames * 100 if total_frames else 0
    orig_duration = total_frames / fps if fps > 0 else 0
    final_duration = kept_count / fps if fps > 0 else 0

    print("\n" + "=" * 52)
    print("  SUMMARY")
    print("=" * 52)
    print(f"  Total frames           : {total_frames}")
    print(f"  Clean frames kept      : {kept_count}")
    print(f"  Hand frames removed    : {hand_removed}")
    print(f"  Buffer frames removed  : {buffer_removed}")
    print(f"  Percent kept           : {pct_kept:.1f}%")
    print(f"  Original duration      : {orig_duration:.2f}s")
    print(f"  Final duration         : {final_duration:.2f}s")
    print("=" * 52)
    print(f"\n  Output video : {output_path}")
    print(f"  CSV report   : {csv_path}\n")


def main():
    args = parse_args()
    process_video(args.input, args.output, args.buffer, args.confidence, args.csv)


if __name__ == "__main__":
    main()
