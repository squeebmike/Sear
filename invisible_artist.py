#!/usr/bin/env python3
"""
InvisibleArtist — hides hands and arms from overhead drawing videos.

Fills hand/arm regions from a clean canvas built exclusively from
hand-free moments, so the drawing appears to build itself.

Usage:
    python3 invisible_artist.py input.mkv output.mp4
    python3 invisible_artist.py input.mkv test.mp4 --max-frames 150 --debug
"""

import argparse
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


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def ensure_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
    if not os.path.exists(model_path):
        print("  Downloading hand detection model (~6 MB) — one-time...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("  Done.\n")
    return model_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Hide hands/arms from drawing videos using temporal compositing."
    )
    p.add_argument("input",  help="Input video (MP4, MKV, MOV…)")
    p.add_argument("output", help="Output MP4 path")
    p.add_argument("--debug",            action="store_true",
                   help="Save debug_masks.mp4 showing what gets masked")
    p.add_argument("--mask-dilate",      type=int,   default=40,
                   help="Pixels to expand hand mask (default: 40)")
    p.add_argument("--arm-length",       type=int,   default=600,
                   help="How far to extend the arm mask from the wrist (default: 600)")
    p.add_argument("--arm-thickness",    type=int,   default=120,
                   help="Thickness of arm mask line in pixels (default: 120)")
    p.add_argument("--feather",          type=int,   default=15,
                   help="Mask edge softness in pixels (default: 15)")
    p.add_argument("--confidence",       type=float, default=0.4,
                   help="Hand detection confidence 0.0-1.0 (default: 0.4)")
    p.add_argument("--max-frames",       type=int,   default=None,
                   help="Stop after N frames (for quick tests)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def make_hand_arm_mask(landmarks_list, h, w, dilate_px, arm_length, arm_thickness):
    """
    Build a mask covering:
      - The hand (convex hull of all landmarks)
      - The arm (thick line from wrist extending away from the hand)
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    for landmarks in landmarks_list:
        pts = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in landmarks],
            dtype=np.int32,
        )

        # --- Hand area: convex hull ---
        cv2.fillPoly(mask, [cv2.convexHull(pts)], 255)

        # --- Arm area: ray from wrist away from palm center ---
        wrist  = pts[0].astype(np.float32)           # landmark 0 = wrist
        center = pts.mean(axis=0).astype(np.float32) # approximate palm center
        arm_dir = wrist - center
        norm = np.linalg.norm(arm_dir)
        if norm > 1e-3:
            arm_dir /= norm
        arm_end = (wrist + arm_dir * arm_length).astype(np.int32)
        cv2.line(mask, tuple(pts[0]), tuple(arm_end), 255, arm_thickness)

    # Dilate to soften and catch finger edges
    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, k)

    return mask


def feather_mask(mask, feather_px):
    """Return float32 alpha [0..1] with soft edges."""
    if feather_px <= 0:
        return mask.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(feather_px))
    return np.clip(blurred / 255.0, 0.0, 1.0)


def blend(frame, canvas, alpha):
    """alpha=1 → canvas (clean), alpha=0 → original frame."""
    a = alpha[:, :, np.newaxis]
    return (frame.astype(np.float32) * (1.0 - a) +
            canvas * a).astype(np.uint8)


# ---------------------------------------------------------------------------
# Canvas bootstrap: sample only hand-free frames
# ---------------------------------------------------------------------------

def build_initial_canvas(video_path, detector, h, w, total, scan_every=4):
    """
    Scan the first portion of the video.
    Collect frames where NO hands are detected.
    Return median of those clean frames → true hand-free background.
    Falls back to brightest-pixel composite if no clean frames found.
    """
    cap = cv2.VideoCapture(video_path)
    clean_frames = []
    all_frames   = []
    limit        = min(total, 300)  # scan up to 300 frames

    for idx in range(limit):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % scan_every != 0:
            continue

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        frame_f = frame.astype(np.float32)
        all_frames.append(frame_f)

        if not result.hand_landmarks:
            clean_frames.append(frame_f)

    cap.release()

    if clean_frames:
        print(f"  Found {len(clean_frames)} hand-free frames for background — using median.")
        return np.median(clean_frames, axis=0)

    # Fallback: no clean frames found; use brightest pixel at each location
    # (white paper tends to be the brightest element)
    print("  No fully hand-free frames found in first 300 — using brightest-pixel fallback.")
    return np.max(all_frames, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_video(args):
    model_path = ensure_model()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Cannot open '{args.input}'")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if args.max_frames:
        total = min(total, args.max_frames)

    print(f"\n  Input       : {args.input}")
    print(f"  Output      : {args.output}")
    print(f"  Resolution  : {width}x{height}  |  FPS: {fps:.2f}  |  Frames: {total}")
    print(f"  Dilate      : {args.mask_dilate}px")
    print(f"  Arm length  : {args.arm_length}px  thickness: {args.arm_thickness}px")
    print(f"  Feather     : {args.feather}px")
    print(f"  Confidence  : {args.confidence}\n")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=2,
        min_hand_detection_confidence=args.confidence,
        min_hand_presence_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )

    # --- Build clean background plate ---
    print("  Building background plate from hand-free frames...")
    with HandLandmarker.create_from_options(options) as detector:
        canvas = build_initial_canvas(
            args.input, detector, height, width, total
        )
    print("  Background ready.\n")

    # --- Output writers ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    debug_writer = None
    if args.debug:
        debug_path = str(Path(args.output).parent / "debug_masks.mp4")
        debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (width, height))

    # --- Main pass ---
    print("  Processing frames...")
    high_coverage = 0
    t0 = time.time()
    cap = cv2.VideoCapture(args.input)

    with HandLandmarker.create_from_options(options) as detector:
        for idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            frame_f = frame.astype(np.float32)

            # Detect hands
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            if result.hand_landmarks:
                mask = make_hand_arm_mask(
                    result.hand_landmarks, height, width,
                    args.mask_dilate, args.arm_length, args.arm_thickness,
                )
            else:
                mask = np.zeros((height, width), dtype=np.uint8)

            # Coverage check
            if mask.mean() / 255.0 > 0.6:
                high_coverage += 1

            # Composite: fill masked pixels from canvas
            if mask.max() == 0:
                output_frame = frame
            else:
                alpha        = feather_mask(mask, args.feather)
                output_frame = blend(frame, canvas, alpha)

            out.write(output_frame)

            # Update canvas ONLY where NO hand mask exists
            # This keeps the canvas strictly hand-free
            clean_px = (mask == 0)

            # Also absorb new ink: if a clean pixel became darker than the
            # canvas (new line drawn), add it so artwork accumulates correctly
            new_ink = (
                clean_px &
                (frame_f.mean(axis=2) < canvas.mean(axis=2) - 10)
            )
            canvas[clean_px | new_ink] = frame_f[clean_px | new_ink]

            # Debug: red overlay on masked area
            if args.debug and debug_writer:
                vis = frame.copy()
                vis[mask > 0] = [0, 0, 200]
                debug_writer.write(vis)

            # Progress
            if idx % 100 == 0 or idx == total - 1:
                elapsed = time.time() - t0
                rate    = (idx + 1) / elapsed if elapsed > 0 else 1
                eta     = (total - idx - 1) / rate
                pct     = (idx + 1) / total * 100
                print(f"  Frame {idx+1:>6}/{total}  ({pct:5.1f}%)  "
                      f"{rate:5.1f} fps  ETA {eta:.0f}s")

    cap.release()
    out.release()
    if debug_writer:
        debug_writer.release()

    if high_coverage > total * 0.3:
        print(f"\n  WARNING: {high_coverage}/{total} frames had >60% of image masked.")
        print("  Try reducing --mask-dilate or --arm-length.\n")

    if args.debug:
        summary = str(Path(args.output).parent / "debug_summary.txt")
        with open(summary, "w") as f:
            f.write(f"Input            : {args.input}\n")
            f.write(f"Output           : {args.output}\n")
            f.write(f"Total frames     : {total}\n")
            f.write(f"High-coverage    : {high_coverage}\n")
            f.write(f"mask-dilate      : {args.mask_dilate}\n")
            f.write(f"arm-length       : {args.arm_length}\n")
            f.write(f"arm-thickness    : {args.arm_thickness}\n")
            f.write(f"feather          : {args.feather}\n")
        print(f"  Debug summary : {summary}")
        print(f"  Debug masks   : debug_masks.mp4")

    print(f"\n{'='*54}")
    print(f"  DONE — {time.time()-t0:.0f}s")
    print(f"{'='*54}")
    print(f"  Output : {args.output}\n")


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("ERROR: Input and output must be different files.")
        sys.exit(1)
    process_video(args)


if __name__ == "__main__":
    main()
