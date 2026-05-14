#!/usr/bin/env python3
"""
InvisibleArtist — hides hands and arms from overhead drawing videos.

Instead of deleting frames, it fills hand/arm regions using a rolling
clean canvas built from nearby hand-free frames, so the drawing appears
to build itself while hands stay invisible.

Usage:
    python3 invisible_artist.py input.mp4 output_invisible_artist.mp4
    python3 invisible_artist.py input.mkv out.mp4 --debug --max-frames 150
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
# Setup helpers
# ---------------------------------------------------------------------------

def ensure_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
    if not os.path.exists(model_path):
        print("  Downloading hand detection model (~6 MB) — one-time download...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("  Model saved.\n")
    return model_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Hide hands/arms from drawing videos using temporal compositing."
    )
    p.add_argument("input", help="Input video (MP4, MKV, MOV, …)")
    p.add_argument("output", help="Output MP4 path")
    p.add_argument("--debug", action="store_true",
                   help="Save debug_masks.mp4 and debug_summary.txt")
    p.add_argument("--mask-dilate", type=int, default=25,
                   help="Pixels to expand hand/arm mask (default: 25)")
    p.add_argument("--motion-threshold", type=int, default=25,
                   help="Motion sensitivity 0-255 (default: 25; lower = more sensitive)")
    p.add_argument("--feather", type=int, default=9,
                   help="Mask edge softness in pixels (default: 9)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N frames — useful for quick tests")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def make_hand_mask(landmarks_list, h, w, dilate_px):
    """Convex hull around each detected hand, then dilate."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for landmarks in landmarks_list:
        pts = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in landmarks],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [cv2.convexHull(pts)], 255)
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
    sigma = max(feather_px, 1)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return np.clip(blurred / 255.0, 0.0, 1.0)


def composite(frame, canvas, alpha):
    """alpha=1 → canvas pixel, alpha=0 → original frame pixel."""
    a = alpha[:, :, np.newaxis]
    return (frame.astype(np.float32) * (1.0 - a) +
            canvas * a).astype(np.uint8)


# ---------------------------------------------------------------------------
# Background bootstrap
# ---------------------------------------------------------------------------

def sample_background(video_path, n_frames):
    """
    Read up to n_frames from the start of the video and return a median image.
    Median across time removes intermittent hands to give a clean paper estimate.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n_frames):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f.astype(np.float32))
    cap.release()
    if not frames:
        return None
    return np.median(frames, axis=0)  # float32


# ---------------------------------------------------------------------------
# Main processing loop
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

    print(f"\n  Input        : {args.input}")
    print(f"  Output       : {args.output}")
    print(f"  Resolution   : {width}x{height}  |  FPS: {fps:.2f}  |  Frames: {total}")
    print(f"  Mask dilate  : {args.mask_dilate}px")
    print(f"  Motion thr   : {args.motion_threshold}")
    print(f"  Feather      : {args.feather}px")
    print(f"  Debug        : {args.debug}\n")

    # --- Build initial canvas from first ~60 frames (median = hand-free paper) ---
    sample_n = min(60, total)
    print(f"  Sampling {sample_n} frames to build background plate...")
    canvas = sample_background(args.input, sample_n)
    if canvas is None:
        print("ERROR: Could not read any frames.")
        sys.exit(1)
    print("  Background ready.\n")

    # --- Morphology kernels ---
    dilate_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.mask_dilate * 2 + 1, args.mask_dilate * 2 + 1)
    )
    noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # --- MediaPipe hand detector ---
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    # --- Output video writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    debug_writer = None
    if args.debug:
        dmask_path = str(Path(args.output).parent / "debug_masks.mp4")
        debug_writer = cv2.VideoWriter(dmask_path, fourcc, fps, (width, height))

    # --- Processing ---
    high_coverage_count = 0
    t0 = time.time()

    cap = cv2.VideoCapture(args.input)

    with HandLandmarker.create_from_options(options) as detector:
        for idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            frame_f = frame.astype(np.float32)

            # 1. Hand mask via MediaPipe landmarks
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            if result.hand_landmarks:
                hand_mask = make_hand_mask(
                    result.hand_landmarks, height, width, args.mask_dilate
                )
            else:
                hand_mask = np.zeros((height, width), dtype=np.uint8)

            # 2. Motion mask — pixels that differ significantly from canvas
            diff = np.abs(frame_f - canvas).max(axis=2).astype(np.uint8)
            _, motion_raw = cv2.threshold(
                diff, args.motion_threshold, 255, cv2.THRESH_BINARY
            )
            # Remove salt-and-pepper noise, then expand to catch arms
            motion_mask = cv2.morphologyEx(motion_raw, cv2.MORPH_OPEN, noise_k)
            motion_mask = cv2.dilate(motion_mask, dilate_k)

            # 3. Combined mask = hand OR moving object
            combined = cv2.bitwise_or(hand_mask, motion_mask)

            # 4. Coverage warning tracking
            if combined.mean() / 255.0 > 0.6:
                high_coverage_count += 1

            # 5. Composite: fill masked pixels from canvas
            if combined.max() == 0:
                output_frame = frame
            else:
                alpha        = feather_mask(combined, args.feather)
                output_frame = composite(frame, canvas, alpha)

            out.write(output_frame)

            # 6. Update canvas
            #    Rule A: pixels not in any mask → update freely (clean view)
            #    Rule B: pixels darker than canvas AND hand is not directly on them
            #            → likely new ink; update so we preserve it going forward
            clean_px    = (combined == 0)
            new_ink_px  = (
                (hand_mask == 0) &
                (frame_f.mean(axis=2) < canvas.mean(axis=2) - 8)
            )
            update_px   = clean_px | new_ink_px
            canvas[update_px] = frame_f[update_px]

            # 7. Debug mask overlay (red = masked)
            if args.debug and debug_writer:
                vis = frame.copy()
                vis[combined > 0] = [0, 0, 200]
                debug_writer.write(vis)

            # 8. Progress
            if idx % 100 == 0 or idx == total - 1:
                elapsed  = time.time() - t0
                rate     = (idx + 1) / elapsed if elapsed > 0 else 1
                eta      = (total - idx - 1) / rate
                pct      = (idx + 1) / total * 100
                print(f"  Frame {idx+1:>6}/{total}  ({pct:5.1f}%)  "
                      f"{rate:5.1f} fps  ETA {eta:5.0f}s")

    cap.release()
    out.release()
    if debug_writer:
        debug_writer.release()

    # --- Warnings ---
    if high_coverage_count > total * 0.3:
        print(f"\n  WARNING: {high_coverage_count}/{total} frames had >60% of the image masked.")
        print("  The output may show ghosting or blurry fills.")
        print("  Try: --mask-dilate 15  or pull your hand away from the art more often.\n")

    # --- Debug summary ---
    if args.debug:
        summary_path = str(Path(args.output).parent / "debug_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Input              : {args.input}\n")
            f.write(f"Output             : {args.output}\n")
            f.write(f"Total frames       : {total}\n")
            f.write(f"High-coverage      : {high_coverage_count}\n")
            f.write(f"mask-dilate        : {args.mask_dilate}\n")
            f.write(f"motion-threshold   : {args.motion_threshold}\n")
            f.write(f"feather            : {args.feather}\n")
        print(f"  Debug summary    : {summary_path}")
        print(f"  Debug mask video : debug_masks.mp4")

    elapsed_total = time.time() - t0
    print(f"\n{'='*54}")
    print(f"  DONE  —  {elapsed_total:.0f}s total")
    print(f"{'='*54}")
    print(f"  Output : {args.output}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("ERROR: Input and output must be different files.")
        sys.exit(1)

    process_video(args)


if __name__ == "__main__":
    main()
