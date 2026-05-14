#!/usr/bin/env python3
"""
InvisibleArtist Pro — two-pass self-drawing video engine.

Pass 1  builds the final canvas: a complete clean-plate of the drawing
        accumulated across the entire video (only from hand-free pixels).

Pass 2  renders output: each masked pixel is filled from the final canvas,
        giving a look-ahead effect — the fill shows what that spot will look
        like after the hand moves away, not just what it looked like before.

Usage:
    python3 invisible_artist_pro.py input.mkv output.mp4 --debug
    python3 invisible_artist_pro.py input.mkv out.mp4 --max-frames 300 --debug \\
        --mask-dilate 55 --arm-length 500 --arm-thickness 220 \\
        --enable-sleeve-mask --enable-marker-mask
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


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def ensure_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
    if not os.path.exists(path):
        print("  Downloading hand model (~6 MB) — one-time...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("  Saved.\n")
    return path


def parse_args():
    p = argparse.ArgumentParser(
        description="InvisibleArtist Pro: two-pass self-drawing video engine."
    )
    p.add_argument("input")
    p.add_argument("output")

    # Run control
    p.add_argument("--debug",              action="store_true",
                   help="Save debug_mask.mp4, debug_canvas.mp4, debug_summary.txt")
    p.add_argument("--max-frames",         type=int,   default=None,
                   help="Stop after N frames (quick tests)")

    # Detection
    p.add_argument("--confidence",         type=float, default=0.45,
                   help="MediaPipe detection confidence (default: 0.45)")

    # Masking geometry
    p.add_argument("--mask-dilate",        type=int,   default=45,
                   help="Dilation on hand hull (default: 45)")
    p.add_argument("--arm-length",         type=int,   default=450,
                   help="Arm ray length in pixels from wrist (default: 450)")
    p.add_argument("--arm-thickness",      type=int,   default=190,
                   help="Arm ray width in pixels (default: 190)")
    p.add_argument("--feather",            type=int,   default=15,
                   help="Mask edge feather in pixels (default: 15)")
    p.add_argument("--mask-memory",        type=int,   default=5,
                   help="Frames to keep mask active after hand disappears — prevents flicker (default: 5)")

    # Optional mask extensions
    p.add_argument("--enable-sleeve-mask", action="store_true",
                   help="Expand mask to dark pixels connected to arm region")
    p.add_argument("--enable-marker-mask", action="store_true",
                   help="Mask marker/pen body near index finger")

    # Canvas / artwork
    p.add_argument("--stable-frames",      type=int,   default=2,
                   help="Clean frames required before committing new artwork (default: 2)")
    p.add_argument("--art-threshold",      type=int,   default=18,
                   help="Min pixel change to count as new artwork (default: 18)")
    p.add_argument("--strict-canvas",      action="store_true",
                   help="Never update canvas from any masked pixel")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Mask builders
# ─────────────────────────────────────────────────────────────────────────────

def build_hand_arm_mask(landmarks_list, h, w,
                        dilate_px, arm_length, arm_thickness,
                        frame_bgr, enable_sleeve, enable_marker):
    """
    Layers:
      1. Convex hull of all hand landmarks
      2. Arm ray from wrist away from palm
      3. (optional) Marker body from wrist toward index tip
      4. (optional) Dark sleeve pixels connected to arm region
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    for lm_list in landmarks_list:
        pts = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in lm_list],
            dtype=np.int32,
        )

        # 1. Hand hull
        cv2.fillPoly(mask, [cv2.convexHull(pts)], 255)

        # 2. Arm ray
        wrist  = pts[0].astype(np.float64)
        center = pts.mean(axis=0).astype(np.float64)
        arm_v  = wrist - center
        arm_n  = np.linalg.norm(arm_v)
        if arm_n > 1e-3:
            arm_v /= arm_n
        arm_end = np.clip(
            (wrist + arm_v * arm_length).astype(np.int32),
            [0, 0], [w - 1, h - 1],
        )
        cv2.line(mask, tuple(pts[0]), tuple(arm_end), 255, arm_thickness)

        # 3. Marker body (wrist → index fingertip, thinner)
        if enable_marker and len(pts) > 8:
            tip = pts[8].astype(np.float64)
            m_v = tip - wrist
            m_n = np.linalg.norm(m_v)
            if m_n > 1e-3:
                m_v /= m_n
            m_end = np.clip(
                (wrist + m_v * arm_length * 0.55).astype(np.int32),
                [0, 0], [w - 1, h - 1],
            )
            thickness = max(12, arm_thickness // 7)
            cv2.line(mask, tuple(pts[0]), tuple(m_end), 255, thickness)

    # 4. Sleeve: dark pixels restricted to the arm ray zone only.
    #    We search for dark clothing pixels ONLY within a wide band along the
    #    arm direction — this prevents dark ink lines on the paper from being
    #    picked up as sleeve pixels.
    if enable_sleeve and mask.max() > 0 and frame_bgr is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dark = (gray < 110).astype(np.uint8) * 255

        # Build an arm-only search zone: just the arm ray, 2× thickness
        arm_zone = np.zeros((h, w), dtype=np.uint8)
        for lm_list in landmarks_list:
            lm_pts = np.array(
                [[int(lm.x * w), int(lm.y * h)] for lm in lm_list],
                dtype=np.int32,
            )
            lm_wrist  = lm_pts[0].astype(np.float64)
            lm_center = lm_pts.mean(axis=0).astype(np.float64)
            lm_v = lm_wrist - lm_center
            lm_n = np.linalg.norm(lm_v)
            if lm_n > 1e-3:
                lm_v /= lm_n
            lm_end = np.clip(
                (lm_wrist + lm_v * arm_length).astype(np.int32),
                [0, 0], [w - 1, h - 1],
            )
            cv2.line(arm_zone, tuple(lm_pts[0]), tuple(lm_end),
                     255, arm_thickness * 2)

        # Dark pixels that fall inside the arm zone = sleeve
        sleeve_px = cv2.bitwise_and(dark, arm_zone)
        # Small closing to fill gaps in fabric texture
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        sleeve_px = cv2.morphologyEx(sleeve_px, cv2.MORPH_CLOSE, close_k)
        mask = cv2.bitwise_or(mask, sleeve_px)

    # Final dilation to cover edges / finger gaps
    if dilate_px > 0:
        dk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, dk)

    return mask


def feather(mask, px):
    if px <= 0:
        return mask.astype(np.float32) / 255.0
    b = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(px))
    return np.clip(b / 255.0, 0.0, 1.0)


def blend(frame, canvas, alpha):
    a = alpha[:, :, np.newaxis]
    return (frame.astype(np.float32) * (1.0 - a) + canvas * a).astype(np.uint8)


def apply_mask_memory(current_mask, persistence, memory_frames):
    """
    Sticky mask: once a pixel is masked, keep it masked for `memory_frames`
    additional frames after the detector stops seeing a hand there.
    This eliminates flicker when the hand is stationary or detection drops briefly.

    Updates `persistence` in-place and returns the combined mask.
    """
    # Where hand is currently detected: reset counter to full memory
    persistence[current_mask > 0] = memory_frames
    # Where hand is gone: count down
    off = (current_mask == 0)
    persistence[off] = np.maximum(0, persistence[off] - 1)
    # Combined: current mask OR still within memory window
    combined = np.where((current_mask > 0) | (persistence > 0),
                        np.uint8(255), np.uint8(0))
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Background plate
# ─────────────────────────────────────────────────────────────────────────────

def build_background(video_path, detector, total, scan_every=4, limit=300):
    """Median of hand-free sampled frames — or brightest-pixel fallback."""
    cap = cv2.VideoCapture(video_path)
    clean_f, all_f = [], []
    cap_limit = min(total, limit)

    for idx in range(cap_limit):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % scan_every != 0:
            continue
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
        f32    = frame.astype(np.float32)
        all_f.append(f32)
        if not result.hand_landmarks:
            clean_f.append(f32)

    cap.release()

    if clean_f:
        print(f"  Background : median of {len(clean_f)} clean frames.")
        return np.median(clean_f, axis=0)
    if all_f:
        print("  Background : no clean frames — brightest-pixel fallback.")
        return np.max(all_f, axis=0)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 — build final canvas
# ─────────────────────────────────────────────────────────────────────────────

def build_final_canvas(video_path, detector, bg, total, args):
    """
    Process every frame forward. Accumulate pixels onto the canvas only when
    they are hand-free and have been stable for >= stable_frames observations.

    The result is the most complete clean view of the drawing across the whole
    video — used as the look-ahead fill source in Pass 2.
    """
    canvas       = bg.copy()            # float32 clean plate
    stable_count = np.zeros(bg.shape[:2], dtype=np.uint16)
    persistence  = np.zeros(bg.shape[:2], dtype=np.uint8)   # mask memory

    cap = cv2.VideoCapture(video_path)
    t0  = time.time()
    print("  Pass 1 — building final canvas...")

    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(Image(image_format=ImageFormat.SRGB, data=rgb))

        if result.hand_landmarks:
            raw_mask = build_hand_arm_mask(
                result.hand_landmarks,
                frame.shape[0], frame.shape[1],
                args.mask_dilate, args.arm_length, args.arm_thickness,
                frame if args.enable_sleeve_mask else None,
                args.enable_sleeve_mask, args.enable_marker_mask,
            )
        else:
            raw_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Apply sticky mask memory — prevents flicker on canvas stability
        mask = apply_mask_memory(raw_mask, persistence, args.mask_memory)

        frame_f  = frame.astype(np.float32)
        clean_px = (mask == 0)

        # Stability: increment clean pixels, reset masked ones
        stable_count[clean_px]  = np.minimum(stable_count[clean_px] + 1, 60000)
        stable_count[~clean_px] = 0

        # Commit pixel once stable
        commit_px = clean_px & (stable_count >= args.stable_frames)
        canvas[commit_px] = frame_f[commit_px]

        if idx % 200 == 0 or idx == total - 1:
            elapsed = max(time.time() - t0, 1e-3)
            rate    = (idx + 1) / elapsed
            eta     = (total - idx - 1) / rate
            print(f"    {idx+1:>6}/{total}  {(idx+1)/total*100:5.1f}%  "
                  f"{rate:5.1f} fps  ETA {eta:.0f}s")

    cap.release()
    print(f"  Pass 1 done — {time.time()-t0:.0f}s\n")
    return canvas   # float32


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2 — render
# ─────────────────────────────────────────────────────────────────────────────

def render(video_path, final_canvas, bg, detector, total, fps, width, height, args):
    """
    For each frame:
      - Unmasked pixels  → output directly from the live frame
      - Masked pixels    → fill from final_canvas (look-ahead clean state)
      - Feather the mask edges for smooth blends
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    dbg_mask   = None
    dbg_canvas = None
    if args.debug:
        base       = Path(args.output).parent
        dbg_mask   = cv2.VideoWriter(str(base / "debug_mask.mp4"),   fourcc, fps, (width, height))
        dbg_canvas = cv2.VideoWriter(str(base / "debug_canvas.mp4"), fourcc, fps, (width, height))

    cap          = cv2.VideoCapture(video_path)
    high_cov     = 0
    persistence  = np.zeros((height, width), dtype=np.uint8)  # mask memory
    t0           = time.time()
    print("  Pass 2 — rendering...")

    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(Image(image_format=ImageFormat.SRGB, data=rgb))

        if result.hand_landmarks:
            raw_mask = build_hand_arm_mask(
                result.hand_landmarks, height, width,
                args.mask_dilate, args.arm_length, args.arm_thickness,
                frame if args.enable_sleeve_mask else None,
                args.enable_sleeve_mask, args.enable_marker_mask,
            )
        else:
            raw_mask = np.zeros((height, width), dtype=np.uint8)

        # Sticky mask: prevents flicker when hand is still or detection wavers
        mask = apply_mask_memory(raw_mask, persistence, args.mask_memory)

        cov = mask.mean() / 255.0
        if cov > 0.6:
            high_cov += 1

        if mask.max() == 0:
            output_frame = frame
        else:
            alpha        = feather(mask, args.feather)
            output_frame = blend(frame, final_canvas, alpha)

        out.write(output_frame)

        # Debug outputs
        if args.debug:
            if dbg_mask:
                vis = frame.copy()
                vis[mask > 0] = [0, 0, 200]   # show the full sticky mask
                dbg_mask.write(vis)
            if dbg_canvas:
                dbg_canvas.write(final_canvas.astype(np.uint8))

        if idx % 200 == 0 or idx == total - 1:
            elapsed = max(time.time() - t0, 1e-3)
            rate    = (idx + 1) / elapsed
            eta     = (total - idx - 1) / rate
            print(f"    {idx+1:>6}/{total}  {(idx+1)/total*100:5.1f}%  "
                  f"{rate:5.1f} fps  ETA {eta:.0f}s")

    cap.release()
    out.release()
    if dbg_mask:   dbg_mask.release()
    if dbg_canvas: dbg_canvas.release()

    print(f"  Pass 2 done — {time.time()-t0:.0f}s\n")
    return high_cov


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("ERROR: Input and output must be different files.")
        sys.exit(1)

    model_path = ensure_model()

    # Read video metadata
    _cap   = cv2.VideoCapture(args.input)
    fps    = _cap.get(cv2.CAP_PROP_FPS)
    width  = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()

    if args.max_frames:
        total = min(total, args.max_frames)

    print(f"\n{'='*58}")
    print(f"  InvisibleArtist Pro")
    print(f"{'='*58}")
    print(f"  Input         : {args.input}")
    print(f"  Output        : {args.output}")
    print(f"  Resolution    : {width}x{height}  FPS: {fps:.2f}  Frames: {total}")
    print(f"  Dilate/Feather: {args.mask_dilate}px / {args.feather}px  Memory: {args.mask_memory} frames")
    print(f"  Arm           : {args.arm_length}px long  {args.arm_thickness}px wide")
    print(f"  Stable frames : {args.stable_frames}  Art threshold: {args.art_threshold}")
    print(f"  Sleeve mask   : {args.enable_sleeve_mask}")
    print(f"  Marker mask   : {args.enable_marker_mask}")
    print(f"  Debug         : {args.debug}")
    print(f"{'='*58}\n")

    detector_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=2,
        min_hand_detection_confidence=args.confidence,
        min_hand_presence_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )

    # Step 1: background plate
    print("  Step 1 — Background plate")
    print("  " + "-" * 38)
    with HandLandmarker.create_from_options(detector_options) as det:
        bg = build_background(args.input, det, total)
    if bg is None:
        print("ERROR: Could not read video frames.")
        sys.exit(1)
    print()

    # Step 2: Pass 1 — final canvas
    print("  Step 2 — Pass 1: build final canvas")
    print("  " + "-" * 38)
    with HandLandmarker.create_from_options(detector_options) as det:
        final_canvas = build_final_canvas(args.input, det, bg, total, args)

    # Step 3: Pass 2 — render
    print("  Step 3 — Pass 2: render output")
    print("  " + "-" * 38)
    with HandLandmarker.create_from_options(detector_options) as det:
        high_cov = render(
            args.input, final_canvas, bg, det, total,
            fps, width, height, args
        )

    # Warnings
    if high_cov > total * 0.3:
        print(f"  WARNING: {high_cov}/{total} frames had >60% mask coverage.")
        print("  The result may show ghosting. Pull hand away more often,")
        print("  or try: --mask-dilate 30 --arm-thickness 150\n")

    # Debug summary
    if args.debug:
        sp = Path(args.output).parent / "debug_summary.txt"
        with open(sp, "w") as f:
            f.write(f"Input           : {args.input}\n")
            f.write(f"Output          : {args.output}\n")
            f.write(f"Resolution      : {width}x{height}\n")
            f.write(f"FPS             : {fps:.2f}\n")
            f.write(f"Total frames    : {total}\n")
            f.write(f"High-cov frames : {high_cov}\n")
            f.write(f"mask-dilate     : {args.mask_dilate}\n")
            f.write(f"arm-length      : {args.arm_length}\n")
            f.write(f"arm-thickness   : {args.arm_thickness}\n")
            f.write(f"feather         : {args.feather}\n")
            f.write(f"stable-frames   : {args.stable_frames}\n")
            f.write(f"art-threshold   : {args.art_threshold}\n")
            f.write(f"sleeve-mask     : {args.enable_sleeve_mask}\n")
            f.write(f"marker-mask     : {args.enable_marker_mask}\n")
        print(f"  Debug files saved:")
        print(f"    debug_mask.mp4       — red overlay of masked regions")
        print(f"    debug_canvas.mp4     — the final canvas used for fills")
        print(f"    debug_summary.txt    — settings and frame stats\n")

    print(f"{'='*58}")
    print(f"  DONE")
    print(f"{'='*58}")
    print(f"  Output : {args.output}\n")


if __name__ == "__main__":
    main()
