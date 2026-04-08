#!/usr/bin/env python3

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from marker import (
    DRAW_BLUE,
    DRAW_GREEN,
    assign_blue_joints_by_vertical,
    detect_blue_green_markers,
    estimate_upper_arm_frame_from_green,
)


RGB_SOCKET = dai.CameraBoardSocket.CAM_A
LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C


COUNTDOWN_SECONDS = 3  # 3,2,1 then Go
RECORD_DURATION = 8.0  # seconds to record after countdown


def _draw_centered_text(frame, text, font_scale=3, thickness=6):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = (w - tw) // 2, (h + th) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


def _draw_detection_status_panel(
    frame_bgr: np.ndarray,
    *,
    shoulder_ok: bool,
    elbow_ok: bool,
    wrist_ok: bool,
    green_ok_count: int,
    upper_arm_frame_ok: bool,
):
    """
    Draw a small panel that indicates if each expected marker is currently detected
    with valid depth (3D) so you can visually verify tracking.
    """
    x0, y0 = 10, 70
    line_h = 22
    pad = 8

    rows = [
        ("BLUE shoulder", shoulder_ok, DRAW_BLUE),
        ("BLUE elbow", elbow_ok, DRAW_BLUE),
        ("BLUE wrist", wrist_ok, DRAW_BLUE),
        ("GREEN markers", green_ok_count >= 3, DRAW_GREEN),
        ("Upper-arm frame", upper_arm_frame_ok, (255, 255, 255)),
    ]

    width = 320
    height = pad * 2 + line_h * len(rows)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    alpha = 0.45
    frame_bgr[:] = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

    for i, (label, ok, color) in enumerate(rows):
        y = y0 + pad + (i + 1) * line_h - 6
        status = "OK" if ok else "MISSING"
        status_color = (0, 200, 0) if ok else (0, 0, 255)
        cv2.putText(frame_bgr, label, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, status, (x0 + 210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Record 3D arm pose from colored markers (OAK-D RGBD, front view).")
    ap.add_argument("--subject", type=int, required=True, help="Subject number (1, 2, 3, ...)")
    ap.add_argument("--motion", type=str, required=True, help="Motion name (e.g. reach, curved_reach)")
    ap.add_argument("--trial", type=int, required=True, help="Trial number")
    ap.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("test_data/processed"),
        help="Root of processed output (subject/motion/trial will be appended)",
    )
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--patch", type=int, default=7)
    ap.add_argument("--min-z", type=float, default=0.05)
    ap.add_argument("--max-z", type=float, default=15.0)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--json", action="store_true", help="Also save JSON sequence")
    ap.add_argument("--show-mask", action="store_true")

    ap.add_argument("--blue-hsv-lo", type=int, nargs=3, default=[95, 80, 60], metavar=("H", "S", "V"))
    ap.add_argument("--blue-hsv-hi", type=int, nargs=3, default=[125, 255, 255], metavar=("H", "S", "V"))
    ap.add_argument("--green-hsv-lo", type=int, nargs=3, default=[35, 80, 80], metavar=("H", "S", "V"))
    ap.add_argument("--green-hsv-hi", type=int, nargs=3, default=[85, 255, 255], metavar=("H", "S", "V"))

    ap.add_argument("--morph-ksize", type=int, default=5)
    ap.add_argument("--min-area", type=float, default=40.0)
    ap.add_argument("--min-circularity", type=float, default=0.75)
    ap.add_argument("--min-fill", type=float, default=0.6)
    ap.add_argument("--min-radius", type=float, default=5.0)
    ap.add_argument("--max-radius", type=float, default=200.0)
    ap.add_argument("--max-markers", type=int, default=10)
    args = ap.parse_args()

    # Output: test_data/processed/subject_01/motion/trial_001/
    outdir = args.processed_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {outdir}")

    show = not args.no_show
    rgb_size = (int(args.width), int(args.height))
    mono_size = (640, 400)

    # Fixed marker ordering in saved arrays:
    # 0 blue_shoulder, 1 blue_elbow, 2 blue_wrist, 3 green_0, 4 green_1, 5 green_2
    KEYPOINT_NAMES = ["blue_shoulder", "blue_elbow", "blue_wrist", "green_0", "green_1", "green_2"]

    all_frames: list[np.ndarray] = []  # list of (6,3) float32 arrays
    all_t: list[float] = []
    all_json: list[dict] = []

    device = dai.Device()
    with dai.Pipeline(device) as pipeline:
        cam_rgb = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
        left = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
        right = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)

        stereo = pipeline.create(dai.node.StereoDepth)
        sync = pipeline.create(dai.node.Sync)

        stereo.setExtendedDisparity(True)
        sync.setSyncThreshold(timedelta(seconds=1 / (2 * max(1, args.fps))))

        rgb_stream = cam_rgb.requestOutput(size=rgb_size, fps=args.fps, enableUndistortion=True)
        left.requestOutput(size=mono_size, fps=args.fps).link(stereo.left)
        right.requestOutput(size=mono_size, fps=args.fps).link(stereo.right)

        rgb_stream.link(stereo.inputAlignTo)
        rgb_stream.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth_aligned"])

        queue = sync.out.createOutputQueue()

        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(RGB_SOCKET, rgb_size[0], rgb_size[1])
        fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])

        pipeline.start()

        if show:
            cv2.namedWindow("get_pose", cv2.WINDOW_NORMAL)
            if args.show_mask:
                cv2.namedWindow("mask_blue", cv2.WINDOW_NORMAL)
                cv2.namedWindow("mask_green", cv2.WINDOW_NORMAL)

        # Show a waiting frame immediately (queue.get() can block at startup)
        if show:
            wait_img = np.zeros((rgb_size[1], rgb_size[0], 3), dtype=np.uint8)
            wait_img[:] = (40, 40, 40)
            cv2.putText(
                wait_img, "Waiting for first frame...", (50, rgb_size[1] // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2,
            )
            cv2.putText(
                wait_img, "Press 'q' to quit once stream starts.", (50, rgb_size[1] // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
            )
            cv2.imshow("get_pose", wait_img)
            cv2.waitKey(1)

        print(f"Countdown {COUNTDOWN_SECONDS}s, then recording {RECORD_DURATION}s. Press 'q' to quit early.")
        phase = "countdown"
        countdown_start = None
        record_start = None
        video_writer = None
        video_path = outdir / "video.mp4"

        last_print_t = 0.0
        while pipeline.isRunning():
            msg_group = queue.get()
            frame_rgb = msg_group["rgb"]
            frame_depth = msg_group["depth_aligned"]

            frame_bgr = frame_rgb.getCvFrame()
            depth_mm = frame_depth.getFrame()

            if countdown_start is None:
                countdown_start = time.time()

            now = time.time()

            # Countdown overlay, then start recording
            if phase == "countdown":
                elapsed = now - countdown_start
                display_frame = frame_bgr.copy()
                if elapsed < 1:
                    _draw_centered_text(display_frame, "3")
                elif elapsed < 2:
                    _draw_centered_text(display_frame, "2")
                elif elapsed < 3:
                    _draw_centered_text(display_frame, "1")
                elif elapsed < 3.5:
                    _draw_centered_text(display_frame, "Go!")
                else:
                    phase = "recording"
                    record_start = time.time()
                    h, w = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
                    print("Recording...")
                if show:
                    cv2.imshow("get_pose", display_frame)
                    if args.show_mask:
                        cv2.imshow("mask_blue", np.zeros(frame_bgr.shape[:2], dtype=np.uint8))
                        cv2.imshow("mask_green", np.zeros(frame_bgr.shape[:2], dtype=np.uint8))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            if phase == "recording":
                record_elapsed = now - record_start
                if record_elapsed >= RECORD_DURATION:
                    break

            blue, green, masks = detect_blue_green_markers(
                frame_bgr,
                depth_mm=depth_mm,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                patch=int(args.patch),
                min_z=float(args.min_z),
                max_z=float(args.max_z),
                blue_hsv_lo=tuple(int(x) for x in args.blue_hsv_lo),
                blue_hsv_hi=tuple(int(x) for x in args.blue_hsv_hi),
                green_hsv_lo=tuple(int(x) for x in args.green_hsv_lo),
                green_hsv_hi=tuple(int(x) for x in args.green_hsv_hi),
                morph_ksize=int(args.morph_ksize),
                min_area=float(args.min_area),
                min_circularity=float(args.min_circularity),
                max_markers_each=int(args.max_markers),
                min_radius=float(args.min_radius),
                max_radius=float(args.max_radius),
                min_fill_ratio=float(args.min_fill),
            )

            # Enforce stable indexing: blue[0] is highest in frame (smallest v),
            # then blue[1], then blue[2]. Assumes markers don't swap vertically.
            blue = sorted(blue, key=lambda d: float(d.get("v", 0.0)))

            joints = assign_blue_joints_by_vertical(blue)
            shoulder_xyz = joints.get("shoulder", {}).get("xyz") if joints else None
            elbow_xyz = joints.get("elbow", {}).get("xyz") if joints else None
            wrist_xyz = joints.get("wrist", {}).get("xyz") if joints else None

            green_xyz = [d["xyz"] for d in green if isinstance(d.get("xyz"), tuple)]
            upper_arm_frame = None
            if isinstance(shoulder_xyz, tuple) and isinstance(elbow_xyz, tuple) and len(green_xyz) >= 3:
                upper_arm_frame = estimate_upper_arm_frame_from_green(
                    shoulder_xyz=shoulder_xyz,
                    elbow_xyz=elbow_xyz,
                    green_xyz=green_xyz[:3],
                )

            # Build (6,3) frame array
            frame_xyz = np.full((len(KEYPOINT_NAMES), 3), np.nan, dtype=np.float32)
            if isinstance(shoulder_xyz, tuple):
                frame_xyz[0] = shoulder_xyz
            if isinstance(elbow_xyz, tuple):
                frame_xyz[1] = elbow_xyz
            if isinstance(wrist_xyz, tuple):
                frame_xyz[2] = wrist_xyz
            for i in range(min(3, len(green_xyz))):
                frame_xyz[3 + i] = green_xyz[i]

            frame_json = None
            if args.json:
                frame_json = {"t": frame_rgb.getTimestampDevice().total_seconds(), "markers": {}}
                for i, name in enumerate(KEYPOINT_NAMES):
                    x, y, z = frame_xyz[i]
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        frame_json["markers"][name] = {"x": float(x), "y": float(y), "z": float(z)}

            vis = frame_bgr.copy() if show else None
            if show:
                # Draw detections
                for i, d in enumerate(blue):
                    u, v, r = float(d["u"]), float(d["v"]), float(d["r"])
                    cv2.circle(vis, (int(round(u)), int(round(v))), int(round(r)), DRAW_BLUE, 2)
                    cv2.putText(vis, f"blue[{i}]", (int(u) + 5, int(v) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW_BLUE, 2)
                for i, d in enumerate(green):
                    u, v, r = float(d["u"]), float(d["v"]), float(d["r"])
                    cv2.circle(vis, (int(round(u)), int(round(v))), int(round(r)), DRAW_GREEN, 2)
                    cv2.putText(vis, f"green[{i}]", (int(u) + 5, int(v) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW_GREEN, 2)

                if joints:
                    cv2.putText(
                        vis,
                        "blue joints: shoulder/elbow/wrist (by vertical sort)",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if upper_arm_frame is not None:
                    cv2.putText(
                        vis,
                        "upper-arm frame: OK",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        vis,
                        "upper-arm frame: missing markers/depth",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (180, 180, 180),
                        2,
                        cv2.LINE_AA,
                    )

                # Status panel: are the expected markers actually detected (with valid depth)?
                _draw_detection_status_panel(
                    vis,
                    shoulder_ok=bool(isinstance(shoulder_xyz, tuple)),
                    elbow_ok=bool(isinstance(elbow_xyz, tuple)),
                    wrist_ok=bool(isinstance(wrist_xyz, tuple)),
                    green_ok_count=int(len(green_xyz)),
                    upper_arm_frame_ok=bool(upper_arm_frame is not None),
                )

            now = time.time()
            if (now - last_print_t) > 0.1:
                last_print_t = now
                if isinstance(shoulder_xyz, tuple):
                    sx, sy, sz = shoulder_xyz
                    print(f"shoulder_xyz_m: {sx:.4f} {sy:.4f} {sz:.4f}")
                if isinstance(elbow_xyz, tuple):
                    ex, ey, ez = elbow_xyz
                    print(f"elbow_xyz_m:    {ex:.4f} {ey:.4f} {ez:.4f}")
                if isinstance(wrist_xyz, tuple):
                    wx, wy, wz = wrist_xyz
                    print(f"wrist_xyz_m:    {wx:.4f} {wy:.4f} {wz:.4f}")
                if upper_arm_frame is not None:
                    R, origin = upper_arm_frame
                    ox, oy, oz = origin
                    print(f"upper_arm_origin_m: {ox:.4f} {oy:.4f} {oz:.4f}")
                    rflat = " ".join(f"{float(x):.5f}" for x in R.reshape(-1))
                    print(f"upper_arm_R_colmajor: {rflat}")

            if show:
                cv2.imshow("get_pose", vis)
                if args.show_mask:
                    cv2.imshow("mask_blue", masks["blue"])
                    cv2.imshow("mask_green", masks["green"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Save during recording
            if video_writer is not None:
                video_writer.write(vis if show else frame_bgr)
            all_frames.append(frame_xyz)
            all_t.append(frame_rgb.getTimestampDevice().total_seconds())
            if frame_json is not None:
                all_json.append(frame_json)

        if show:
            cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved video: {video_path}")

    seq = np.stack(all_frames, axis=0) if all_frames else np.zeros((0, len(KEYPOINT_NAMES), 3), dtype=np.float32)
    t = np.array(all_t, dtype=np.float64)

    # Keep same naming convention as 3d_pose.py for downstream compatibility
    npy_seq_path = outdir / "left_arm_seq_camera.npy"
    npy_t_path = outdir / "left_arm_t.npy"
    np.save(npy_seq_path, seq)
    np.save(npy_t_path, t)

    meta = {
        "subject": args.subject,
        "motion": args.motion,
        "trial": args.trial,
        "shape": list(seq.shape),
        "keypoint_names": KEYPOINT_NAMES,
        "source": "get_pose.py (marker-based, OAK-D live)",
        "record_duration_sec": RECORD_DURATION,
        "marker_setup": {
            "blue": ["shoulder", "elbow", "wrist"],
            "green": ["upper_arm_0", "upper_arm_1", "upper_arm_2"],
            "blue_joint_assignment": "sorted by image v (top->bottom)",
            "green_ordering": "top-3 green detections (by score) in returned order",
        },
        "detection_params": {
            "blue_hsv_lo": [int(x) for x in args.blue_hsv_lo],
            "blue_hsv_hi": [int(x) for x in args.blue_hsv_hi],
            "green_hsv_lo": [int(x) for x in args.green_hsv_lo],
            "green_hsv_hi": [int(x) for x in args.green_hsv_hi],
            "morph_ksize": int(args.morph_ksize),
            "min_area": float(args.min_area),
            "min_circularity": float(args.min_circularity),
            "min_fill": float(args.min_fill),
            "min_radius": float(args.min_radius),
            "max_radius": float(args.max_radius),
            "max_markers": int(args.max_markers),
            "patch": int(args.patch),
            "min_z": float(args.min_z),
            "max_z": float(args.max_z),
        },
    }
    if (outdir / "video.mp4").exists():
        meta["video"] = "video.mp4"
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved:\n  {npy_seq_path}  shape={seq.shape}\n  {npy_t_path}  shape={t.shape}\n  {outdir / 'meta.json'}")

    if args.json:
        json_path = outdir / "left_arm_sequence.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_json, f, indent=2)
        print(f"  {json_path}  frames={len(all_json)}")


if __name__ == "__main__":
    main()

