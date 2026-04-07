#!/usr/bin/env python3

import argparse
import time
from datetime import timedelta

import cv2
import depthai as dai
import numpy as np


RGB_SOCKET = dai.CameraBoardSocket.CAM_A
LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C

# Matte blue marker defaults (HSV in OpenCV: H∈[0,180], S,V∈[0,255])
# Tune with --hsv-* args if lighting differs.
DEFAULT_BLUE_HSV_LO = (95, 80, 60)
DEFAULT_BLUE_HSV_HI = (125, 255, 255)


def deproject(u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float):
    """Pixel (u, v) with depth z (meters) -> camera-frame XYZ (meters)."""
    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy
    return float(x), float(y), float(z_m)


def depth_at(depth_mm: np.ndarray, u: float, v: float, patch: int = 7):
    """Median depth (meters) in a patch around (u, v). depth_mm is uint16 in mm."""
    if patch <= 1:
        patch = 1
    if patch % 2 == 0:
        patch += 1

    h, w = depth_mm.shape[:2]
    u0, v0 = int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))
    r = patch // 2
    x1, x2 = max(0, u0 - r), min(w, u0 + r + 1)
    y1, y2 = max(0, v0 - r), min(h, v0 + r + 1)

    roi = depth_mm[y1:y2, x1:x2].astype(np.float32)
    roi = roi[roi > 0]  # drop invalid
    if roi.size == 0:
        return None
    return float(np.median(roi)) / 1000.0  # mm -> m


def _detect_best_colored_circle(
    frame_bgr: np.ndarray,
    *,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    morph_ksize: int,
    min_area: float,
    min_circularity: float,
):
    """
    Detect a colored circular blob and return (u, v, r, mask) or None.
    Uses HSV thresholding + morphology + contour circularity + minEnclosingCircle.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(hsv_lo, dtype=np.uint8)
    hi = np.array(hsv_hi, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    k = int(max(1, morph_ksize))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = -1.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area):
            continue

        perim = float(cv2.arcLength(cnt, True))
        if perim <= 0:
            continue

        circularity = float(4.0 * np.pi * area / (perim * perim))
        if circularity < float(min_circularity):
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        score = area * circularity
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(radius), mask)

    return best


def main():
    ap = argparse.ArgumentParser(description="Detect circular marker in RGB, read aligned depth, output 3D XYZ (DepthAI v3).")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--patch", type=int, default=7, help="Median patch size (pixels) for depth sampling")
    ap.add_argument("--min-z", type=float, default=0.05, help="Min valid depth (m)")
    ap.add_argument("--max-z", type=float, default=15.0, help="Max valid depth (m)")
    ap.add_argument("--no-show", action="store_true", help="Run headless (no imshow)")
    ap.add_argument("--show-mask", action="store_true", help="Show HSV mask window (debug)")

    # Color segmentation defaults for a matte blue marker
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=list(DEFAULT_BLUE_HSV_LO), metavar=("H", "S", "V"))
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=list(DEFAULT_BLUE_HSV_HI), metavar=("H", "S", "V"))
    ap.add_argument("--morph-ksize", type=int, default=5, help="Morphology kernel size (odd)")
    ap.add_argument("--min-area", type=float, default=120.0, help="Min blob area in pixels")
    ap.add_argument("--min-circularity", type=float, default=0.6, help="Min contour circularity [0..1]")
    args = ap.parse_args()

    show = not args.no_show
    rgb_size = (int(args.width), int(args.height))
    mono_size = (640, 400)

    device = dai.Device()
    with dai.Pipeline(device) as pipeline:
        cam_rgb = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
        left = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
        right = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)

        stereo = pipeline.create(dai.node.StereoDepth)
        sync = pipeline.create(dai.node.Sync)

        stereo.setExtendedDisparity(True)
        sync.setSyncThreshold(timedelta(seconds=1 / (2 * max(1, args.fps))))

        # Streams
        rgb_stream = cam_rgb.requestOutput(size=rgb_size, fps=args.fps, enableUndistortion=True)
        left.requestOutput(size=mono_size, fps=args.fps).link(stereo.left)
        right.requestOutput(size=mono_size, fps=args.fps).link(stereo.right)

        # Align depth to RGB by feeding the RGB stream into stereo.inputAlignTo (DepthAI v3 API).
        rgb_stream.link(stereo.inputAlignTo)
        rgb_stream.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth_aligned"])

        queue = sync.out.createOutputQueue()

        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(RGB_SOCKET, rgb_size[0], rgb_size[1])
        fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])

        pipeline.start()

        if show:
            cv2.namedWindow("marker", cv2.WINDOW_NORMAL)
            if args.show_mask:
                cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

        last_print_t = 0.0
        while pipeline.isRunning():
            msg_group = queue.get()
            frame_rgb = msg_group["rgb"]
            frame_depth = msg_group["depth_aligned"]

            frame_bgr = frame_rgb.getCvFrame()
            depth_mm = frame_depth.getFrame()

            det = _detect_best_colored_circle(
                frame_bgr,
                hsv_lo=tuple(int(x) for x in args.hsv_lo),
                hsv_hi=tuple(int(x) for x in args.hsv_hi),
                morph_ksize=int(args.morph_ksize),
                min_area=float(args.min_area),
                min_circularity=float(args.min_circularity),
            )

            vis = frame_bgr.copy() if show else None
            xyz = None
            mask = None
            if det is not None:
                u, v, r, mask = det
                z_m = depth_at(depth_mm, u, v, patch=args.patch)
                if z_m is not None and args.min_z <= z_m <= args.max_z:
                    xyz = deproject(u, v, z_m, fx, fy, cx, cy)

                if show:
                    draw_color = (255, 0, 0)  # BGR: blue
                    cv2.circle(vis, (int(round(u)), int(round(v))), int(round(r)), draw_color, 2)
                    cv2.circle(vis, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)
                    cv2.rectangle(
                        vis,
                        (int(round(u - args.patch // 2)), int(round(v - args.patch // 2))),
                        (int(round(u + args.patch // 2)), int(round(v + args.patch // 2))),
                        draw_color,
                        1,
                    )
                    label = f"u={u:.1f} v={v:.1f} r={r:.1f}"
                    if xyz is not None:
                        x, y, z = xyz
                        label += f" | X={x:.3f} Y={y:.3f} Z={z:.3f} m"
                    else:
                        label += " | depth invalid"
                    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2, cv2.LINE_AA)

            # Print at ~10 Hz when we have a valid 3D point
            now = time.time()
            if xyz is not None and (now - last_print_t) > 0.1:
                last_print_t = now
                x, y, z = xyz
                print(f"marker_xyz_m: {x:.4f} {y:.4f} {z:.4f}")

            if show:
                cv2.imshow("marker", vis)
                if args.show_mask:
                    if mask is None:
                        cv2.imshow("mask", np.zeros(frame_bgr.shape[:2], dtype=np.uint8))
                    else:
                        cv2.imshow("mask", mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
