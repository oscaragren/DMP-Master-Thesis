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

# HSV in OpenCV: H∈[0,180], S,V∈[0,255]. Tune per lighting.
# Note: variable names reflect *physical marker color*.
DEFAULT_GREEN_HSV_LO = (35, 80, 80)
DEFAULT_GREEN_HSV_HI = (85, 255, 255)
DEFAULT_BLUE_HSV_LO = (95, 80, 60)
DEFAULT_BLUE_HSV_HI = (125, 255, 255)

# Drawing colors (OpenCV uses BGR).
DRAW_GREEN = (0, 255, 0)
DRAW_BLUE = (255, 0, 0)


def _norm(v: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.linalg.norm(v) + eps)


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = _norm(v, eps=eps)
    return v / n


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


def _detect_colored_circles(
    frame_bgr: np.ndarray,
    *,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    morph_ksize: int,
    min_area: float,
    min_circularity: float,
    max_markers: int | None = None,
    min_radius: float | None = None,
    max_radius: float | None = None,
    min_fill_ratio: float | None = None,
):
    """
    Detect colored circular blobs.

    Returns (detections, mask) where detections is a list of dicts:
      { "u","v","r","area","circularity","fill","score" }
    Sorted by descending score (area * circularity).
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
        return [], mask

    dets: list[dict] = []
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
        if min_radius is not None and float(radius) < float(min_radius):
            continue
        if max_radius is not None and float(radius) > float(max_radius):
            continue

        circle_area = float(np.pi * float(radius) * float(radius))
        fill = float(area / circle_area) if circle_area > 1e-9 else 0.0
        if min_fill_ratio is not None and fill < float(min_fill_ratio):
            continue

        score = area * circularity
        dets.append(
            {
                "u": float(x),
                "v": float(y),
                "r": float(radius),
                "area": area,
                "circularity": circularity,
                "fill": fill,
                "score": score,
            }
        )

    dets.sort(key=lambda d: float(d["score"]), reverse=True)
    if max_markers is not None:
        mm = int(max_markers)
        if mm >= 0:
            dets = dets[:mm]
    return dets, mask


def detect_markers_2d(
    frame_bgr: np.ndarray,
    *,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    morph_ksize: int = 5,
    min_area: float = 120.0,
    min_circularity: float = 0.7,
    max_markers: int | None = None,
    min_radius: float | None = None,
    max_radius: float | None = None,
    min_fill_ratio: float | None = 0.55,
):
    """Detect circular color markers in 2D (no depth)."""
    return _detect_colored_circles(
        frame_bgr,
        hsv_lo=hsv_lo,
        hsv_hi=hsv_hi,
        morph_ksize=morph_ksize,
        min_area=min_area,
        min_circularity=min_circularity,
        max_markers=max_markers,
        min_radius=min_radius,
        max_radius=max_radius,
        min_fill_ratio=min_fill_ratio,
    )


def attach_depth_xyz(
    dets_2d: list[dict],
    *,
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    patch: int = 7,
    min_z: float = 0.05,
    max_z: float = 15.0,
):
    """Attach z + XYZ to each 2D detection; returns a new list."""
    out: list[dict] = []
    for d in dets_2d:
        u, v = float(d["u"]), float(d["v"])
        z_m = depth_at(depth_mm, u, v, patch=patch)
        if z_m is not None and float(min_z) <= float(z_m) <= float(max_z):
            xyz = deproject(u, v, float(z_m), float(fx), float(fy), float(cx), float(cy))
        else:
            xyz = None

        dd = dict(d)
        dd["z_m"] = float(z_m) if z_m is not None else None
        dd["xyz"] = xyz
        out.append(dd)
    return out


def detect_blue_green_markers(
    frame_bgr: np.ndarray,
    *,
    depth_mm: np.ndarray | None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    patch: int = 7,
    min_z: float = 0.05,
    max_z: float = 15.0,
    blue_hsv_lo: tuple[int, int, int] = DEFAULT_BLUE_HSV_LO,
    blue_hsv_hi: tuple[int, int, int] = DEFAULT_BLUE_HSV_HI,
    green_hsv_lo: tuple[int, int, int] = DEFAULT_GREEN_HSV_LO,
    green_hsv_hi: tuple[int, int, int] = DEFAULT_GREEN_HSV_HI,
    morph_ksize: int = 5,
    min_area: float = 120.0,
    min_circularity: float = 0.7,
    max_markers_each: int | None = 10,
    min_radius: float | None = None,
    max_radius: float | None = None,
    min_fill_ratio: float | None = 0.55,
):
    """
    Detect blue + green circular markers.

    Returns:
      (blue_dets, green_dets, masks) where masks is {"blue": mask, "green": mask}
    Each det includes u,v,r,... and (if depth provided) z_m + xyz.
    """
    blue_2d, blue_mask = detect_markers_2d(
        frame_bgr,
        hsv_lo=blue_hsv_lo,
        hsv_hi=blue_hsv_hi,
        morph_ksize=morph_ksize,
        min_area=min_area,
        min_circularity=min_circularity,
        max_markers=max_markers_each,
        min_radius=min_radius,
        max_radius=max_radius,
        min_fill_ratio=min_fill_ratio,
    )
    green_2d, green_mask = detect_markers_2d(
        frame_bgr,
        hsv_lo=green_hsv_lo,
        hsv_hi=green_hsv_hi,
        morph_ksize=morph_ksize,
        min_area=min_area,
        min_circularity=min_circularity,
        max_markers=max_markers_each,
        min_radius=min_radius,
        max_radius=max_radius,
        min_fill_ratio=min_fill_ratio,
    )

    if depth_mm is not None:
        if fx is None or fy is None or cx is None or cy is None:
            raise ValueError("fx, fy, cx, cy must be provided when depth_mm is provided")
        blue = attach_depth_xyz(blue_2d, depth_mm=depth_mm, fx=fx, fy=fy, cx=cx, cy=cy, patch=patch, min_z=min_z, max_z=max_z)
        green = attach_depth_xyz(green_2d, depth_mm=depth_mm, fx=fx, fy=fy, cx=cx, cy=cy, patch=patch, min_z=min_z, max_z=max_z)
    else:
        blue, green = blue_2d, green_2d

    return blue, green, {"blue": blue_mask, "green": green_mask}


def assign_blue_joints_by_vertical(blue_dets: list[dict]):
    """
    Heuristic for single front view: sort by image y (v).

    Returns dict with keys shoulder/elbow/wrist if >=3 detections, else {}.
    """
    if len(blue_dets) < 3:
        return {}
    sel = sorted(blue_dets, key=lambda d: float(d["v"]))[:3]
    return {"shoulder": sel[0], "elbow": sel[1], "wrist": sel[2]}


def estimate_upper_arm_frame_from_green(
    *,
    shoulder_xyz: tuple[float, float, float],
    elbow_xyz: tuple[float, float, float],
    green_xyz: list[tuple[float, float, float]],
):
    """
    Build a simple right-handed frame for the upper arm.

    - y-axis: humerus axis (shoulder -> elbow)
    - x-axis: in marker plane, perpendicular to y (derived from green-plane normal)
    - z-axis: completes right-handed frame

    Returns: (R, origin) where R is 3x3 with columns [x y z] in camera coords.
    """
    if len(green_xyz) < 3:
        return None

    s = np.array(shoulder_xyz, dtype=np.float64)
    e = np.array(elbow_xyz, dtype=np.float64)
    y = _unit(e - s)

    g1 = np.array(green_xyz[0], dtype=np.float64)
    g2 = np.array(green_xyz[1], dtype=np.float64)
    g3 = np.array(green_xyz[2], dtype=np.float64)
    n = np.cross(g2 - g1, g3 - g1)
    if _norm(n) < 1e-6:
        return None
    n = _unit(n)

    x = np.cross(n, y)
    if _norm(x) < 1e-6:
        return None
    x = _unit(x)
    z = _unit(np.cross(y, x))

    R = np.stack([x, y, z], axis=1)  # columns
    return R, tuple(float(v) for v in s)


def main():
    ap = argparse.ArgumentParser(description="Detect circular markers (blue/green) in RGB, read aligned depth, output 3D XYZ (DepthAI v3).")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--patch", type=int, default=7, help="Median patch size (pixels) for depth sampling")
    ap.add_argument("--min-z", type=float, default=0.05, help="Min valid depth (m)")
    ap.add_argument("--max-z", type=float, default=15.0, help="Max valid depth (m)")
    ap.add_argument("--no-show", action="store_true", help="Run headless (no imshow)")
    ap.add_argument("--show-mask", action="store_true", help="Show HSV mask windows (debug)")

    ap.add_argument("--blue-hsv-lo", type=int, nargs=3, default=list(DEFAULT_BLUE_HSV_LO), metavar=("H", "S", "V"))
    ap.add_argument("--blue-hsv-hi", type=int, nargs=3, default=list(DEFAULT_BLUE_HSV_HI), metavar=("H", "S", "V"))
    ap.add_argument("--green-hsv-lo", type=int, nargs=3, default=list(DEFAULT_GREEN_HSV_LO), metavar=("H", "S", "V"))
    ap.add_argument("--green-hsv-hi", type=int, nargs=3, default=list(DEFAULT_GREEN_HSV_HI), metavar=("H", "S", "V"))

    ap.add_argument("--morph-ksize", type=int, default=5, help="Morphology kernel size (odd)")
    ap.add_argument("--min-area", type=float, default=120.0, help="Min blob area in pixels")
    ap.add_argument("--min-circularity", type=float, default=0.7, help="Min contour circularity [0..1]")
    ap.add_argument("--min-fill", type=float, default=0.55, help="Min fill ratio area/(pi*r^2)")
    ap.add_argument("--min-radius", type=float, default=4.0, help="Min circle radius (px)")
    ap.add_argument("--max-radius", type=float, default=200.0, help="Max circle radius (px)")
    ap.add_argument("--max-markers", type=int, default=10, help="Max markers per color to report/draw (sorted by score)")
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
                min_radius=float(args.min_radius) if args.min_radius is not None else None,
                max_radius=float(args.max_radius) if args.max_radius is not None else None,
                min_fill_ratio=float(args.min_fill),
            )

            vis = frame_bgr.copy() if show else None
            xyz_list: list[tuple[str, int, tuple[float, float, float]] | None] = []
            overlay_rows = 0
            for color_name, dets, draw_color in (("blue", blue, DRAW_BLUE), ("green", green, DRAW_GREEN)):
                for i, d in enumerate(dets):
                    u, v, r = float(d["u"]), float(d["v"]), float(d["r"])
                    xyz = d.get("xyz")
                    if isinstance(xyz, tuple):
                        xyz_list.append((color_name, i, xyz))

                    if show:
                        cv2.circle(vis, (int(round(u)), int(round(v))), int(round(r)), draw_color, 2)
                        cv2.circle(vis, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)
                        label = f"{color_name}[{i}] u={u:.1f} v={v:.1f} r={r:.1f}"
                        if isinstance(xyz, tuple):
                            x, y, z = xyz
                            label += f" | X={x:.3f} Y={y:.3f} Z={z:.3f} m"
                        else:
                            label += " | depth invalid"
                        cv2.putText(
                            vis,
                            label,
                            (10, 30 + 20 * overlay_rows),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            draw_color,
                            2,
                            cv2.LINE_AA,
                        )
                        overlay_rows += 1

            # Print at ~10 Hz when we have a valid 3D point
            now = time.time()
            if xyz_list and (now - last_print_t) > 0.1:
                last_print_t = now
                for color_name, i, xyz in xyz_list:
                    x, y, z = xyz
                    print(f"{color_name}_{i}_xyz_m: {x:.4f} {y:.4f} {z:.4f}")

            if show:
                cv2.imshow("marker", vis)
                if args.show_mask:
                    cv2.imshow("mask_blue", masks["blue"])
                    cv2.imshow("mask_green", masks["green"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
