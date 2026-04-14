"""
Live camera preview used by `collect_data.py`.

This is intentionally lightweight: it only opens a stream and shows it in an
OpenCV window so the operator can verify framing/lighting before recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time
from pathlib import Path
import cv2
import numpy as np


# Same subset as recording (`capture/record_data.py`)
POSE_KEYPOINT_IDS = [11, 13, 15, 12, 23, 24]
POSE_KEYPOINT_NAMES = {
    11: "left_shoulder",
    13: "left_elbow",
    15: "left_wrist",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
}
POSE_CONNECTIONS = [(11, 13), (13, 15), (11, 12)]


@dataclass(frozen=True)
class LivePreviewConfig:
    window_name: str = "Live preview"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    show_fps: bool = True
    show_pose: bool = True
    model_path: str = "capture/pose_landmarker_lite.task"


def _overlay_help(frame_bgr, *, lines: list[str]) -> None:
    h, w = frame_bgr.shape[:2]
    pad = 10
    x0, y0 = pad, pad
    line_h = 22
    box_h = pad * 2 + line_h * len(lines)
    box_w = min(w - 2 * pad, 720)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    frame_bgr[:] = cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0)

    for i, line in enumerate(lines):
        y = y0 + pad + (i + 1) * line_h - 6
        cv2.putText(
            frame_bgr,
            line,
            (x0 + pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )


def _draw_pose_overlay(frame_bgr: np.ndarray, pose_landmarks) -> None:
    h, w = frame_bgr.shape[:2]
    if pose_landmarks is None or len(pose_landmarks) < 25:
        return

    def to_px(idx: int) -> tuple[int, int]:
        lm = pose_landmarks[idx]
        return (int(float(lm.x) * w), int(float(lm.y) * h))

    for i, j in POSE_CONNECTIONS:
        if i < len(pose_landmarks) and j < len(pose_landmarks):
            cv2.line(frame_bgr, to_px(i), to_px(j), (0, 255, 0), 2, cv2.LINE_AA)

    for idx in POSE_KEYPOINT_IDS:
        if idx < len(pose_landmarks):
            u, v = to_px(idx)
            cv2.circle(frame_bgr, (u, v), 5, (0, 255, 0), -1)
            cv2.circle(frame_bgr, (u, v), 5, (255, 255, 255), 1)
            name = POSE_KEYPOINT_NAMES.get(idx, str(idx))
            cv2.putText(frame_bgr, name, (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)


def _run_preview_depthai(*, cfg: LivePreviewConfig) -> None:
    import depthai as dai
    import mediapipe as mp

    frame_size = (int(cfg.width), int(cfg.height))

    landmarker = None
    last_ts_ms = -1
    if cfg.show_pose:
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        model_path = Path(cfg.model_path)
        if not model_path.exists():
            model_path = (Path(__file__).resolve().parents[1] / cfg.model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Pose Landmarker model not found at '{cfg.model_path}' (also tried '{model_path}')")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = PoseLandmarker.create_from_options(options)

    try:
        device = dai.Device()
        with dai.Pipeline(device) as pipeline:
            cam = pipeline.create(dai.node.Camera).build()
            video_queue = cam.requestOutput(frame_size, fps=float(cfg.fps), enableUndistortion=True).createOutputQueue()

            pipeline.start()
            cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)

            last_t = time.time()
            fps_est: Optional[float] = None

            while pipeline.isRunning():
                img = video_queue.get()
                frame_bgr = img.getCvFrame()

                now = time.time()
                dt = max(1e-6, now - last_t)
                last_t = now
                if cfg.show_fps:
                    inst = 1.0 / dt
                    fps_est = inst if fps_est is None else (0.9 * fps_est + 0.1 * inst)

                view = frame_bgr.copy()

                pose_ok = False
                if landmarker is not None:
                    t_sec = img.getTimestampDevice().total_seconds()
                    ts_ms = int(t_sec * 1000)
                    if ts_ms <= last_ts_ms:
                        ts_ms = last_ts_ms + 1
                    last_ts_ms = ts_ms

                    frame_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    result = landmarker.detect_for_video(mp_image, ts_ms)
                    if result.pose_landmarks and len(result.pose_landmarks) > 0:
                        pose_ok = True
                        _draw_pose_overlay(view, result.pose_landmarks[0])

                lines = [
                    "SPACE: continue    q: quit",
                    "Verify framing + pose detection before recording.",
                ]
                if cfg.show_pose:
                    lines.append(f"pose: {'OK' if pose_ok else 'MISSING'}")
                if fps_est is not None and cfg.show_fps:
                    lines.append(f"stream fps ~ {fps_est:0.1f}")
                _overlay_help(view, lines=lines)

                cv2.imshow(cfg.window_name, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise SystemExit(0)
                if key == ord(" "):
                    break
    finally:
        if landmarker is not None:
            landmarker.close()
        cv2.destroyWindow(cfg.window_name)

def live_preview(*, args) -> None:
    """
    Show a live camera preview.

    Expected operator flow:
    - Start script
    - Verify camera feed
    - Press SPACE to continue to the recording loop
    - Press 'q' to quit the program
    """
    fps = float(getattr(args, "fps_nominal", 30.0) or 30.0)
    cfg = LivePreviewConfig(
        fps=fps,
        model_path=str(getattr(args, "model", LivePreviewConfig.model_path) or LivePreviewConfig.model_path),
        show_pose=True,
    )

    try:
        _run_preview_depthai(cfg=cfg)
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True

