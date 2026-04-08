#!/usr/bin/env python3

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from marker import deproject, depth_at


RGB_SOCKET = dai.CameraBoardSocket.CAM_A
LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C

COUNTDOWN_SECONDS = 3  # 3,2,1 then Go
RECORD_DURATION = 8.0  # seconds to record after countdown


# COCO-17 joint indices (common for HRNet COCO models).
COCO17 = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Keep the same output ordering as capture/3d_pose.py.
OUTPUT_JOINT_NAMES = ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "left_hip", "right_hip"]
OUTPUT_JOINT_COCO_IDS = [COCO17[n] for n in OUTPUT_JOINT_NAMES]


def _draw_centered_text(frame, text, font_scale=3, thickness=6):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = (w - tw) // 2, (h + th) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


def _infer_heatmap_layout(flat_len: int, *, num_joints: int, hm_h: int | None, hm_w: int | None):
    if hm_h is not None and hm_w is not None:
        expected = int(num_joints) * int(hm_h) * int(hm_w)
        if expected != int(flat_len):
            raise ValueError(
                f"NN output length mismatch: got {flat_len}, expected {expected} "
                f"(num_joints={num_joints}, hm_h={hm_h}, hm_w={hm_w})."
            )
        return int(num_joints), int(hm_h), int(hm_w)

    if flat_len % int(num_joints) != 0:
        raise ValueError(
            f"Cannot infer heatmap size: output length {flat_len} not divisible by num_joints={num_joints}. "
            "Pass --hm-h/--hm-w."
        )
    per = flat_len // int(num_joints)
    # Typical HRNet COCO: 64x48 (=3072) or 96x72 (=6912)
    for (h, w) in ((64, 48), (48, 64), (96, 72), (72, 96), (80, 60), (60, 80)):
        if per == h * w:
            return int(num_joints), int(h), int(w)
    s = int(round(np.sqrt(per)))
    if s * s == per:
        return int(num_joints), int(s), int(s)

    raise ValueError(f"Cannot infer heatmap HxW from per-joint size {per}. Pass --hm-h/--hm-w.")


def _argmax_2d(hm: np.ndarray):
    # hm shape: (H, W)
    idx = int(np.argmax(hm))
    y, x = divmod(idx, int(hm.shape[1]))
    score = float(hm[y, x])
    return float(x), float(y), score


def decode_hrnet_heatmaps(
    heatmaps: np.ndarray, *, input_w: int, input_h: int, conf_threshold: float = 0.0
):
    """
    heatmaps: (J, HmH, HmW)
    Returns:
      u_in, v_in, score arrays of shape (J,)
    Coordinates are in NN input pixel space [0..input_w/input_h).
    """
    j, hm_h, hm_w = heatmaps.shape
    stride_x = float(input_w) / float(hm_w)
    stride_y = float(input_h) / float(hm_h)

    u_in = np.full((j,), np.nan, dtype=np.float32)
    v_in = np.full((j,), np.nan, dtype=np.float32)
    scores = np.full((j,), np.nan, dtype=np.float32)

    for k in range(j):
        x_hm, y_hm, s = _argmax_2d(heatmaps[k])
        if float(s) < float(conf_threshold):
            continue
        # Center-of-cell projection to input space.
        u_in[k] = (x_hm + 0.5) * stride_x
        v_in[k] = (y_hm + 0.5) * stride_y
        scores[k] = float(s)

    return u_in, v_in, scores


def main():
    ap = argparse.ArgumentParser(description="HRNet pose estimation on-device (DepthAI) + aligned depth -> 3D joints.")
    ap.add_argument("--subject", type=int, required=True, help="Subject number (1, 2, 3, ...)")
    ap.add_argument("--motion", type=str, required=True, help="Motion name (e.g. reach, curved_reach)")
    ap.add_argument("--trial", type=int, required=True, help="Trial number")
    ap.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("test_data/processed"),
        help="Root of processed output (subject/motion/trial will be appended)",
    )

    ap.add_argument("--blob", type=str, required=True, help="Path to HRNet OpenVINO blob (e.g. *.blob)")
    ap.add_argument("--output-layer", type=str, default="", help="Heatmap output layer name (leave empty to auto-pick)")
    ap.add_argument("--num-joints", type=int, default=17, help="Number of joints in the heatmap output (COCO-17 default)")
    ap.add_argument("--hm-h", type=int, default=64, help="Heatmap height (typical HRNet: 64)")
    ap.add_argument("--hm-w", type=int, default=48, help="Heatmap width (typical HRNet: 48)")
    ap.add_argument("--conf", type=float, default=0.1, help="Min heatmap peak score to accept a joint")

    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--rgb-w", type=int, default=640)
    ap.add_argument("--rgb-h", type=int, default=400)
    ap.add_argument("--nn-w", type=int, default=192, help="NN input width (e.g. 192 for HRNet-w32 256x192)")
    ap.add_argument("--nn-h", type=int, default=256, help="NN input height (e.g. 256 for HRNet-w32 256x192)")
    ap.add_argument("--patch", type=int, default=7, help="Median patch size for depth sampling")
    ap.add_argument("--min-z", type=float, default=0.0, help="Minimum depth for valid joint")
    ap.add_argument("--max-z", type=float, default=10.0, help="Maximum depth for valid joint")
    ap.add_argument("--json", action="store_true", help="Also save JSON sequence")
    ap.add_argument("--no-show", action="store_true", help="Disable cv2.imshow (run headless)")
    ap.add_argument("--show-depth", action="store_true", help="Show a second window with depth colormap")
    args = ap.parse_args()

    show_window = not args.no_show

    outdir = args.processed_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {outdir}")

    rgb_size = (int(args.rgb_w), int(args.rgb_h))
    nn_size = (int(args.nn_w), int(args.nn_h))
    mono_size = (640, 400)

    all_frames: list[np.ndarray] = []  # (T, 6, 3)
    all_t: list[float] = []
    all_json: list[dict] = []

    device = dai.Device()
    with dai.Pipeline(device) as pipeline:
        cam_rgb = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
        left = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
        right = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)

        stereo = pipeline.create(dai.node.StereoDepth)
        sync = pipeline.create(dai.node.Sync)

        manip = pipeline.create(dai.node.ImageManip)
        nn = pipeline.create(dai.node.NeuralNetwork)

        stereo.setExtendedDisparity(True)
        sync.setSyncThreshold(timedelta(seconds=1 / (2 * max(1, int(args.fps)))))

        rgb_stream = cam_rgb.requestOutput(size=rgb_size, fps=int(args.fps), enableUndistortion=True)
        left.requestOutput(size=mono_size, fps=int(args.fps)).link(stereo.left)
        right.requestOutput(size=mono_size, fps=int(args.fps)).link(stereo.right)

        # Align depth to RGB.
        rgb_stream.link(stereo.inputAlignTo)

        # NN preproc.
        manip.initialConfig.setResize(nn_size[0], nn_size[1])
        manip.initialConfig.setKeepAspectRatio(False)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        rgb_stream.link(manip.inputImage)

        nn.setBlobPath(str(Path(args.blob)))
        manip.out.link(nn.input)

        # Sync streams to host.
        rgb_stream.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth_aligned"])
        nn.out.link(sync.inputs["nn"])
        queue = sync.out.createOutputQueue()

        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(RGB_SOCKET, rgb_size[0], rgb_size[1])
        fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])

        pipeline.start()

        if show_window:
            cv2.namedWindow("OAK-D: HRNet Pose 3D", cv2.WINDOW_NORMAL)
            if args.show_depth:
                cv2.namedWindow("OAK-D: Depth", cv2.WINDOW_NORMAL)
            wait_img = np.zeros((rgb_size[1], rgb_size[0], 3), dtype=np.uint8)
            wait_img[:] = (40, 40, 40)
            cv2.putText(
                wait_img,
                "Waiting for first frame...",
                (50, rgb_size[1] // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )
            cv2.imshow("OAK-D: HRNet Pose 3D", wait_img)
            cv2.waitKey(1)

        print(f"Countdown {COUNTDOWN_SECONDS}s, then recording {RECORD_DURATION}s. Press 'q' to quit early.")
        phase = "countdown"
        countdown_start = None
        record_start = None
        video_writer = None
        video_path = outdir / "video.mp4"

        while pipeline.isRunning():
            msg_group = queue.get()
            frame_rgb = msg_group["rgb"]
            frame_depth = msg_group["depth_aligned"]
            nn_data = msg_group["nn"]

            frame_bgr = frame_rgb.getCvFrame()
            depth_mm = frame_depth.getFrame()

            if countdown_start is None:
                countdown_start = time.time()

            now = time.time()
            t_sec = frame_rgb.getTimestampDevice().total_seconds()

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
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, int(args.fps), (w, h))
                    print("Recording...")
                if show_window:
                    cv2.imshow("OAK-D: HRNet Pose 3D", display_frame)
                    if args.show_depth:
                        cv2.imshow("OAK-D: Depth", np.zeros_like(display_frame))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            if phase == "recording":
                record_elapsed = now - record_start
                if record_elapsed >= RECORD_DURATION:
                    break

            # Extract heatmaps.
            layer_names = list(nn_data.getAllLayerNames())
            if args.output_layer:
                layer = args.output_layer
            else:
                if not layer_names:
                    raise RuntimeError("NNData contains no layers. Check your blob / network output.")
                layer = layer_names[0]

            try:
                flat = np.array(nn_data.getLayerFp16(layer), dtype=np.float32)
            except Exception:
                # Fallback: some blobs output u8.
                flat = np.array(nn_data.getLayerUInt8(layer), dtype=np.float32)

            j, hm_h, hm_w = _infer_heatmap_layout(
                int(flat.size), num_joints=int(args.num_joints), hm_h=int(args.hm_h), hm_w=int(args.hm_w)
            )
            heatmaps = flat.reshape((j, hm_h, hm_w))

            u_in, v_in, scores = decode_hrnet_heatmaps(
                heatmaps, input_w=nn_size[0], input_h=nn_size[1], conf_threshold=float(args.conf)
            )

            # Map NN-input coords -> RGB coords (manip is a direct resize, no padding).
            sx = float(rgb_size[0]) / float(nn_size[0])
            sy = float(rgb_size[1]) / float(nn_size[1])
            u_rgb = u_in * sx
            v_rgb = v_in * sy

            pose_xyz = np.full((len(OUTPUT_JOINT_COCO_IDS), 3), np.nan, dtype=np.float32)
            frame_json = {"t": t_sec, "joints": {}} if args.json else None

            for out_i, coco_id in enumerate(OUTPUT_JOINT_COCO_IDS):
                u = float(u_rgb[coco_id]) if coco_id < u_rgb.size else float("nan")
                v = float(v_rgb[coco_id]) if coco_id < v_rgb.size else float("nan")
                s = float(scores[coco_id]) if coco_id < scores.size else float("nan")

                if not np.isfinite(u) or not np.isfinite(v):
                    continue

                z_m = depth_at(depth_mm, u, v, patch=int(args.patch))
                if z_m is None or float(z_m) < float(args.min_z) or float(z_m) > float(args.max_z):
                    continue

                x, y, z = deproject(u, v, float(z_m), fx, fy, cx, cy)
                pose_xyz[out_i] = (x, y, z)

                if frame_json is not None:
                    frame_json["joints"][OUTPUT_JOINT_NAMES[out_i]] = {
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "u": float(u),
                        "v": float(v),
                        "score": float(s),
                    }

                # Overlay
                if show_window:
                    cv2.circle(frame_bgr, (int(round(u)), int(round(v))), 3, (0, 255, 0), -1)
                    cv2.putText(
                        frame_bgr,
                        f"{OUTPUT_JOINT_NAMES[out_i]} s={s:.2f} z={z:.2f}m",
                        (int(round(u)) + 5, int(round(v)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            if phase == "recording":
                if video_writer is not None:
                    video_writer.write(frame_bgr)
                all_frames.append(pose_xyz)
                all_t.append(t_sec)
                if frame_json is not None:
                    all_json.append(frame_json)

            if show_window:
                cv2.imshow("OAK-D: HRNet Pose 3D", frame_bgr)
                if args.show_depth:
                    depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                    cv2.imshow("OAK-D: Depth", depth_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if show_window:
            cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved video: {video_path}")

    seq = np.stack(all_frames, axis=0) if all_frames else np.zeros((0, len(OUTPUT_JOINT_NAMES), 3), dtype=np.float32)
    t = np.array(all_t, dtype=np.float64)

    npy_seq_path = outdir / "left_arm_seq_camera.npy"
    npy_t_path = outdir / "left_arm_t.npy"
    np.save(npy_seq_path, seq)
    np.save(npy_t_path, t)

    meta = {
        "subject": args.subject,
        "motion": args.motion,
        "trial": args.trial,
        "shape": list(seq.shape),
        "keypoint_names": OUTPUT_JOINT_NAMES,
        "source": "hrnet_pe.py (DepthAI on-device HRNet + aligned depth)",
        "record_duration_sec": RECORD_DURATION,
        "pose_model": {
            "framework": "HRNet heatmap",
            "blob": str(args.blob),
            "num_joints": int(args.num_joints),
            "hm_h": int(args.hm_h),
            "hm_w": int(args.hm_w),
            "nn_input_w": int(args.nn_w),
            "nn_input_h": int(args.nn_h),
            "conf_threshold": float(args.conf),
            "output_layer": args.output_layer or None,
        },
        "depth_sampling": {
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

