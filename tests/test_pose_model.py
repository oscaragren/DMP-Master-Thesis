import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


# MediaPipe Pose landmark indices: left arm + right shoulder
POSE_KEYPOINT_IDS = [11, 13, 15, 12]  # left_shoulder, left_elbow, left_wrist, right_shoulder
POSE_KEYPOINT_NAMES = {
    11: "left_shoulder",
    13: "left_elbow",
    15: "left_wrist",
    12: "right_shoulder",
}


@dataclass
class PoseSequence:
    """Container for a single model's pose sequence on one video."""

    coords: np.ndarray  # (T, 4, 3) x,y,z in normalized image coordinates
    visibility: np.ndarray  # (T, 4)
    presence: np.ndarray  # (T, 4)
    timestamps: np.ndarray  # (T,) seconds from video start
    runtime_sec: float


def _load_video_frames(video_path: Path):
    """Yield (frame_bgr, t_sec) from a video file.

    Timestamps are computed from frame index and FPS. This assumes the video
    has a constant frame rate, which is true for the captured test data.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Fallback: treat as 25 FPS if metadata is missing
        fps = 25.0

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = frame_idx / fps
            frame_idx += 1
            yield frame, t_sec
    finally:
        cap.release()


def _run_pose_model_on_video(model_path: Path, video_path: Path) -> PoseSequence:
    """Run a MediaPipe Pose Landmarker model on a single video."""
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    base_options = BaseOptions(model_asset_path=str(model_path))
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_coords: list[np.ndarray] = []
    all_vis: list[np.ndarray] = []
    all_pres: list[np.ndarray] = []
    all_t: list[float] = []

    start_time = time.perf_counter()
    ts_ms_last = -1

    with mp_tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_bgr, t_sec in _load_video_frames(video_path):
            # MediaPipe image expects SRGB (RGB) order
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Monotonic timestamps in ms for VIDEO mode
            ts_ms = int(t_sec * 1000.0)
            if ts_ms <= ts_ms_last:
                ts_ms = ts_ms_last + 1
            ts_ms_last = ts_ms

            result = landmarker.detect_for_video(mp_image, ts_ms)

            coords = np.full((len(POSE_KEYPOINT_IDS), 3), np.nan, dtype=np.float32)
            vis = np.zeros((len(POSE_KEYPOINT_IDS),), dtype=np.float32)
            pres = np.zeros((len(POSE_KEYPOINT_IDS),), dtype=np.float32)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                pose0 = result.pose_landmarks[0]
                for j, idx in enumerate(POSE_KEYPOINT_IDS):
                    lm = pose0[idx]
                    coords[j] = (lm.x, lm.y, lm.z)
                    vis[j] = float(getattr(lm, "visibility", 1.0))
                    pres[j] = float(getattr(lm, "presence", 1.0))

            all_coords.append(coords)
            all_vis.append(vis)
            all_pres.append(pres)
            all_t.append(t_sec)

    runtime_sec = time.perf_counter() - start_time

    if not all_coords:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    coords_arr = np.stack(all_coords, axis=0)
    vis_arr = np.stack(all_vis, axis=0)
    pres_arr = np.stack(all_pres, axis=0)
    t_arr = np.asarray(all_t, dtype=np.float64)

    return PoseSequence(
        coords=coords_arr,
        visibility=vis_arr,
        presence=pres_arr,
        timestamps=t_arr,
        runtime_sec=runtime_sec,
    )


def _bone_lengths(seq: PoseSequence, joint_a: int, joint_b: int, vis_threshold: float = 0.5):
    """Compute bone lengths over time for a pair of keypoint indices (0..3)."""
    a = seq.coords[:, joint_a, :]
    b = seq.coords[:, joint_b, :]
    va = seq.visibility[:, joint_a]
    vb = seq.visibility[:, joint_b]

    mask = (va >= vis_threshold) & (vb >= vis_threshold)
    if not np.any(mask):
        return np.array([], dtype=np.float32)
    diff = b[mask] - a[mask]
    return np.linalg.norm(diff, axis=-1)


def _smoothness_metric(seq: PoseSequence, vis_threshold: float = 0.5) -> float:
    """Simple smoothness metric: mean L2 velocity of visible joints."""
    coords = seq.coords
    t = seq.timestamps
    if coords.shape[0] < 2:
        return float("nan")

    dt = np.diff(t)
    dt[dt <= 1e-6] = 1e-6

    vel = np.diff(coords, axis=0) / dt[:, None, None]

    vis = seq.visibility[:-1] * seq.visibility[1:]
    mask = vis >= vis_threshold
    if not np.any(mask):
        return float("nan")

    vel_mag = np.linalg.norm(vel, axis=-1)
    return float(np.mean(vel_mag[mask]))


def _glitch_rate(seq: PoseSequence, threshold_scale: float = 5.0, vis_threshold: float = 0.5) -> float:
    """Estimate fraction of frames containing large, likely spurious jumps."""
    coords = seq.coords
    if coords.shape[0] < 3:
        return 0.0

    diffs = np.diff(coords, axis=0)
    vis = seq.visibility[:-1] * seq.visibility[1:]
    mask = vis >= vis_threshold
    if not np.any(mask):
        return 0.0

    step_mag = np.linalg.norm(diffs, axis=-1)
    baseline = np.median(step_mag[mask])
    if baseline <= 0:
        return 0.0

    glitches = (step_mag > threshold_scale * baseline) & mask
    glitch_frames = np.any(glitches, axis=1)
    return float(np.mean(glitch_frames.astype(np.float32)))


def _summarize_sequence_metrics(seq: PoseSequence) -> dict:
    """Compute a dictionary of scalar metrics for one sequence."""
    upper_arm = _bone_lengths(seq, 0, 1)  # shoulder-elbow
    forearm = _bone_lengths(seq, 1, 2)  # elbow-wrist
    shoulders = _bone_lengths(seq, 0, 3)  # left-right shoulder distance

    def _stats(x: np.ndarray):
        if x.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "cv": float("nan")}
        mean = float(np.mean(x))
        std = float(np.std(x))
        cv = float(std / mean) if mean != 0 else float("nan")
        return {"mean": mean, "std": std, "cv": cv}

    smoothness = _smoothness_metric(seq)
    glitch = _glitch_rate(seq)

    T = seq.coords.shape[0]
    duration = float(seq.timestamps[-1] - seq.timestamps[0]) if T > 1 else 0.0
    effective_fps = float(T / seq.runtime_sec) if seq.runtime_sec > 0 else float("nan")

    return {
        "frames": T,
        "duration_sec": duration,
        "runtime_sec": float(seq.runtime_sec),
        "processing_fps": effective_fps,
        "upper_arm": _stats(upper_arm),
        "forearm": _stats(forearm),
        "shoulders": _stats(shoulders),
        "smoothness": smoothness,
        "glitch_rate": glitch,
    }


def _ensure_results_dir(project_root: Path) -> Path:
    """Create and return the directory for pose model result plots."""
    outdir = project_root / "tests" / "pose_model_results"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _plot_motion_results(
    motion: str,
    sequences: dict[str, PoseSequence],
    metrics: dict[str, dict],
    outdir: Path,
) -> None:
    """Generate and save plots comparing models for a single motion."""
    if not sequences:
        return

    # Consistent model order: lite (left), full (middle), heavy (right)
    preferred_order = ["lite", "full", "heavy"]
    models = [m for m in preferred_order if m in sequences]

    # 1) Bone length time-series (frame index on x-axis)
    segments = [
        ("upper_arm", 0, 1),
        ("forearm", 1, 2),
        ("shoulders", 0, 3),
    ]

    fig, axes = plt.subplots(len(segments), 1, figsize=(10, 8), sharex=True)
    if len(segments) == 1:
        axes = [axes]

    for ax, (seg_name, i, j) in zip(axes, segments):
        for model in models:
            seq = sequences[model]
            lengths = _bone_lengths(seq, i, j)
            if lengths.size == 0:
                continue
            ax.plot(lengths, label=model)
        ax.set_ylabel(f"{seg_name} length")
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"Bone lengths over time – {motion}")
    axes[-1].set_xlabel("Frame index (visible frames)")
    axes[0].legend()

    fig.tight_layout()
    fig.savefig(outdir / f"{motion}_bone_lengths.png", dpi=150)
    plt.close(fig)

    # 2) Bar plot for smoothness, glitch rate, runtime, processing FPS
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    x = np.arange(len(models))

    smooth_vals = [metrics[m]["smoothness"] for m in models]
    glitch_vals = [metrics[m]["glitch_rate"] for m in models]
    runtime_vals = [metrics[m]["runtime_sec"] for m in models]
    fps_vals = [metrics[m]["processing_fps"] for m in models]

    axes[0].bar(x, smooth_vals)
    axes[0].set_title("Smoothness (lower is better)")
    axes[0].set_xticks(x, models)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, glitch_vals)
    axes[1].set_title("Glitch rate (fraction of frames)")
    axes[1].set_xticks(x, models)
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, runtime_vals)
    axes[2].set_title("Total runtime [s]")
    axes[2].set_xticks(x, models)
    axes[2].grid(True, axis="y", alpha=0.3)

    axes[3].bar(x, fps_vals)
    axes[3].set_title("Processing FPS")
    axes[3].set_xticks(x, models)
    axes[3].grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Pose model metrics – {motion}")
    fig.tight_layout()
    fig.savefig(outdir / f"{motion}_metrics.png", dpi=150)
    plt.close(fig)


def _find_trial_videos(project_root: Path) -> dict[str, Path]:
    """Return mapping motion -> video path for subject_01/*/trial_001."""
    videos: dict[str, Path] = {}
    raw_root = project_root / "test_data" / "raw" / "subject_01"
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data root not found: {raw_root}")

    for motion_dir in sorted(raw_root.iterdir()):
        if not motion_dir.is_dir():
            continue
        trial_dir = motion_dir / "trial_001"
        if not trial_dir.exists():
            continue
        for mp4 in trial_dir.glob("*.mp4"):
            videos[motion_dir.name] = mp4
            break
    return videos


def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = _ensure_results_dir(project_root)

    model_paths = {
        "lite": project_root / "capture" / "pose_landmarker_lite.task",
        "full": project_root / "capture" / "pose_landmarker_full.task",
        "heavy": project_root / "capture" / "pose_landmarker_heavy.task",
    }

    videos = _find_trial_videos(project_root)
    if not videos:
        raise RuntimeError("No trial_001 videos found under test_data/raw/subject_01/*/trial_001")

    print("Comparing pose models on the following videos (subject_01, trial_001):")
    for motion, path in videos.items():
        print(f"  - {motion}: {path}")

    results: dict[str, dict[str, dict]] = {}
    sequences: dict[str, dict[str, PoseSequence]] = {}

    for motion, video_path in videos.items():
        results[motion] = {}
        sequences[motion] = {}
        print(f"\n=== Motion: {motion} ===")
        for name, model_path in model_paths.items():
            if not model_path.exists():
                print(f"  [SKIP] Model '{name}' not found at {model_path}")
                continue
            print(f"  Running model: {name} ({model_path.name})")
            seq = _run_pose_model_on_video(model_path, video_path)
            metrics = _summarize_sequence_metrics(seq)
            results[motion][name] = metrics
            sequences[motion][name] = seq

            print(
                f"    frames={metrics['frames']}, "
                f"duration={metrics['duration_sec']:.2f}s, "
                f"runtime={metrics['runtime_sec']:.2f}s, "
                f"proc_fps={metrics['processing_fps']:.1f}, "
                f"smoothness={metrics['smoothness']:.4f}, "
                f"glitch_rate={metrics['glitch_rate']:.3f}"
            )
            for segment in ("upper_arm", "forearm", "shoulders"):
                seg_stats = metrics[segment]
                print(
                    f"      {segment}: mean={seg_stats['mean']:.4f}, "
                    f"std={seg_stats['std']:.4f}, cv={seg_stats['cv']:.4f}"
                )

        # After processing all models for this motion, generate plots
        _plot_motion_results(motion, sequences[motion], results[motion], results_dir)

    print("\n=== Summary ===")
    for motion, models in results.items():
        print(f"\nMotion: {motion}")
        for name, m in models.items():
            print(
                f"  {name:5s}: runtime={m['runtime_sec']:.2f}s, "
                f"proc_fps={m['processing_fps']:.1f}, "
                f"smoothness={m['smoothness']:.4f}, "
                f"glitch_rate={m['glitch_rate']:.3f}"
            )


if __name__ == "__main__":
    main()

