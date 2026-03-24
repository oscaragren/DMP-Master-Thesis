# Pipeline Documentation

This document describes the current end-to-end pipeline implemented in `run_full_pipeline.py`.

## Scope

The script runs a sweep for a single trial and compares:

- RAW trajectory path (no keypoint cleaning)
- CLEAN trajectory paths (joint-angle cleaning by default, keypoint cleaning optional)

For each path, it generates angle plots, DMP fit/rollout plots, and `.npz` artifacts.

## Entry Point

- Main script: `run_full_pipeline.py`
- Core function: `run_full_pipeline(trial_dir, clean_cutoff_hz, clean_target_dt, filter_orders, n_basis_list)`

## Inputs Per Trial

Required in the trial directory:

- `left_arm_seq_camera.npy` with shape `(T, 4, 3)`
- `left_arm_t.npy` with shape `(T,)` (optional fallback to index time)

Optional:

- `meta.json` (used for plot metadata/titles)

## High-Level Flow

1. Load raw keypoint sequence and timestamps.
2. Plot RAW keypoint trajectory (3D movement + position over time).
3. Convert RAW keypoints to joint angles (radians).
4. Save/plot RAW angles.
4. For each cleaning filter order:
   - By default: clean in joint-angle space.
   - Optional (`--use-keypoint-cleaning`): clean keypoints first, then map to angles.
   - Save/plot cleaned angles.
5. Build RAW vs CLEAN angle overlay grid.
6. For each DMP basis size:
   - Fit/roll out DMP for RAW.
   - Fit/roll out DMP for each CLEAN variant.
   - Save DMP trajectories and model metadata (`.npz`).
   - Plot RAW vs CLEAN DMP overlays.
7. Create per-joint order-vs-basis grid plots.
8. Zip all generated outputs from this run.

## Detailed Stages

### 1) RAW sequence loading

- Function: `_load_raw_seq_t(...)`
- Validates shape `(T, 4, 3)` for sequence.
- Ensures time vector length matches `T`.
- Normalizes timestamps to start at `0` (`t = t - t[0]`).

### 2) RAW mapping to joint space

- Function: `mapping.sequence_to_angles.sequence_to_angles_rad(...)`
- Output:
  - `elbow_rad`: `(T,)`
  - `shoulder_rad`: `(T, 3)`

### 3) CLEAN sweep (default: joint space)

Default path:

- Function: `kinematics.clean_angles.clean_angles_trajectory(...)`
- Cleaning is done after RAW keypoints are mapped to joint angles.

Optional keypoint-space path (enable `--use-keypoint-cleaning`):

- Function: `capture.clean_keypoints.run_clean_left_arm_sequence(...)`
- Cleaning is done in keypoint space (`x, y, z`) per channel:
  - Butterworth low-pass filter
  - Optional resampling to `target_dt`
- Cleaned files written by cleaner:
  - `left_arm_seq_camera_cleaned.npy`
  - `left_arm_t_cleaned.npy`
- Cleaned sequence is then mapped to joint angles.

### 4) RAW keypoint trajectory artifact

Also generated from RAW sequence:

- `*keypoints_raw_trajectory.png`:
  - Left panel: keypoint movement in 3D (physically up convention)
  - Right panel: x/up/z positions over time

### 5) Angle artifacts

Per variant (RAW and each CLEAN order), saved as:

- `*angles_*.npz` with:
  - `elbow_rad`, `shoulder_rad`
  - `elbow_deg`, `shoulder_deg`
- `*angles_*.png` angle plots

Also generated:

- `*angles_overlay_raw_vs_clean_orders.png`

### 6) DMP stage

For each basis count in `n_basis_list`:

- Build joint demo matrix `q = [elbow, shoulder...]`.
- Drop rows with non-finite values.
- Fit with `dmp.fit(...)`.
- Roll out with `dmp.rollout_simple(...)`.

Current `dmp.fit(...)` details:

- Uses normalized RBF basis features (`phi = normalized_rbf(x)`).
- Uses width values built from neighboring center differences.
- Uses ridge regularization solve.
- Uses Savitzky-Golay derivative estimation inside fitting for forcing-target computation.

DMP outputs per variant:

- Trajectory plots (`*.png`)
- Rollout artifacts (`*.npz`) containing:
  - demo and generated trajectories in rad/deg
  - `dt`, `tau`, `n_basis`
  - DMP params (`alpha_*`, centers, widths, weights)

Also generated:

- Overlay per basis: `*dmp_overlay_raw_vs_clean_orders_n{n_basis}.png`
- Per-joint order-vs-basis grids from `vis.plotting.plot_dmp_order_basis_grids_per_joint(...)`

### 7) Bundle step

- All files generated during the current run are zipped to:
  - `*raw_clean_sweep_outputs.zip`

## Default Sweep Parameters

Defaults in `run_full_pipeline.py`:

- `filter_orders`: `[1, 2, 4, 6]`
- `n_basis_list`: `[10, 30, 60]`
- `clean_cutoff_hz`: `5.0`
- `clean_target_dt`: `0.04` seconds

## CLI Usage

Run by direct trial path:

```bash
python3 run_full_pipeline.py --path path/to/trial
```

Run by index:

```bash
python3 run_full_pipeline.py --subject 1 --motion reach --trial 1
```

Useful options:

- `--clean-cutoff-hz`
- `--clean-target-dt` (set `0` to disable resample)
- `--filter-orders 1 2 4 6`
- `--n-basis 10 30 60`
- `--use-keypoint-cleaning` (default is joint-space cleaning)
- `--data-dir`

## Output Naming Pattern

Outputs are prefixed using `vis.trial_naming.trial_prefix(trial_dir)`, then suffixed with artifact names such as:

- `angles_raw.npz`, `angles_clean_o2.npz`
- `keypoints_raw_trajectory.png`
- `angles_raw.png`, `angles_clean_o2.png`
- `dmp_rollout_raw_n30.npz`, `dmp_rollout_clean_o2_n30.npz`
- `dmp_trajectory_raw_n30.png`, `dmp_trajectory_clean_o2_n30.png`
- `raw_clean_sweep_outputs.zip`

## Notes

- Default CLEAN sweep compares joint-space cleaning effects across filter orders.
- `--use-keypoint-cleaning` lets you compare keypoint-space cleaning effects instead.
