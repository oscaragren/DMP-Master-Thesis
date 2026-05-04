import numpy as np
from dmp.dmp import DMPModel
import os
from dmp.dmp import rollout_simple, rollout_simple_with_coupling
from kinematics.simple_kinematics import get_angles
from kinematics.clean_angles import _lowpass_angles
from analyze_data import _load_raw_seq_t, _load_meta, _interpolate_nan
from pathlib import Path


def get_trajectories():
    """
    Get the trajectories for the experiment.
    """

    subject = 10
    trial = 6
    motion = "move_cup"
    raw_session_dir = os.path.join(os.path.dirname(__file__), "data", "raw", f"subject_{subject:02d}", motion)
    processed_session_dir = os.path.join(os.path.dirname(__file__), "data", "processed", f"subject_{subject:02d}", motion)

    seq, t = _load_raw_seq_t(Path(raw_session_dir) / f"trial_{(trial):03d}")
    # 4) Convert the sequence to angles and interpolate NaN values
    angles = _interpolate_nan(get_angles(seq))
    # Maybe add low_pass filter here
    angles = _lowpass_angles(angles, fps=25.0, cutoff_hz=5.0, order=2)

    # Load the saved model parameters and reconstruct a proper DMPModel object
    model_npz = np.load(os.path.join(processed_session_dir, f"trial_{(trial):03d}", "dmp_model.npz"))
    base_model = DMPModel(
        weights=np.asarray(model_npz["weights"], dtype=float),
        centers=np.asarray(model_npz["centers"], dtype=float),
        widths=np.asarray(model_npz["widths"], dtype=float),
        alpha_canonical=float(np.atleast_1d(model_npz["alpha_canonical"])[0]),
        alpha_transformation=float(np.atleast_1d(model_npz["alpha_transformation"])[0]),
        beta_transformation=float(np.atleast_1d(model_npz["beta_transformation"])[0]),
        tau=float(np.atleast_1d(model_npz["tau"])[0]),
        n_joints=int(np.asarray(model_npz["weights"]).shape[0]),
        curvature_weights=np.zeros_like(np.asarray(model_npz["weights"], dtype=float)),
    )

    # Personalized (subject-averaged) curvature weights saved by couple.py
    cw_npz = np.load(os.path.join(processed_session_dir, "curvature_weights_mean.npz"))
    personalized_model = DMPModel(
        weights=base_model.weights,
        centers=base_model.centers,
        widths=base_model.widths,
        alpha_canonical=base_model.alpha_canonical,
        alpha_transformation=base_model.alpha_transformation,
        beta_transformation=base_model.beta_transformation,
        tau=base_model.tau,
        n_joints=base_model.n_joints,
        curvature_weights=np.asarray(cw_npz["curvature_weights"], dtype=float),
    )

    # Roll out with and without curvature weights
    q0 = angles[0]
    g = angles[-1]
    tau = 1.0
    dt = 1.0 / (angles.shape[0] - 1)
    q = rollout_simple(base_model, q0, g, tau, dt)
    personalized_q = rollout_simple_with_coupling(personalized_model, q0, g, tau, dt)
    return q, personalized_q


if __name__ == "__main__":


    # Get the DMP model and curvature weights for the subject
    q, personalized_q = get_trajectories()    