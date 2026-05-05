import numpy as np
from dmp.dmp import DMPModel
import os
from dmp.dmp import rollout_simple_with_coupling
from mapping.retarget import JOINT_LIMITS_DEG

def get_trajectories(subject: int = 10):
    """
    Get the trajectories for the experiment.
    """

    demo = np.array([[60.0, 29.637, 7.208, -1.261], 
                    [60.0, 29.135, 7.159, -1.148], 
                    [60.0, 28.288, 7.094, -1.007], 
                    [60.0, 27.012, 7.059, -0.865], 
                    [60.0, 25.562, 7.071, -0.78], 
                    [60.0, 24.08, 7.121, -0.808], 
                    [60.0, 22.777, 7.179, -0.873], 
                    [60.0, 21.854, 7.204, -0.926], 
                    [60.0, 21.423, 7.171, -0.918], 
                    [60.0, 21.568, 7.075, -0.971], 
                    [60.0, 22.103, 6.93, -1.164], 
                    [60.0, 23.036, 6.76, -1.426], 
                    [60.0, 24.282, 6.586, -1.745], 
                    [60.0, 25.6, 6.417, -1.996], 
                    [60.0, 26.653, 6.253, -1.956], 
                    [60.0, 27.333, 6.095, -1.514], 
                    [59.672, 27.663, 5.954, -0.751], 
                    [58.627, 27.885, 5.851, 0.13], 
                    [56.759, 28.002, 5.823, 0.888], 
                    [54.009, 28.328, 5.897, 1.178], 
                    [50.497, 28.964, 6.092, 0.95], 
                    [46.599, 29.551, 6.391, 0.207], 
                    [42.712, 30.898, 6.754, -0.89], 
                    [39.372, 32.736, 7.134, -2.056], 
                    [37.198, 34.523, 7.47, -3.03], 
                    [36.344, 36.092, 7.681, -3.605], 
                    [36.864, 37.512, 7.713, -3.752], 
                    [38.568, 39.015, 7.561, -3.525], 
                    [41.054, 40.681, 7.271, -2.983], 
                    [43.924, 42.53, 6.893, -2.017], 
                    [46.63, 44.767, 6.49, -0.463], 
                    [48.968, 47.285, 6.105, 1.995], 
                    [50.975, 49.653, 5.778, 5.398], 
                    [52.53, 51.548, 5.487, 9.803], 
                    [53.354, 52.585, 5.248, 14.734], 
                    [53.456, 52.495, 5.085, 19.407], 
                    [52.915, 51.24, 5.0, 23.268], 
                    [52.011, 48.953, 5.0, 25.961], 
                    [50.845, 46.027, 5.0, 27.044], 
                    [49.746, 42.778, 5.0, 26.282], 
                    [48.873, 39.548, 5.003, 23.997], 
                    [48.298, 36.696, 5.023, 20.994], 
                    [47.901, 34.681, 5.035, 18.222], 
                    [47.689, 33.568, 5.044, 16.211], 
                    [47.755, 33.242, 5.049, 15.157], 
                    [48.002, 33.532, 5.048, 15.025], 
                    [48.269, 34.446, 5.04, 15.585], 
                    [48.507, 35.802, 5.02, 16.627], 
                    [48.813, 37.378, 5.0, 17.921], 
                    [49.42, 39.045, 5.0, 19.343], 
                    [50.429, 40.538, 5.031, 20.783], 
                    [52.019, 41.558, 5.186, 21.836], 
                    [54.159, 42.05, 5.476, 22.045], 
                    [56.496, 42.336, 5.908, 20.749], 
                    [58.557, 42.782, 6.456, 17.303], 
                    [59.896, 43.982, 7.078, 11.803], 
                    [60.0, 45.875, 7.759, 4.14], 
                    [59.696, 48.409, 8.402, -4.75], 
                    [58.418, 51.172, 8.844, -13.05], 
                    [56.808, 53.198, 9.01, -19.592], 
                    [55.291, 53.185, 8.86, -23.499], 
                    [54.419, 49.585, 8.367, -22.883], 
                    [54.521, 40.963, 7.462, -16.212], 
                    [55.997, 25.993, 6.086, -2.747], 
                    [59.25, 3.449, 5.0, 18.025]])

    
    coupling_dir = os.path.join(os.path.dirname(__file__), "coupling")

    # Load the saved model parameters and reconstruct a proper DMPModel object
    model_npz = np.load(os.path.join(coupling_dir, "dmp_model.npz"))
    cw_npz = np.load(os.path.join(coupling_dir, f"S{subject:02d}_curv_weights_mean.npz"))

    dmp_model = DMPModel(
        weights=np.asarray(model_npz["weights"], dtype=float),
        centers=np.asarray(model_npz["centers"], dtype=float),
        widths=np.asarray(model_npz["widths"], dtype=float),
        alpha_canonical=float(np.atleast_1d(model_npz["alpha_canonical"])[0]),
        alpha_transformation=float(np.atleast_1d(model_npz["alpha_transformation"])[0]),
        beta_transformation=float(np.atleast_1d(model_npz["beta_transformation"])[0]),
        tau=float(np.atleast_1d(model_npz["tau"])[0]),
        n_joints=int(np.asarray(model_npz["weights"]).shape[0]),
        curvature_weights=np.asarray(cw_npz["curvature_weights"], dtype=float),
    )

    # Roll out with and without curvature weights
    q0 = demo[0]
    g = demo[-1]
    tau = 1.0
    dt = 1.0 / (demo.shape[0] - 1)
    personalized_q = rollout_simple_with_coupling(dmp_model, q0, g, tau, dt)
    personalized_q = np.clip(personalized_q, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])
    return demo, personalized_q


if __name__ == "__main__":

    # Print arrays without scientific notation (no 1.23e+04 formatting)
    np.set_printoptions(suppress=True, floatmode="fixed", precision=4)

    # Get the DMP model and curvature weights for the subject
    q1, personalized_q1 = get_trajectories(subject=1)
    q2, personalized_q2 = get_trajectories(subject=2)
    q3, personalized_q3 = get_trajectories(subject=3)
    q4, personalized_q4 = get_trajectories(subject=4)
    q5, personalized_q5 = get_trajectories(subject=5)
    q6, personalized_q6 = get_trajectories(subject=6)
    q7, personalized_q7 = get_trajectories(subject=7)
    q9, personalized_q9 = get_trajectories(subject=9)
    q10, personalized_q10 = get_trajectories(subject=10)

    # Comma-separated (CSV-like) formatting for nicer copy/paste
    print(np.array2string(personalized_q1, separator=", "))
    
 