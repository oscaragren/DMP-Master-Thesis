# DMP Module Documentation

This folder contains the core Dynamic Movement Primitive (DMP) implementation used by the project.

Main file:

- `dmp/dmp.py`

Supporting file:

- `dmp/integration.py` (RK4 integrator utility used by `rollout_rk4`)

## Overview

The implementation learns a forcing-term model from demonstrated joint trajectories and then rolls out generated trajectories with configurable start and goal states.

The DMP state follows:

- Canonical phase: `x(t) = exp(-alpha_canonical * t / tau)`
- Transformation dynamics per joint:
  - `ddq = (alpha_z * beta_z * (g - q) - alpha_z * dq + (g - q0) * f(x)) / tau^2`

where `f(x)` is represented by normalized RBFs and learned weights.

## Core API

### `fit(...) -> DMPModel`

Learns DMP weights from one or more demos (`(T, n_joints)` each).

Important implementation details (current behavior):

- **Regression features from normalized RBFs**:
  - `phi = normalized_rbf(x)`
- **RBF widths from neighboring center differences**:
  - widths are built from `np.diff(centers)` with the final value repeated
- **Ridge regularized solve**:
  - solves `(Phi^T Phi + lambda I) w = Phi^T f_target`
- **Savitzky-Golay derivatives in fit path**:
  - `dq` and `ddq` for target forcing are computed with Savitzky-Golay filtering
- **Small-scale protection for `(g - q0)`**:
  - if `abs(g - q0) < diff_g_q0_eps`, scaling falls back to `1.0`

### `rollout_simple(...) -> np.ndarray`

Euler rollout of the learned model from user-provided:

- `q0` (initial position)
- `g` (goal position)
- `tau`, `dt`

Forcing term uses phase scaling during rollout:

- `f = x * (psi_norm dot w)`

### `rollout_rk4(...) -> np.ndarray`

RK4 rollout variant with the same DMP dynamics and forcing definition.

## Helper Functions in `dmp.py`

- `canonical_phase(...)`: canonical `x(t)` computation
- `estimate_derivatives(...)`: derivative estimation (`gradient` or `savgol`)

Forcing-fit diagnostic helpers are now located in `vis/plotting.py`:
- `forcing_target_from_trajectory(...)`
- `forcing_fit_from_phase(...)`

## Typical Usage Pattern

1. Prepare demo trajectory `q_demo` with shape `(T, n_joints)` (radians).
2. Choose `tau` and `dt` (often `tau=1.0`, `dt=tau/(T-1)`).
3. Fit:
  - `model = fit([...], ...)`
4. Roll out:
  - `q_gen = rollout_simple(model, q0, g, tau, dt)`

## Notes

- This module assumes demos passed to `fit` have consistent `(T, n_joints)` shape.
- If you are comparing forcing-target diagnostics, make sure derivative method/settings match your analysis script.
- Current debug prints in `fit` include `q0`, `g`, `g-q0`, `|g-q0|`, and derivative magnitude summaries for quick spike diagnosis.

