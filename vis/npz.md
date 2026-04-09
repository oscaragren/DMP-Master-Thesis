# What is inside NPZ files

## q_demo
The **demonstration trajectory** used to fit the DMP. 
Shape (T, D) where T is timestamps and D is number of joints/DoF (4).
Stored in rad.

## q_gen
The **generated rollout trajectory** produced by the DMP after fitting.
Shape (T, D).
Stored in degrees.

## q_gen_rad
The **generated rollout trajectory** produced by the DMP after fitting.
Shape (T, D).
Stored in radians.

## t
The **time vector** for the trail samples.
Shape (T,)
Corresponds to when each row of q_demo/q_gen occurs.
Not shifted to 0 as is, starts at around 5.55 seconds.

## dt
The **time step** between samples.
dt = 1/(T-1) since tau = 1.

## q0
The **start joint configuration** for the rollout.
Shape (D,).

## qT
The **goal/end joint configuration** for the rollout.
Shape (D,)
