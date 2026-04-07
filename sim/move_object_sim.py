"""
Spawn a simple object in PyBullet and move it along a chosen axis.

Usage (from project root):

    python3 sim/move_object_sim.py

Options:
    --shape cube|sphere
    --axis x|y|z
    --speed 0.2
    --amplitude 0.5
"""

from __future__ import annotations

import argparse
import math
import time

import pybullet as p


def main() -> None:
    parser = argparse.ArgumentParser(description="Spawn an object and move it along a chosen axis in PyBullet.")
    parser.add_argument(
        "--shape",
        type=str,
        choices=["cube", "sphere"],
        default="cube",
        help="Object shape to spawn (default: cube).",
    )
    parser.add_argument(
        "--axis",
        type=str,
        choices=["x", "y", "z"],
        default="x",
        help="Axis to move along (default: x).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.2,
        help="Speed scaling for the motion (default: 0.2).",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="Motion amplitude along the chosen axis in meters (default: 0.5).",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=0.2,
        help="Height above world origin (meters) (default: 0.2).",
    )
    args = parser.parse_args()

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    if args.shape == "cube":
        half_extents = [0.05, 0.05, 0.05]
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.6, 0.9, 1.0])
    else:
        radius = 0.06
        col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0.9, 0.4, 0.2, 1.0])

    body_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0.0, 0.0, float(args.z)],
    )

    dt = 1.0 / 240.0
    t0 = time.time()
    try:
        while True:
            if not p.isConnected():
                break
            t = time.time() - t0
            d = float(args.amplitude) * math.sin(2.0 * math.pi * float(args.speed) * t)

            pos = [0.0, 0.0, float(args.z)]
            if args.axis == "x":
                pos[0] = d
            elif args.axis == "y":
                pos[1] = d
            else:
                pos[2] = float(args.z) + d

            p.resetBasePositionAndOrientation(body_id, pos, [0.0, 0.0, 0.0, 1.0])
            p.stepSimulation()
            time.sleep(dt)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()

