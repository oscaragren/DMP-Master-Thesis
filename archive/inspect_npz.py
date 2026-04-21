"""
Inspect a single .npz file and print what's inside.

Example:
  python3 vis/inspect_npz.py --path test_data/processed/subject_03/random/trial_002/dmp_rollout_clean.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _format_scalar(x) -> str:
    try:
        # numpy scalar
        return str(np.asarray(x).item())
    except Exception:
        return str(x)


def _preview_array(arr: np.ndarray, *, preview: int) -> str:
    if arr.size == 0:
        return "(empty)"
    flat = np.ravel(arr)
    n = int(min(preview, flat.size))
    return np.array2string(flat[:n], threshold=n, edgeitems=n)


def _try_numeric_stats(arr: np.ndarray) -> dict[str, str] | None:
    # Only attempt stats for numeric arrays.
    if arr.size == 0:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    try:
        a = np.asarray(arr, dtype=np.float64)
        return {
            "min": f"{np.nanmin(a):.6g}",
            "max": f"{np.nanmax(a):.6g}",
            "mean": f"{np.nanmean(a):.6g}",
        }
    except Exception:
        return None


def inspect_npz(path: Path, *, preview: int = 8, stats: bool = True) -> None:
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Expected a .npz file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())
    #print(data["q_gen"])

    print(f"File: {path}")
    print(f"Type: .npz")
    print(f"Keys ({len(keys)}): {keys}")

    for k in keys:
        v = data[k]
        if not isinstance(v, np.ndarray):
            print(f"\n[{k}] type={type(v)} value={v}")
            continue

        print(f"\n[{k}] shape={v.shape} dtype={v.dtype}")

        if v.ndim == 0:
            print(f"  value: {_format_scalar(v)}")
            continue

        if stats:
            s = _try_numeric_stats(v)
            if s is not None:
                print(f"  stats: min={s['min']} max={s['max']} mean={s['mean']}")

        print(f"  preview (first {min(preview, v.size)} values, flattened): {_preview_array(v, preview=preview)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the contents of a single .npz file.")
    parser.add_argument("--path", type=Path, required=True, help="Path to a .npz file.")
    parser.add_argument(
        "--preview",
        type=int,
        default=8,
        help="How many flattened values to preview per array (default: 8).",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable numeric min/max/mean summary for numeric arrays.",
    )
    args = parser.parse_args()

    inspect_npz(args.path.resolve(), preview=int(args.preview), stats=not bool(args.no_stats))


if __name__ == "__main__":
    main()

