from __future__ import annotations

import re
from pathlib import Path


_SUBJECT_RE = re.compile(r"^subject_(\d+)$")
_TRIAL_RE = re.compile(r"^trial_(\d+)$")


def trial_prefix(trial_dir: Path, motion_code: str | None = None) -> str:
    """
    Return a prefix like '01L002_' for a processed trial directory.

    Expected processed layout:
      .../subject_01/lift/trial_002

    Rules:
    - subject: 2 digits
    - motion:  first letter uppercased (lift -> 'L', reach -> 'R', ...)
    - trial:   3 digits
    """
    trial_dir = Path(trial_dir)

    motion = trial_dir.parent.name
    subject_dir = trial_dir.parent.parent.name
    trial_dir_name = trial_dir.name

    m_subj = _SUBJECT_RE.match(subject_dir)
    m_trial = _TRIAL_RE.match(trial_dir_name)
    if not m_subj or not m_trial:
        raise ValueError(
            f"Could not derive subject/trial from path '{trial_dir}'. "
            "Expected .../subject_XX/<motion>/trial_YYY"
        )

    subject = int(m_subj.group(1))
    trial = int(m_trial.group(1))
    code = motion_code or (motion[:1].upper() if motion else "X")
    return f"{subject:02d}{code}{trial:03d}_"


def prefixed_filename(trial_dir: Path, filename: str, motion_code: str | None = None) -> str:
    """Prefix a filename with the trial code, e.g. angles.png -> 01L002_angles.png"""
    return f"{trial_prefix(trial_dir, motion_code=motion_code)}{filename}"

