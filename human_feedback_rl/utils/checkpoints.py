"""
Checkpoint management utilities for the RLHF training pipeline.

Moved from scripts/train_christiano.py.
"""

import os


def keep_latest_checkpoints(checkpoint_dir: str, keep: int = 2) -> None:
    """Delete old reward predictor checkpoints, keeping only the `keep` most recent."""
    if not os.path.exists(checkpoint_dir):
        return
    files = sorted(
        [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("reward_predictor_") and f.endswith(".pt")
        ],
        key=lambda f: int(f[len("reward_predictor_") : -len(".pt")]),
    )
    for old_file in files[:-keep]:
        os.remove(os.path.join(checkpoint_dir, old_file))


def drain_demo_pipe(demo_pipe, demo_db) -> int:
    """Drain all pending demo pairs from demo_pipe into demo_db. Returns count added."""
    added = 0
    while True:
        try:
            exp_frames, ag_frames, pref = demo_pipe.get_nowait()
            demo_db.append(exp_frames, ag_frames, pref)
            added += 1
        except Exception:
            break
    return added
