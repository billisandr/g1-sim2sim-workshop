"""
Downloads one exptech/g1-moves clip's per-clip MOTION-TRACKING policy
(policy/<clip>_policy.pt + training/<clip>.npz) for
sim/motion_tracking.py's "motion_tracking" sim-loop mode.

This is a different pipeline stage than hf/g1_moves_convert.py, which only
ever used retarget/<clip>.csv for kinematic playback. See
sim/motion_tracking.py's module docstring and docs/HUGGINGFACE_GUIDE.md
§3.4 for why this policy needs its own sim-loop mode rather than dropping
into either existing one.

No conversion happens here — both files are used as downloaded. The only
work is fetching them into a layout sim/motion_tracking.py expects:
    motions/pre_staged/tracking/<clip>/policy.pt
    motions/pre_staged/tracking/<clip>/motion.npz
"""

import os
import urllib.request

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

DATASET_BASE_URL = "https://huggingface.co/datasets/exptech/g1-moves/resolve/main"


def download(clip_name, category, output_dir=None):
    """Download <category>/<clip_name>/policy/<clip_name>_policy.pt and
    .../training/<clip_name>.npz into motions/pre_staged/tracking/<clip_name>/.
    Returns that directory's path."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "motions", "pre_staged", "tracking", clip_name)
    os.makedirs(output_dir, exist_ok=True)

    policy_url = f"{DATASET_BASE_URL}/{category}/{clip_name}/policy/{clip_name}_policy.pt"
    motion_url = f"{DATASET_BASE_URL}/{category}/{clip_name}/training/{clip_name}.npz"

    urllib.request.urlretrieve(policy_url, os.path.join(output_dir, "policy.pt"))
    urllib.request.urlretrieve(motion_url, os.path.join(output_dir, "motion.npz"))
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("clip_name", help="e.g. M_ShortMove12")
    parser.add_argument("category", choices=["dance", "karate", "bonus"])
    args = parser.parse_args()

    path = download(args.clip_name, args.category)
    print(f"Downloaded tracking policy + reference motion to: {path}")
