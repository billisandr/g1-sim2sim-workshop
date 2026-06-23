"""
Converts retargeted motion clips from the Hugging Face dataset
exptech/g1-moves (https://huggingface.co/datasets/exptech/g1-moves) into
MuJoCo-qpos-ready CSVs for sim/g1_mujoco_liveknobs.py's motion-clip
playback mode.

The dataset's retarget/<clip>.csv files are 36 columns: 3 root position +
4 root quaternion in **xyzw** order + 29 joint angles (Unitree G1, 29-DOF,
mode 15 — see assets/g1_description/g1_29dof.xml, which uses the identical
joint order, verified joint-for-joint against the dataset's documented
table). MuJoCo's qpos free-joint convention is **wxyz** — load this data
as qpos without reordering and the rotation comes out scrambled. This
module does that one reorder and nothing else; the position and joint
angle columns are already in the right order/units (meters, radians).

Each clip lives at:
    https://huggingface.co/datasets/exptech/g1-moves/resolve/main/<category>/<clip>/retarget/<clip>.csv
where <category> is "dance", "karate", or "bonus".
"""

import os

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

DATASET_BASE_URL = "https://huggingface.co/datasets/exptech/g1-moves/resolve/main"
SKELETON_XML = "assets/g1_description/g1_29dof.xml"


def convert_qpos_xyzw_to_wxyz(raw):
    """raw: (N, 36) array, cols 3:7 are a quaternion in xyzw order.
    Returns a new (N, 36) array with cols 3:7 reordered to wxyz — the only
    difference from the input; everything else passes through unchanged."""
    if raw.ndim != 2 or raw.shape[1] != 36:
        raise ValueError(f"Expected (N, 36) shape (3 pos + 4 quat + 29 joints), got {raw.shape}")
    converted = raw.copy()
    x, y, z, w = raw[:, 3], raw[:, 4], raw[:, 5], raw[:, 6]
    converted[:, 3] = w
    converted[:, 4] = x
    converted[:, 5] = y
    converted[:, 6] = z
    return converted


def convert_file(raw_csv_path, output_csv_path):
    """Read a raw g1-moves retarget CSV, reorder the quaternion, and write
    the result. Returns the output path."""
    raw = np.loadtxt(raw_csv_path, delimiter=",")
    converted = convert_qpos_xyzw_to_wxyz(raw)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    np.savetxt(output_csv_path, converted, delimiter=",")
    return output_csv_path


def download_and_convert(clip_name, category, output_dir=None):
    """Download <category>/<clip_name>/retarget/<clip_name>.csv from the
    g1-moves dataset and convert it in one step. Returns the output path."""
    import urllib.request

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "motions", "pre_staged")

    url = f"{DATASET_BASE_URL}/{category}/{clip_name}/retarget/{clip_name}.csv"
    raw_path = os.path.join(output_dir, f"_raw_{clip_name}.csv")
    output_path = os.path.join(output_dir, f"g1_moves_{clip_name}.csv")

    os.makedirs(output_dir, exist_ok=True)
    urllib.request.urlretrieve(url, raw_path)
    convert_file(raw_path, output_path)
    os.remove(raw_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("clip_name", help="e.g. M_ShortMove12")
    parser.add_argument("category", choices=["dance", "karate", "bonus"])
    args = parser.parse_args()

    path = download_and_convert(args.clip_name, args.category)
    print(f"Converted: {path}")
    print(f"Skeleton: {SKELETON_XML}")
