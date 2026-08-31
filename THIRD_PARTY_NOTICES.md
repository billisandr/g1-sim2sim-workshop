# Third-Party Notices

This workshop's own code (Python, YAML, shell scripts, documentation) is MIT
licensed (see LICENSE). It also bundles or depends on the following
third-party material, under its own separate terms.

## Unitree G1 robot description (URDF/MJCF and meshes)

`assets/g1_description/` and `assets/g1skel34_kimodo/` contain the G1's mesh
and kinematic description, sourced from Unitree Robotics' published
`unitree_ros`/`unitree_rl_gym` description packages. Unitree publishes these
under BSD-3-Clause. Verify current terms against Unitree's own repositories
before reusing the assets outside this workshop:

- https://github.com/unitreerobotics/unitree_ros
- https://github.com/unitreerobotics/unitree_rl_gym

## `unitree_rl_gym` sim loop and deployment reference

The MuJoCo control loop in `sim/g1_mujoco_liveknobs.py` is adapted from
`unitree_rl_gym`'s `deploy/deploy_mujoco/deploy_mujoco.py` reference
implementation (Unitree Robotics, BSD-3-Clause):
https://github.com/unitreerobotics/unitree_rl_gym

## `exptech/g1-moves` motion clips

The pre-staged clips named `g1_moves_*.csv` under `motions/pre_staged/`, and
the tracking policy under `motions/pre_staged/tracking/`, are converted from
the `exptech/g1-moves` dataset on Hugging Face:
https://huggingface.co/datasets/exptech/g1-moves

Licensed CC-BY-4.0. Attribution: motion capture data by exptech, captured
with MOVIN TRACIN markerless motion capture and retargeted onto the Unitree
G1 via [movin_sdk_python](https://github.com/MOVIN3D/movin_sdk_python). See
[docs/HUGGINGFACE_GUIDE.md](docs/HUGGINGFACE_GUIDE.md) for how these were
selected, converted, and verified.

## NVIDIA Kimodo

`kimodo_bridge/generate_motion.py` wraps a call to Kimodo (NVIDIA's
text-to-motion model). Kimodo itself is not bundled in this repository, only
the thin wrapper script that calls it. Kimodo's own license and access terms
apply and are set by NVIDIA, not by this workshop.
