"""
Headless smoke test for the G1 live-knobs workshop.

No display, no robot needed. Loads the pre-staged policy and XML, steps the
sim for N ticks under a fixed forward-walk command, and asserts the torso
stays above a fall threshold. This is the only piece of the workshop that's
runtime-verifiable without WSLg/a display (see PLAN.md §5).

Run from WSL2:
    source /root/venvs/kimodo/bin/activate
    python3 tests/smoke_test.py
"""

import os
import sys

import mujoco
import numpy as np
import torch
import yaml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from sim.g1_mujoco_liveknobs import get_gravity_orientation, pd_control  # noqa: E402

NUM_TICKS = 1000
FALL_HEIGHT_THRESHOLD = 0.4  # metres; pelvis z below this means it fell over


def run_smoke_test():
    config_path = os.path.join(PROJECT_ROOT, "config", "g1_liveknobs.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pcfg = cfg["policy"]
    xml_path = os.path.join(PROJECT_ROOT, pcfg["xml_path"])
    policy_path = os.path.join(PROJECT_ROOT, pcfg["default_policy_path"])

    simulation_dt = pcfg["simulation_dt"]
    control_decimation = pcfg["control_decimation"]
    kps = np.array(pcfg["kps"], dtype=np.float32)
    kds = np.array(pcfg["kds"], dtype=np.float32)
    default_angles = np.array(pcfg["default_angles"], dtype=np.float32)
    ang_vel_scale = pcfg["ang_vel_scale"]
    dof_pos_scale = pcfg["dof_pos_scale"]
    dof_vel_scale = pcfg["dof_vel_scale"]
    action_scale = pcfg["action_scale"]
    cmd_scale = np.array(pcfg["cmd_scale"], dtype=np.float32)
    num_actions = pcfg["num_actions"]
    num_obs = pcfg["num_obs"]
    cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)

    min_pelvis_z = float("inf")

    for counter in range(NUM_TICKS):
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        pelvis_z = d.qpos[2]
        min_pelvis_z = min(min_pelvis_z, pelvis_z)

        if counter % control_decimation == 0:
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj = (qj - default_angles) * dof_pos_scale
            dqj = dqj * dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * ang_vel_scale

            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9: 9 + num_actions] = qj
            obs[9 + num_actions: 9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions: 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions: 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)

            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

    print(f"Ran {NUM_TICKS} ticks. Minimum pelvis height: {min_pelvis_z:.3f} m "
          f"(fall threshold: {FALL_HEIGHT_THRESHOLD} m)")

    assert min_pelvis_z > FALL_HEIGHT_THRESHOLD, (
        f"Robot fell: minimum pelvis height {min_pelvis_z:.3f} m did not stay "
        f"above the {FALL_HEIGHT_THRESHOLD} m threshold over {NUM_TICKS} ticks."
    )
    print("PASS: robot stayed upright.")


if __name__ == "__main__":
    run_smoke_test()
