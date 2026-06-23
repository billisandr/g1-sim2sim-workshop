"""
Headless smoke test for the "motion_tracking" sim-loop mode
(sim/motion_tracking.py). Same convention as tests/smoke_test.py: no
display, real mujoco.MjModel/MjData physics, steps the full reference
clip once and asserts the torso doesn't collapse to the floor.

Unlike smoke_test.py's fixed forward-walk command, this drives the actual
closed loop sim/g1_mujoco_liveknobs.py's run_motion_tracking_mode() uses:
each control tick reads the robot's real MuJoCo state (torso world pose,
joint pos/vel) via mj_name2id/d.xpos/d.xquat, builds the 160-dim obs, and
feeds the tracking policy's own output back as next-tick's target — this
is what actually exercises the obs-composition decoded in
sim/motion_tracking.py's docstring, not just shape-checking with frozen
synthetic state.

Run from WSL2:
    source /root/venvs/kimodo/bin/activate
    python3 tests/smoke_test_motion_tracking.py
"""

import os
import sys

import mujoco
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sim"))

import motion_tracking as mt  # noqa: E402
from sim.g1_mujoco_liveknobs import pd_control  # noqa: E402

TRACKING_DIR = os.path.join(PROJECT_ROOT, "motions", "pre_staged", "tracking", "M_ShortMove12")
XML_PATH = os.path.join(PROJECT_ROOT, "assets", "g1_description", "g1_29dof.xml")
SIMULATION_DT = 0.005
CONTROL_DECIMATION = 4
FALL_HEIGHT_THRESHOLD = 0.4  # metres; pelvis z below this means it fell over


def run_smoke_test():
    actor = mt.TrackingActor(os.path.join(TRACKING_DIR, "policy.pt"))
    ref = mt.ReferenceMotion(os.path.join(TRACKING_DIR, "motion.npz"))

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    m.opt.timestep = SIMULATION_DT
    # g1_29dof.xml has no <option>, so MuJoCo defaults to explicit Euler —
    # unstable for these high-stiffness PD gains at this timestep. mjlab's
    # own env.yaml trained against integrator: implicitfast; match it.
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.dof_armature[6:35] = mt.ARMATURE
    torso_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    assert torso_body_id > 0, "torso_link not found in g1_29dof.xml"

    default_pos = mt.DEFAULT_JOINT_POS
    root_pos0, root_quat0 = ref.root_pose(0)
    d.qpos[0:3] = root_pos0
    d.qpos[3:7] = root_quat0
    d.qpos[7:36] = ref.joint_pos[0]  # consistent with root_pos0 (same FK frame)
    mujoco.mj_forward(m, d)

    last_action = np.zeros(mt.NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = ref.joint_pos[0].copy()
    frame_idx = 0
    min_pelvis_z = float("inf")
    num_ticks = ref.num_frames  # one full pass through the reference clip

    for counter in range(num_ticks * CONTROL_DECIMATION):
        tau = pd_control(target_dof_pos, d.qpos[7:36], mt.KP, np.zeros_like(mt.KD), d.qvel[6:35], mt.KD)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        min_pelvis_z = min(min_pelvis_z, d.qpos[2])

        if (counter + 1) % CONTROL_DECIMATION == 0:
            quat = d.qpos[3:7]
            base_lin_vel = mt.quat_apply(mt.quat_inv(quat), d.qvel[0:3])
            base_ang_vel = d.qvel[3:6]
            joint_pos = d.qpos[7:36].copy()
            joint_vel = d.qvel[6:35].copy()
            robot_anchor_pos = d.xpos[torso_body_id].copy()
            robot_anchor_quat = d.xquat[torso_body_id].copy()

            obs = mt.build_obs(
                ref, frame_idx, robot_anchor_pos, robot_anchor_quat,
                base_lin_vel, base_ang_vel, joint_pos, joint_vel, last_action,
            )
            last_action = actor.act(obs)
            target_dof_pos = last_action * mt.ACTION_SCALE + default_pos
            frame_idx = (frame_idx + 1) % ref.num_frames

    print(f"Ran {num_ticks} control ticks ({num_ticks * CONTROL_DECIMATION} physics steps) "
          f"through clip M_ShortMove12. Minimum pelvis height: {min_pelvis_z:.3f} m "
          f"(fall threshold: {FALL_HEIGHT_THRESHOLD} m)")

    assert min_pelvis_z > FALL_HEIGHT_THRESHOLD, (
        f"Robot fell: minimum pelvis height {min_pelvis_z:.3f} m did not stay "
        f"above the {FALL_HEIGHT_THRESHOLD} m threshold over the clip."
    )
    print("PASS: robot stayed upright through the tracking routine.")


if __name__ == "__main__":
    run_smoke_test()
