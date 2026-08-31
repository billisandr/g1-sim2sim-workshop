"""
G1 live-knobs sim loop.

Adapts unitree_rl_gym's deploy/deploy_mujoco/deploy_mujoco.py loop to read
live knobs from sim/runtime_state.json every control tick instead of using
fixed constants, and adds a second playback mode for Kimodo motion clips.

Two modes, selected via runtime_state.json's "mode" field:

- "policy": an RL policy (TorchScript .pt) reacts live to vx/vy/yaw/
  pd_scale/action_scale/push knobs. This is deploy_mujoco.py's loop,
  parameterized.
- "motion_clip": a fixed Kimodo qpos trajectory (.csv) is played back
  kinematically (data.qpos[:] = frame; mj_forward(); viewer.sync()) — it
  has no velocity input to react to, so the velocity knobs do nothing in
  this mode (the UI greys them out and shows a scrub bar instead).

These two modes use different MuJoCo skeletons (different nq) — see
config/g1_liveknobs.yaml's comment on policy.xml_path vs motion_clip.xml_path.
A Kimodo clip cannot be played back on the policy's 12-leg-joint XML because
the joint counts and ordering don't match.

Run from WSL2 (needs WSLg for the passive viewer window):
    source /root/venvs/kimodo/bin/activate
    python3 sim/g1_mujoco_liveknobs.py [--config config/g1_liveknobs.yaml]
"""

import argparse
import json
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

# Needed so `import motion_tracking` resolves both when this file is run
# directly (python3 sim/g1_mujoco_liveknobs.py, sys.path[0] is already
# sim/) and when it's imported as sim.g1_mujoco_liveknobs from a script
# elsewhere (tests/smoke_test.py), where sim/ wouldn't otherwise be on
# sys.path.
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import motion_tracking  # noqa: E402


def _resolve(path):
    """Config paths are relative to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def load_runtime_state(path, defaults):
    """Best-effort read; falls back to defaults if the UI hasn't written yet
    or the file is mid-write (we tolerate a torn read by ignoring JSON errors
    for one tick — the UI writes every ~100ms so this self-corrects)."""
    try:
        with open(path) as f:
            state = json.load(f)
        merged = dict(defaults)
        merged.update(state)
        return merged
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(defaults)


DEFAULT_RUNTIME_STATE = {
    "mode": "policy",
    "vx": 0.5,
    "vy": 0.0,
    "yaw": 0.0,
    "pd_scale": 1.0,
    "action_scale_override": None,
    "push_impulse": 0.0,
    "push_requested": False,
    "motion_path": None,
    "motion_paused": False,
    "motion_scrub_frame": None,
    "tracking_dir": None,
}


def run_policy_mode(cfg, runtime_state_path):
    pcfg = cfg["policy"]

    xml_path = _resolve(pcfg["xml_path"])
    policy_path = _resolve(pcfg.get("active_policy_path", pcfg["default_policy_path"]))

    simulation_dt = pcfg["simulation_dt"]
    control_decimation = pcfg["control_decimation"]

    kps_base = np.array(pcfg["kps"], dtype=np.float32)
    kds_base = np.array(pcfg["kds"], dtype=np.float32)
    default_angles = np.array(pcfg["default_angles"], dtype=np.float32)

    ang_vel_scale = pcfg["ang_vel_scale"]
    dof_pos_scale = pcfg["dof_pos_scale"]
    dof_vel_scale = pcfg["dof_vel_scale"]
    action_scale_base = pcfg["action_scale"]
    cmd_scale = np.array(pcfg["cmd_scale"], dtype=np.float32)

    num_actions = pcfg["num_actions"]
    num_obs = pcfg["num_obs"]

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)

    counter = 0
    print(f"[g1_mujoco_liveknobs] policy mode: {policy_path}")
    print(f"[g1_mujoco_liveknobs] xml: {xml_path}")
    print(f"[g1_mujoco_liveknobs] reading live knobs from {runtime_state_path}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            state = load_runtime_state(runtime_state_path, DEFAULT_RUNTIME_STATE)
            if state.get("mode") != "policy":
                # UI switched to motion-clip mode; this loop exits and the
                # caller (main()) restarts in the other mode.
                return "mode_switch"

            pd_scale = float(state.get("pd_scale", 1.0))
            kps = kps_base * pd_scale
            kds = kds_base * pd_scale
            action_scale = float(state.get("action_scale_override") or action_scale_base)
            cmd = np.array(
                [state.get("vx", 0.0), state.get("vy", 0.0), state.get("yaw", 0.0)],
                dtype=np.float32,
            )

            if state.get("push_requested"):
                impulse = float(state.get("push_impulse", 0.0))
                # Apply as an instantaneous horizontal velocity kick to the
                # torso (free joint linear velocity, qvel[0:3]) scaled by
                # body mass so the knob's units stay "impulse-like" (N*s)
                # regardless of step rate.
                torso_mass = m.body_mass[1] if m.nbody > 1 else 1.0
                d.qvel[0] += impulse / max(torso_mass, 1e-3)
                _clear_push_request(runtime_state_path, state)

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1
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

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return "window_closed"


def _clear_push_request(runtime_state_path, state):
    """Write push_requested back to false so the impulse fires once, not
    every tick. Best-effort — if this races with a UI write, the UI's next
    write (every ~100ms) re-establishes the correct value anyway."""
    try:
        state["push_requested"] = False
        with open(runtime_state_path, "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def _force_mode(runtime_state_path, mode):
    """Best-effort: flip mode (and clear motion_path) so the UI's selector
    reflects reality instead of silently retrying a broken clip forever."""
    try:
        state = load_runtime_state(runtime_state_path, DEFAULT_RUNTIME_STATE)
        state["mode"] = mode
        state["motion_path"] = None
        with open(runtime_state_path, "w") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


def _resolve_skeleton_xml(cfg, motion_path):
    """Match motion_path's basename against motion_clip.skeleton_by_prefix
    rules (first match wins; keep the "" rule last as the fallback) and
    return (xml_path, skeleton_name). Different motion sources use
    different, non-interchangeable skeletons that happen to share the
    same nq by coincidence, so this is resolved by filename, not by nq."""
    mcfg = cfg["motion_clip"]
    basename = os.path.basename(motion_path)
    for rule in mcfg["skeleton_by_prefix"]:
        if basename.startswith(rule["prefix"]):
            skeleton_name = rule["skeleton"]
            return _resolve(mcfg["skeletons"][skeleton_name]["xml_path"]), skeleton_name
    raise ValueError(f"No skeleton_by_prefix rule matched {basename!r}")


def run_motion_clip_mode(cfg, runtime_state_path):
    mcfg = cfg["motion_clip"]
    fps = mcfg.get("playback_fps", 30)

    state = load_runtime_state(runtime_state_path, DEFAULT_RUNTIME_STATE)
    motion_path = state.get("motion_path") or _resolve(mcfg["default_motion_path"])
    motion_path = _resolve(motion_path) if not os.path.isabs(motion_path) else motion_path
    xml_path, skeleton_name = _resolve_skeleton_xml(cfg, motion_path)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    print(f"[g1_mujoco_liveknobs] motion-clip mode, skeleton: {skeleton_name}, xml: {xml_path}")

    loaded_path = None
    qpos_frames = None

    with mujoco.viewer.launch_passive(m, d) as viewer:
        frame_idx = 0
        while viewer.is_running():
            state = load_runtime_state(runtime_state_path, DEFAULT_RUNTIME_STATE)
            if state.get("mode") != "motion_clip":
                return "mode_switch"

            motion_path = state.get("motion_path") or _resolve(mcfg["default_motion_path"])
            motion_path = _resolve(motion_path) if not os.path.isabs(motion_path) else motion_path

            # A clip that needs a different skeleton than the one this
            # viewer/model was opened with can't be swapped in place —
            # MuJoCo's model is fixed for the life of a viewer context.
            # Exit and let main() re-enter fresh with the right skeleton.
            try:
                _, new_skeleton_name = _resolve_skeleton_xml(cfg, motion_path)
            except (KeyError, ValueError) as e:
                print(f"[g1_mujoco_liveknobs] {e}; falling back to policy mode.")
                _force_mode(runtime_state_path, "policy")
                return "mode_switch"
            if new_skeleton_name != skeleton_name:
                return "mode_switch"

            if motion_path != loaded_path:
                try:
                    candidate_frames = np.loadtxt(motion_path, delimiter=",")
                except OSError as e:
                    print(f"[g1_mujoco_liveknobs] motion clip {motion_path} could not be "
                          f"read ({e}); falling back to policy mode instead of crashing.")
                    _force_mode(runtime_state_path, "policy")
                    return "mode_switch"
                if candidate_frames.ndim != 2 or candidate_frames.shape[1] != m.nq:
                    got = candidate_frames.shape if candidate_frames.ndim == 2 else candidate_frames.shape
                    print(f"[g1_mujoco_liveknobs] motion clip {motion_path} has shape {got}, "
                          f"expected nq={m.nq} columns — wrong skeleton for this clip. "
                          f"Falling back to policy mode instead of crashing.")
                    _force_mode(runtime_state_path, "policy")
                    return "mode_switch"
                qpos_frames = candidate_frames
                loaded_path = motion_path
                frame_idx = 0
                print(f"[g1_mujoco_liveknobs] loaded motion clip: {motion_path} "
                      f"({qpos_frames.shape[0]} frames)")

            scrub = state.get("motion_scrub_frame")
            if scrub is not None:
                frame_idx = int(scrub) % len(qpos_frames)
            elif not state.get("motion_paused", False):
                frame_idx = (frame_idx + 1) % len(qpos_frames)

            d.qpos[:] = qpos_frames[frame_idx]
            mujoco.mj_forward(m, d)
            viewer.sync()
            time.sleep(1.0 / fps)

    return "window_closed"


def run_motion_tracking_mode(cfg, runtime_state_path):
    """Third mode: an imitation-trained tracking policy performs ONE fixed
    reference routine through a real PD-driven MuJoCo sim (reacts to pushes/
    physics, but NOT to vx/vy/yaw — see sim/motion_tracking.py's docstring
    for why this is a distinct category from both other modes)."""
    tcfg = cfg["motion_tracking"]
    xml_path = _resolve(tcfg["xml_path"])
    simulation_dt = tcfg["simulation_dt"]
    control_decimation = tcfg["control_decimation"]

    state = load_runtime_state(runtime_state_path, DEFAULT_RUNTIME_STATE)
    tracking_dir = state.get("tracking_dir") or _resolve(tcfg["default_tracking_dir"])
    tracking_dir = _resolve(tracking_dir) if not os.path.isabs(tracking_dir) else tracking_dir

    actor = motion_tracking.TrackingActor(os.path.join(tracking_dir, "policy.pt"))
    ref = motion_tracking.ReferenceMotion(os.path.join(tracking_dir, "motion.npz"))

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    # g1_29dof.xml has no <option>, so MuJoCo defaults to explicit Euler —
    # unstable for these high-stiffness PD gains at this timestep. mjlab's
    # own env.yaml trained against integrator: implicitfast; match it.
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # g1_29dof.xml also ships zero <joint> armature on every actuated
    # joint; mjlab's robot spec sets a nonzero rotor armature per joint
    # (these values). Without it the same PD gains see far less effective
    # inertia than trained against and the sim diverges within ~10 ticks.
    m.dof_armature[6:35] = motion_tracking.ARMATURE
    torso_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    default_pos = motion_tracking.DEFAULT_JOINT_POS
    kp_base = motion_tracking.KP
    kd_base = motion_tracking.KD

    # The XML has no <keyframe>, so the free joint defaults to qpos[0:7] =
    # (0,0,0, 1,0,0,0) — pelvis origin AT the floor plane, which explodes
    # the very first contact step. Spawn at the reference motion's own
    # frame-0 root pose/joint angles instead (this is what the reference
    # actually used, so it's self-consistent — spawning at this root pose
    # with a *different* joint configuration like the generic default pose
    # can clip a foot through the floor).
    root_pos0, root_quat0 = ref.root_pose(0)
    d.qpos[0:3] = root_pos0
    d.qpos[3:7] = root_quat0
    d.qpos[7:36] = ref.joint_pos[0]
    mujoco.mj_forward(m, d)

    last_action = np.zeros(motion_tracking.NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = ref.joint_pos[0].copy()
    frame_idx = 0
    counter = 0

    print(f"[g1_mujoco_liveknobs] motion-tracking mode: {tracking_dir}")
    print(f"[g1_mujoco_liveknobs] reference motion: {ref.num_frames} frames "
          f"(native {ref.fps} fps; played back 1 frame per control tick)")
    print(f"[g1_mujoco_liveknobs] xml: {xml_path}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            state = load_runtime_state(runtime_state_path, DEFAULT_RUNTIME_STATE)
            if state.get("mode") != "motion_tracking":
                return "mode_switch"

            pd_scale = float(state.get("pd_scale", 1.0))
            kp = kp_base * pd_scale
            kd = kd_base * pd_scale

            if state.get("push_requested"):
                impulse = float(state.get("push_impulse", 0.0))
                torso_mass = m.body_mass[1] if m.nbody > 1 else 1.0
                d.qvel[0] += impulse / max(torso_mass, 1e-3)
                _clear_push_request(runtime_state_path, state)

            tau = pd_control(target_dof_pos, d.qpos[7:36], kp, np.zeros_like(kd), d.qvel[6:35], kd)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                quat = d.qpos[3:7]
                lin_vel_world = d.qvel[0:3]
                base_lin_vel = motion_tracking.quat_apply(motion_tracking.quat_inv(quat), lin_vel_world)
                base_ang_vel = d.qvel[3:6]

                joint_pos = d.qpos[7:36].copy()
                joint_vel = d.qvel[6:35].copy()

                robot_anchor_pos = d.xpos[torso_body_id].copy()
                robot_anchor_quat = d.xquat[torso_body_id].copy()

                obs = motion_tracking.build_obs(
                    ref, frame_idx, robot_anchor_pos, robot_anchor_quat,
                    base_lin_vel, base_ang_vel, joint_pos, joint_vel, last_action,
                )
                last_action = actor.act(obs)
                target_dof_pos = last_action * motion_tracking.ACTION_SCALE + default_pos

                frame_idx = (frame_idx + 1) % ref.num_frames

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return "window_closed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "g1_liveknobs.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    runtime_state_path = _resolve(cfg["runtime_state_path"])
    # g1_sim always starts in policy (live vx/vy/yaw driving) mode, regardless
    # of whatever mode a previous session left behind — switching to a motion
    # clip is a live, UI-driven action within a session, not a persisted
    # startup default. Other knob values (if the file already exists) are
    # preserved rather than reset.
    initial_state = load_runtime_state(runtime_state_path, DEFAULT_RUNTIME_STATE)
    initial_state["mode"] = "policy"
    with open(runtime_state_path, "w") as f:
        json.dump(initial_state, f, indent=2)

    # Loop so the UI can switch modes (policy <-> motion_clip) without
    # restarting this process — each run_*_mode() returns "mode_switch" when
    # it detects the mode no longer matches (re-enter in the new mode), or
    # "window_closed" when the user closed the viewer (exit for good).
    while True:
        state = load_runtime_state(runtime_state_path, DEFAULT_RUNTIME_STATE)
        mode = state.get("mode")
        if mode == "motion_clip":
            result = run_motion_clip_mode(cfg, runtime_state_path)
        elif mode == "motion_tracking":
            result = run_motion_tracking_mode(cfg, runtime_state_path)
        else:
            result = run_policy_mode(cfg, runtime_state_path)
        if result == "window_closed":
            break


if __name__ == "__main__":
    main()
