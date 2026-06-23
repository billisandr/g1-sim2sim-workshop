"""
Third sim-loop mode: an imitation-trained MOTION-TRACKING policy.

This is deliberately a different thing from both existing modes:

- "policy" mode: a general locomotion controller, reacts to vx/vy/yaw.
- "motion_clip" mode: a fixed recording, kinematic playback, no physics.
- "motion_tracking" mode (this file): a neural net trained to perform ONE
  specific reference routine (e.g. a karate move) while a real PD-driven
  MuJoCo simulation runs underneath it — so it responds to pushes/physics,
  but it does NOT take vx/vy/yaw commands. See docs/HUGGINGFACE_GUIDE.md
  §3.4 and the PolicyDiversity plan §3.1 for the full background on why
  this is a third, distinct category, not a variant of the other two.

These policies come from `exptech/g1-moves`'s per-clip `policy/` folders,
trained with the `mjlab` framework (not unitree_rl_gym/legged_gym, unlike
this workshop's other policy). Two facts had to be reverse-engineered from
mjlab's actual source (github.com/mujocolab/mjlab) rather than trusted from
the per-clip YAML, per this project's own due-diligence convention
(docs/HUGGINGFACE_GUIDE.md §5.3/§6):

1. The checkpoint is a raw RSL-RL-style training checkpoint (a dict with
   `model_state_dict`, optimizer state, etc.) — NOT a TorchScript export
   like this workshop's other policy. `torch.jit.load()` fails on it; you
   have to `torch.load()` the dict and reconstruct the actor MLP yourself
   from `model_state_dict`'s tensor shapes.
2. The actor's 160-dim observation is NOT vx/vy/yaw-shaped. By reading
   mjlab's actual `mjlab/tasks/tracking/mdp/{commands,observations}.py` and
   `mjlab/envs/mdp/observations.py` source and matching dimension-for-
   dimension against the checkpoint's real input shape, it decomposes as:
     58  command            = [ref_joint_pos(29), ref_joint_vel(29)] at the
                               current reference-motion frame
      3  motion_anchor_pos_b = reference torso position, robot-local frame
      6  motion_anchor_ori_b = reference torso orientation (first two
                               columns of the relative rotation matrix —
                               mjlab's "6D rotation" trick, avoids the
                               quaternion-sign discontinuity)
      3  base_lin_vel        = robot's own body-frame linear velocity
      3  base_ang_vel        = robot's own body-frame angular velocity
     29  joint_pos (rel)     = robot's actual joint pos minus default pose
     29  joint_vel (rel)     = robot's actual joint vel minus default (0)
     29  last_action         = the action this policy output last tick
    ---
    160
   This exactly matches the checkpoint's real `actor.0.weight` shape
   (512, 160) — the cross-check that justifies trusting this decomposition
   instead of guessing.

The per-joint PD gains, action scale, and default pose below come from a
clip's `policy/env.yaml` (the `mjlab.asset_zoo.robots.unitree_g1.g1_constants`
robot spec) — confirmed byte-identical across two different clips'
env.yaml (M_ShortMove12 vs M_ShortMove13), so these are robot-level
constants shared by every g1-moves tracking policy, not something to
re-derive per clip. The 29-joint canonical order matches
docs/HUGGINGFACE_GUIDE.md §3.3's table, itself verified against
assets/g1_description/g1_29dof.xml's own <joint>/<actuator> order.
"""

import numpy as np
import torch
import torch.nn as nn

# fmt: off
# Index   Joint                    Group
#   0  left_hip_pitch_joint       hip_pitch/hip_yaw/waist_yaw
#   1  left_hip_roll_joint        hip_roll/knee
#   2  left_hip_yaw_joint         hip_pitch/hip_yaw/waist_yaw
#   3  left_knee_joint            hip_roll/knee
#   4  left_ankle_pitch_joint     ankle/waist_pitch/waist_roll
#   5  left_ankle_roll_joint      ankle/waist_pitch/waist_roll
#   6  right_hip_pitch_joint      hip_pitch/hip_yaw/waist_yaw
#   7  right_hip_roll_joint       hip_roll/knee
#   8  right_hip_yaw_joint        hip_pitch/hip_yaw/waist_yaw
#   9  right_knee_joint           hip_roll/knee
#  10  right_ankle_pitch_joint    ankle/waist_pitch/waist_roll
#  11  right_ankle_roll_joint     ankle/waist_pitch/waist_roll
#  12  waist_yaw_joint            hip_pitch/hip_yaw/waist_yaw
#  13  waist_roll_joint           ankle/waist_pitch/waist_roll
#  14  waist_pitch_joint          ankle/waist_pitch/waist_roll
#  15  left_shoulder_pitch_joint  elbow/shoulder/wrist_roll
#  16  left_shoulder_roll_joint   elbow/shoulder/wrist_roll
#  17  left_shoulder_yaw_joint    elbow/shoulder/wrist_roll
#  18  left_elbow_joint           elbow/shoulder/wrist_roll
#  19  left_wrist_roll_joint      elbow/shoulder/wrist_roll
#  20  left_wrist_pitch_joint     wrist_pitch/wrist_yaw
#  21  left_wrist_yaw_joint       wrist_pitch/wrist_yaw
#  22  right_shoulder_pitch_joint elbow/shoulder/wrist_roll
#  23  right_shoulder_roll_joint  elbow/shoulder/wrist_roll
#  24  right_shoulder_yaw_joint   elbow/shoulder/wrist_roll
#  25  right_elbow_joint          elbow/shoulder/wrist_roll
#  26  right_wrist_roll_joint     elbow/shoulder/wrist_roll
#  27  right_wrist_pitch_joint    wrist_pitch/wrist_yaw
#  28  right_wrist_yaw_joint      wrist_pitch/wrist_yaw
# fmt: on

_HIP_PITCH_YAW_WAIST_YAW = dict(kp=40.17923863450712, kd=2.557889775413375, scale=0.5475464629911068,
                                 armature=0.01017752004132231)
_HIP_ROLL_KNEE = dict(kp=99.09842777666111, kd=6.308801853496639, scale=0.35066146637882434,
                       armature=0.025101924999999997)
_ANKLE_WAIST_PITCH_ROLL = dict(kp=28.50124619574858, kd=1.814445686584846, scale=0.43857731392336724,
                                armature=0.00721945)
_SHOULDER_ELBOW_WRISTROLL = dict(kp=14.25062309787429, kd=0.907222843292423, scale=0.43857731392336724,
                                  armature=0.003609725)
_WRIST_PITCH_YAW = dict(kp=16.77832748089279, kd=1.06814150219, scale=0.07450087032950714,
                         armature=0.00425)

_GROUP_BY_JOINT_INDEX = [
    _HIP_PITCH_YAW_WAIST_YAW, _HIP_ROLL_KNEE, _HIP_PITCH_YAW_WAIST_YAW, _HIP_ROLL_KNEE,
    _ANKLE_WAIST_PITCH_ROLL, _ANKLE_WAIST_PITCH_ROLL,
    _HIP_PITCH_YAW_WAIST_YAW, _HIP_ROLL_KNEE, _HIP_PITCH_YAW_WAIST_YAW, _HIP_ROLL_KNEE,
    _ANKLE_WAIST_PITCH_ROLL, _ANKLE_WAIST_PITCH_ROLL,
    _HIP_PITCH_YAW_WAIST_YAW, _ANKLE_WAIST_PITCH_ROLL, _ANKLE_WAIST_PITCH_ROLL,
    _SHOULDER_ELBOW_WRISTROLL, _SHOULDER_ELBOW_WRISTROLL, _SHOULDER_ELBOW_WRISTROLL,
    _SHOULDER_ELBOW_WRISTROLL, _SHOULDER_ELBOW_WRISTROLL,
    _WRIST_PITCH_YAW, _WRIST_PITCH_YAW,
    _SHOULDER_ELBOW_WRISTROLL, _SHOULDER_ELBOW_WRISTROLL, _SHOULDER_ELBOW_WRISTROLL,
    _SHOULDER_ELBOW_WRISTROLL, _SHOULDER_ELBOW_WRISTROLL,
    _WRIST_PITCH_YAW, _WRIST_PITCH_YAW,
]

KP = np.array([g["kp"] for g in _GROUP_BY_JOINT_INDEX], dtype=np.float32)
KD = np.array([g["kd"] for g in _GROUP_BY_JOINT_INDEX], dtype=np.float32)
ACTION_SCALE = np.array([g["scale"] for g in _GROUP_BY_JOINT_INDEX], dtype=np.float32)
# g1_29dof.xml ships with zero <joint armature> on every actuated joint.
# mjlab's spec_fn sets a nonzero rotor armature per joint (these values,
# from the same env.yaml) — without it, the same PD gains see far less
# effective inertia than the policy was trained against and the sim
# diverges within a few ticks (confirmed: this is what was actually
# happening before this was added, not an obs/action bug). Applied to
# m.dof_armature[6:35] by the caller before stepping.
ARMATURE = np.array([g["armature"] for g in _GROUP_BY_JOINT_INDEX], dtype=np.float32)

# From env.yaml's scene.entities.robot.init_state.joint_pos regex defaults;
# every joint not listed there defaults to 0.0.
DEFAULT_JOINT_POS = np.zeros(29, dtype=np.float32)
DEFAULT_JOINT_POS[[0, 6]] = -0.312       # *_hip_pitch_joint
DEFAULT_JOINT_POS[[3, 9]] = 0.669        # *_knee_joint
DEFAULT_JOINT_POS[[4, 10]] = -0.363      # *_ankle_pitch_joint
DEFAULT_JOINT_POS[[18, 25]] = 0.6        # *_elbow_joint
DEFAULT_JOINT_POS[15] = 0.2              # left_shoulder_pitch_joint
DEFAULT_JOINT_POS[16] = 0.2              # left_shoulder_roll_joint
DEFAULT_JOINT_POS[22] = 0.2              # right_shoulder_pitch_joint
DEFAULT_JOINT_POS[23] = -0.2             # right_shoulder_roll_joint

# Index into the npz's 30-body axis (matches g1_29dof.xml's <body> tree
# order — see docs/HUGGINGFACE_GUIDE.md §3.3's verification methodology).
ANCHOR_BODY_INDEX = 15  # torso_link

NUM_ACTIONS = 29
NUM_OBS = 160


# --------------------------------------------------------------------------
# Quaternion math (w, x, y, z convention — matches MuJoCo's qpos and
# mjlab's own convention, so no xyzw/wxyz reorder is needed here, unlike
# the g1-moves retarget CSVs).
# --------------------------------------------------------------------------

def quat_inv(q):
    w, x, y, z = q
    n2 = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z], dtype=np.float64) / max(n2, 1e-9)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_apply(q, v):
    w, x, y, z = q
    xyz = np.array([x, y, z])
    t = 2 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def matrix_from_quat_first_two_cols(q):
    """First two columns of the rotation matrix for q (w,x,y,z) — mjlab's
    6D rotation representation (avoids the quaternion sign discontinuity).

    mjlab does `mat[..., :2].reshape(B, -1)` on a (B,3,3) tensor, which
    flattens row-major: [R00,R01, R10,R11, R20,R21] (each row's two
    column-0/column-1 entries, row by row) — NOT column0-then-column1.
    Getting this order wrong still "mostly works" early on (small angles
    look similar either way) but compounds into a slow collapse exactly
    like what was observed before this fix."""
    w, x, y, z = q
    s = 2.0 / max(w * w + x * x + y * y + z * z, 1e-9)
    r00, r01 = 1 - s * (y * y + z * z), s * (x * y - w * z)
    r10, r11 = s * (x * y + w * z), 1 - s * (x * x + z * z)
    r20, r21 = s * (x * z - w * y), s * (y * z + w * x)
    return np.array([r00, r01, r10, r11, r20, r21])


def subtract_frame_transforms(t01, q01, t02, q02):
    """T12 = T01^-1 * T02 — position/orientation of frame 2 expressed in
    frame 1's local frame. Mirrors mjlab.utils.lab_api.math's function of
    the same name (verified against its actual source)."""
    q10 = quat_inv(q01)
    q12 = quat_mul(q10, q02)
    t12 = quat_apply(q10, t02 - t01)
    return t12, q12


# --------------------------------------------------------------------------
# Actor network
# --------------------------------------------------------------------------

class TrackingActor:
    """Wraps one g1-moves `<clip>_policy.pt` checkpoint for inference.

    This is a raw RSL-RL training checkpoint (model_state_dict + optimizer
    state), not a TorchScript export — the actor MLP is reconstructed here
    from the checkpoint's own tensor shapes (512, 256, 128 hidden, ELU,
    per agent.yaml) rather than assumed.
    """

    def __init__(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"]

        in_dim = sd["actor.0.weight"].shape[1]
        out_dim = sd["actor.6.weight"].shape[0]
        if in_dim != NUM_OBS or out_dim != NUM_ACTIONS:
            raise ValueError(
                f"{checkpoint_path}: actor is {in_dim}-in/{out_dim}-out, "
                f"expected {NUM_OBS}-in/{NUM_ACTIONS}-out — this checkpoint "
                "doesn't match the g1-moves tracking-policy contract this "
                "mode assumes."
            )

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, out_dim),
        )
        with torch.no_grad():
            self.net[0].weight.copy_(sd["actor.0.weight"]); self.net[0].bias.copy_(sd["actor.0.bias"])
            self.net[2].weight.copy_(sd["actor.2.weight"]); self.net[2].bias.copy_(sd["actor.2.bias"])
            self.net[4].weight.copy_(sd["actor.4.weight"]); self.net[4].bias.copy_(sd["actor.4.bias"])
            self.net[6].weight.copy_(sd["actor.6.weight"]); self.net[6].bias.copy_(sd["actor.6.bias"])
        self.net.eval()

        self.obs_mean = sd["actor_obs_normalizer._mean"].numpy().reshape(-1)
        self.obs_std = sd["actor_obs_normalizer._std"].numpy().reshape(-1)

    @torch.no_grad()
    def act(self, obs):
        normed = (obs - self.obs_mean) / np.clip(self.obs_std, 1e-6, None)
        out = self.net(torch.from_numpy(normed.astype(np.float32)).unsqueeze(0))
        return out.squeeze(0).numpy()


# --------------------------------------------------------------------------
# Reference motion
# --------------------------------------------------------------------------

class ReferenceMotion:
    """One g1-moves `training/<clip>.npz` — per-frame joint + body FK data.

    Frame indexing matches mjlab's own MotionCommand (`time_steps += 1`
    once per control step, wrapping at the array length) — NOT real-time
    playback at the npz's `fps` metadata, which is the original mocap
    capture rate, not the control rate the policy was trained against. See
    this module's docstring.
    """

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.joint_pos = data["joint_pos"].astype(np.float32)
        self.joint_vel = data["joint_vel"].astype(np.float32)
        self.body_pos_w = data["body_pos_w"].astype(np.float64)
        self.body_quat_w = data["body_quat_w"].astype(np.float64)
        self.fps = float(data["fps"][0]) if "fps" in data.files else None
        self.num_frames = self.joint_pos.shape[0]

    def command(self, t):
        return np.concatenate([self.joint_pos[t], self.joint_vel[t]])

    def anchor_pose(self, t):
        return self.body_pos_w[t, ANCHOR_BODY_INDEX], self.body_quat_w[t, ANCHOR_BODY_INDEX]

    def root_pose(self, t):
        """Pelvis (body index 0) world pose at frame t — pelvis is the body
        directly welded to the free joint with zero offset, so this is
        usable directly as a free-joint qpos[0:7] initial value."""
        return self.body_pos_w[t, 0], self.body_quat_w[t, 0]


def build_obs(ref, frame_idx, robot_anchor_pos, robot_anchor_quat,
              base_lin_vel, base_ang_vel, joint_pos, joint_vel, last_action):
    """Assemble the 160-dim actor observation for one control tick. Term
    order/composition matches mjlab's tracking task obs_groups.policy —
    see this module's docstring for the derivation."""
    command = ref.command(frame_idx)

    ref_anchor_pos, ref_anchor_quat = ref.anchor_pose(frame_idx)
    anchor_pos_b, anchor_quat_b = subtract_frame_transforms(
        robot_anchor_pos, robot_anchor_quat, ref_anchor_pos, ref_anchor_quat
    )
    anchor_ori_b = matrix_from_quat_first_two_cols(anchor_quat_b)

    joint_pos_rel = joint_pos - DEFAULT_JOINT_POS
    joint_vel_rel = joint_vel  # default_joint_vel is 0 for every joint

    return np.concatenate([
        command,                      # 58
        anchor_pos_b.astype(np.float32),    # 3
        anchor_ori_b.astype(np.float32),    # 6
        base_lin_vel.astype(np.float32),    # 3
        base_ang_vel.astype(np.float32),    # 3
        joint_pos_rel.astype(np.float32),   # 29
        joint_vel_rel.astype(np.float32),   # 29
        last_action.astype(np.float32),     # 29
    ]).astype(np.float32)
