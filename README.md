# G1 RL Live-Knobs Workshop

A **non-coder** workshop for the Unitree G1 (23-DOF EDU Standard, no
grippers): turn live sliders, watch a walking policy react in a MuJoCo
viewer, then an instructor uploads the validated policy to the real robot.
No code editing required — every "exercise" is a slider, dropdown, or
button in a browser tab.

This is the non-coder sibling of
[`g1-policy-deploy-workshop`](../g1-policy-deploy-workshop/) (the coding
exercise covering the same train→validate→deploy arc), and reuses the
Hugging Face browsing idea from
[`g1-policy-literacy-workshop`](../g1-policy-literacy-workshop/). It's
modeled structurally on
[`ros_z1_sim_marker-real-camera`](../ros_z1_sim_marker-real-camera/)'s
live-knobs Streamlit UI, but runs **WSL2-only, no Docker** (see
[Runtime environment](#runtime-environment-wsl2-only-no-docker) below).

---

## Runtime environment: WSL2-only, no Docker

This workshop runs entirely inside **WSL2 (Ubuntu-22.04)**, reusing the
venv at `/root/venvs/kimodo`. There is **no Dockerfile** and no container
for any part of it.

Why: this project's GPU work (NVIDIA Kimodo, text-to-motion generation)
only installs successfully in WSL2 on this hardware — native Windows lacks
the right Python version and C++ build chain. Once WSL2 was already set up
and GPU-verified there, adding Docker on top would mean re-proving GPU
passthrough and X11/GUI forwarding through a third layer for no functional
benefit — WSL2/WSLg already does both natively on this machine.

**Trade-off accepted:** this workshop is less portable to a different lab
machine than a Dockerized workshop would be. A new machine needs to redo
the WSL2 setup (see [Setup](#setup) below), not just `docker build`. If
portability becomes a real requirement, Docker can be revisited then.

---

## What participants do

1. **Get a walking motion or policy** — pick the pre-staged one (always
   works, no internet/GPU needed), browse Hugging Face for a published one
   (mock mode by default), or generate a brand-new motion live with
   NVIDIA's Kimodo text-to-motion model from a text prompt.
2. **Watch it in the MuJoCo viewer** while turning live knobs — forward
   speed, turn rate, joint stiffness, gait temperament, push disturbance —
   and see the robot react in real time, no relaunch needed.
3. **An instructor uploads the validated policy to the real, physical G1**,
   following the safety ritual in
   [`G1_RL_Training_and_Deployment_Guide.md`](../../G1_RL_Training_and_Deployment_Guide.md)
   at the repo root.

Three different "things" can be loaded, and the UI treats them differently
— see `0.Plan&Docs/g1-rl-sim2sim-workshop-PolicyDiversity-PLAN.md` for why
clips (cheap, no GPU) vastly outnumber policies (real RL training, or
finding someone else's already-trained checkpoint) in this workshop:

- A **live policy** (a neural net, `.pt`) reacts to your vx/vy/yaw/stiffness
  knobs — this is the "you are the joystick" experience. One pre-staged
  example: `motions/pre_staged/g1_walk_policy.pt`.
- A **recorded motion clip** (Kimodo's output, or an `exptech/g1-moves`
  retarget, `.csv`) is a fixed sequence of poses — it has no velocity input
  to react to. The UI disables the velocity knobs and shows a play/pause/
  scrub control instead when a clip is selected. 12 pre-staged clips as of
  this writing.
- An **imitation-tracking policy** (`motions/pre_staged/tracking/<clip>/`)
  is also a neural net, but trained to perform ONE specific reference
  routine rather than react to vx/vy/yaw — it's a real PD-driven
  simulation (reacts to pushes, in Group C), just locked to one routine
  instead of being a general controller. See
  [docs/HUGGINGFACE_GUIDE.md](docs/HUGGINGFACE_GUIDE.md) §3.4 and
  `sim/motion_tracking.py`'s module docstring for the full background —
  this is `exptech/g1-moves`'s own per-clip `policy/` output, not
  hypothetical.

Pretending a fixed clip or a single-routine policy responds to vx/vy/yaw
commands would be misleading, so the UI is explicit about which of the
three modes is active.

---

## Setup

From a Windows terminal:

```powershell
wsl.exe -d Ubuntu-22.04 -u root -- bash /mnt/e/SenseLAB_TUC/Projects/SpaceSmSc/2.CodeRepos/g1-sim2sim-workshop/setup_wsl_env.sh
```

This assumes `/root/venvs/kimodo` already exists (Kimodo's own install —
see [docs/STARTUP.md](docs/STARTUP.md) if it doesn't yet) and adds
`mujoco`, `streamlit`, `pyyaml`, `huggingface_hub` to it.

## Running it

Two terminals, both inside WSL2 (`wsl.exe -d Ubuntu-22.04`):

```bash
g1_sim    # opens the MuJoCo viewer window (via WSLg)
g1_ui     # starts the Streamlit knobs UI at http://localhost:8501
```

(`g1_sim`/`g1_ui` are shell aliases added to `~/.bashrc` during setup — see
[docs/STARTUP.md](docs/STARTUP.md) for the full reference if you need to
run the underlying commands directly.)

---

## Architecture

```
config/g1_liveknobs.yaml   — central config: policy/XML paths, PD gains, knob ranges
sim/g1_mujoco_liveknobs.py — MuJoCo loop; reads sim/runtime_state.json every control tick
sim/motion_tracking.py     — third mode: imitation-tracking policy obs/PD/actor (see its docstring)
sim/runtime_state.json     — written by the UI, read by the sim (the "live" part)
ui/workshop_ui.py          — Streamlit: knobs, motion/policy/tracking selector, HF panel, Kimodo panel
hf/hf_browse.py            — Hugging Face search (mock mode by default)
hf/g1_moves_convert.py     — downloads + converts a g1-moves retarget CSV (motion-clip mode)
hf/g1_moves_tracking_convert.py — downloads a g1-moves per-clip tracking policy + reference motion
kimodo_bridge/generate_motion.py — wraps `kimodo_gen` for live text-to-motion generation
motions/pre_staged/        — vetted, known-good policy + clips + tracking/ subfolder
motions/generated/         — Kimodo/HF downloads land here at runtime
assets/                    — MuJoCo XML + meshes for both skeletons used (see below)
tests/smoke_test.py                — headless: load the pre-staged policy, step N ticks, assert it stays upright
tests/smoke_test_motion_tracking.py — headless: same, for the imitation-tracking mode
```

The sim loop is adapted from `unitree_rl_gym`'s
`deploy/deploy_mujoco/deploy_mujoco.py` reference loop, parameterized so
`vx`/`vy`/`yaw`/`pd_scale`/`action_scale`/`push` come from
`runtime_state.json` instead of being fixed at startup.

**Three skeletons, not one:** the locomotion policy uses a 12-leg-joint
skeleton (`assets/g1_description/g1_12dof.xml`, nq=19) from
`unitree_rl_gym`. Motion clips use one of two other, full-body skeletons
that are **not interchangeable with each other or with the policy's**,
despite two of them sharing the same nq by coincidence:
- Kimodo's own 34-joint animation rig (`assets/g1skel34_kimodo/`, nq=36).
- The real Unitree G1 29-DOF skeleton (`assets/g1_description/g1_29dof.xml`,
  also nq=36) used by the `exptech/g1-moves` mocap clips — see
  [docs/HUGGINGFACE_GUIDE.md](docs/HUGGINGFACE_GUIDE.md) for how those were
  sourced and converted.

A clip can only be played back on the skeleton it was made for,
kinematically (`qpos` set directly each frame), never run through the
policy's PD controller. `sim/g1_mujoco_liveknobs.py` resolves which
skeleton a clip needs from its filename prefix
(`config/g1_liveknobs.yaml`'s `motion_clip.skeleton_by_prefix`) — nq
matching alone isn't enough to tell two same-sized but different
skeletons apart.

---

## Documentation

- [docs/STARTUP.md](docs/STARTUP.md) — full run reference (setup, launch, config)
- [docs/KNOBS_CHEATSHEET.md](docs/KNOBS_CHEATSHEET.md) — what each knob does + what to say
- [docs/REAL_ROBOT_UPLOAD.md](docs/REAL_ROBOT_UPLOAD.md) — both real-robot upload paths
- [docs/HUGGINGFACE_GUIDE.md](docs/HUGGINGFACE_GUIDE.md) — how Hugging Face repos work, the `exptech/g1-moves` motion clips in depth, and a reusable prompt for adding more sources
- [INSTRUCTORS_GUIDE.md](INSTRUCTORS_GUIDE.md) — pre-flight, timing, per-stage notes, safety

---

## Safety

The real-robot stage always defers to
[`G1_RL_Training_and_Deployment_Guide.md`](../../G1_RL_Training_and_Deployment_Guide.md)
at the repo root — hoist, two-person rule, debug mode (L2+R2). This
workshop's sim2sim stage (knobs + MuJoCo viewer) is the validation gate
before anyone touches hardware: if the policy doesn't look stable under
slider changes and pushes here, it does not go on the real robot.

---

## Troubleshooting

| Problem | Fix |
| ------- | --- |
| Viewer window never appears | Confirm `DISPLAY` is set (WSLg sets this automatically in an interactive WSL2 session) and that you're running inside WSL2, not native Windows Python |
| `ModuleNotFoundError: mujoco`/`streamlit`/etc. | Re-run `setup_wsl_env.sh`, or confirm you activated `/root/venvs/kimodo` |
| Kimodo "Generate" button hangs for a long time | Expected — encoder load+encode alone takes ~30-40s, before diffusion sampling. See [docs/KNOBS_CHEATSHEET.md](docs/KNOBS_CHEATSHEET.md) |
| Motion clip fails with an `nq` mismatch error | You loaded a Kimodo clip while in policy mode, or vice versa — switch modes in the selector tab first |
| Tracking mode has no policies to pick | Download one: `python3 hf/g1_moves_tracking_convert.py <ClipName> <category>` (see [docs/HUGGINGFACE_GUIDE.md](docs/HUGGINGFACE_GUIDE.md) §3.4) |
| Real-robot step | Not verifiable in a dev sandbox — needs the physical G1, hoist, and a second person. See [docs/REAL_ROBOT_UPLOAD.md](docs/REAL_ROBOT_UPLOAD.md) |
