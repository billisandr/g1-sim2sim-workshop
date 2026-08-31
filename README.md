# G1 RL Live-Knobs Workshop

A non-coder workshop built around the Unitree G1 (23-DOF EDU Standard, no
grippers). Participants turn live sliders, watch a walking policy react in a
MuJoCo viewer, and an instructor uploads the validated policy to the real
robot at the end. Nobody edits code. Every "exercise" is a slider, a
dropdown, or a button in a browser tab.

It's one of three related G1 workshops built for the same lab: this one
(no coding, live sliders), a coding-exercise sibling that walks through the
same train-validate-deploy arc in code, and a policy-literacy workshop built
around browsing Hugging Face. This repo borrows the Hugging Face browsing
idea from that third one, and its live-knobs Streamlit UI is modeled on an
earlier robot-arm workshop's reflex-tuning interface. None of those other
repos are included here.

## Why WSL2 and not Docker

The whole thing runs inside WSL2 (Ubuntu-22.04), reusing an existing Python
venv. There's no Dockerfile anywhere in this repo, and that's a deliberate
choice, not an oversight.

The reason is GPU support. This project's text-to-motion generation
(NVIDIA's Kimodo) only installs cleanly in WSL2 on the hardware it was built
on. Native Windows doesn't have the right Python version or C++ build chain
for it. Once WSL2 was already set up and GPU-verified, putting Docker on top
would have meant re-proving GPU passthrough and X11 forwarding through a
third layer, for no real benefit, since WSL2 and WSLg already handle both
natively on this machine.

The trade-off is portability. Moving this workshop to a different lab
machine means redoing the WSL2 setup below, not just running `docker build`.
If that portability ever becomes a real requirement, Docker is worth
revisiting then, but it wasn't worth the setup cost up front for a
single-machine workshop.

## What participants actually do

**1. Get a walking motion or policy.** Pick the pre-staged one, which
always works and needs no internet or GPU. Or browse Hugging Face for a
published one (mock mode by default, so nobody worries about breaking
anything). Or generate a brand-new motion live with NVIDIA's Kimodo
text-to-motion model from a text prompt.

**2. Watch it move in the MuJoCo viewer** while turning live knobs: forward
speed, turn rate, joint stiffness, gait temperament, push disturbance. The
robot reacts in real time. No relaunch needed between changes.

**3. An instructor uploads the validated policy to the real, physical G1**,
following the safety procedure in
[docs/REAL_ROBOT_UPLOAD.md](docs/REAL_ROBOT_UPLOAD.md).

Three different kinds of things can be loaded here, and the UI treats them
differently on purpose:

- A **live policy** (a neural net, `.pt`) reacts to your vx/vy/yaw/stiffness
  knobs. This is the "you are the joystick" mode. One pre-staged example
  ships in the repo: `motions/pre_staged/g1_walk_policy.pt`.
- A **recorded motion clip** (Kimodo's output, or a retargeted mocap clip
  from `exptech/g1-moves`, `.csv`) is a fixed sequence of poses. It has no
  velocity input to react to, so the UI disables the velocity knobs and
  shows a play/pause/scrub control instead when a clip is selected. Twelve
  pre-staged clips ship as of this writing.
- An **imitation-tracking policy**
  (`motions/pre_staged/tracking/<clip>/`) is also a neural net, but trained
  to perform one specific reference routine rather than react to
  vx/vy/yaw. It's a real PD-driven simulation, so it reacts to pushes the
  same as the live policy does, just locked to one routine instead of
  general commands. See
  [docs/HUGGINGFACE_GUIDE.md](docs/HUGGINGFACE_GUIDE.md) section 3.4 and
  the module docstring in `sim/motion_tracking.py` for the full story on
  where this data came from.

Clips vastly outnumber policies in this workshop, and that's not an
accident. A clip is cheap: retarget the mocap data, fix the quaternion
convention, done. A policy needs real RL training hours on a GPU, or finding
someone else's already-trained checkpoint. The file counts on disk are a
pretty honest picture of that cost difference, and it's worth pointing out
to participants directly.

Pretending a fixed clip or a single-routine policy responds to vx/vy/yaw
commands would be misleading, so the UI stays explicit about which of the
three modes is active at any time.

## Setup

From a Windows terminal:

```powershell
wsl.exe -d Ubuntu-22.04 -u root -- bash /path/to/g1-sim2sim-workshop/setup_wsl_env.sh
```

This assumes a Python venv with Kimodo already installed exists in WSL2 (see
[docs/STARTUP.md](docs/STARTUP.md) if it doesn't yet), and adds `mujoco`,
`streamlit`, `pyyaml`, and `huggingface_hub` to it.

## Running it

Two terminals, both inside WSL2:

```bash
g1_sim    # opens the MuJoCo viewer window (via WSLg)
g1_ui     # starts the Streamlit knobs UI at http://localhost:8501
```

`g1_sim` and `g1_ui` are shell aliases added to `~/.bashrc` during setup.
See [docs/STARTUP.md](docs/STARTUP.md) for the full reference if you'd
rather run the underlying commands directly.

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
`runtime_state.json` instead of being fixed at startup. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

### Three skeletons, not one

The locomotion policy uses a 12-leg-joint skeleton
(`assets/g1_description/g1_12dof.xml`, nq=19) from `unitree_rl_gym`. Motion
clips use one of two other, full-body skeletons that aren't interchangeable
with each other or with the policy's, even though two of them share the
same nq by coincidence:

- Kimodo's own 34-joint animation rig (`assets/g1skel34_kimodo/`, nq=36).
- The real Unitree G1 29-DOF skeleton (`assets/g1_description/g1_29dof.xml`,
  also nq=36), used by the `exptech/g1-moves` mocap clips. See
  [docs/HUGGINGFACE_GUIDE.md](docs/HUGGINGFACE_GUIDE.md) for how those were
  sourced and converted.

A clip can only play back on the skeleton it was made for, kinematically
(`qpos` set directly each frame), and never runs through the policy's PD
controller. `sim/g1_mujoco_liveknobs.py` resolves which skeleton a clip
needs from its filename prefix
(`config/g1_liveknobs.yaml`'s `motion_clip.skeleton_by_prefix`), since nq
matching alone can't tell two same-sized but different skeletons apart.

## Documentation

- [docs/STARTUP.md](docs/STARTUP.md) — full run reference (setup, launch, config)
- [docs/KNOBS_CHEATSHEET.md](docs/KNOBS_CHEATSHEET.md) — what each knob does and what to say
- [docs/REAL_ROBOT_UPLOAD.md](docs/REAL_ROBOT_UPLOAD.md) — safety procedure and both real-robot upload paths
- [docs/HUGGINGFACE_GUIDE.md](docs/HUGGINGFACE_GUIDE.md) — how Hugging Face repos work, the `exptech/g1-moves` motion clips in depth, and a reusable prompt for adding more sources
- [INSTRUCTORS_GUIDE.md](INSTRUCTORS_GUIDE.md) — pre-flight, timing, per-stage notes, safety

## Safety

The real-robot stage is always instructor-only and always follows the
procedure in [docs/REAL_ROBOT_UPLOAD.md](docs/REAL_ROBOT_UPLOAD.md): hoist,
two-person rule, debug mode via the remote's L2+R2 combo. This workshop's
sim2sim stage (knobs plus the MuJoCo viewer) is the validation gate before
anyone touches hardware. If a policy doesn't hold up under slider changes
and pushes here, it doesn't go on the real robot, full stop.

## Troubleshooting

| Problem | Fix |
| ------- | --- |
| Viewer window never appears | Confirm `DISPLAY` is set (WSLg sets this automatically in an interactive WSL2 session), and that you're running inside WSL2, not native Windows Python |
| `ModuleNotFoundError: mujoco`/`streamlit`/etc. | Re-run `setup_wsl_env.sh`, or confirm you activated the right venv |
| Kimodo "Generate" button hangs for a while | Expected. Encoder load and encode alone take ~30-40s, before diffusion sampling even starts. See [docs/KNOBS_CHEATSHEET.md](docs/KNOBS_CHEATSHEET.md) |
| Motion clip fails with an `nq` mismatch error | You loaded a Kimodo clip while in policy mode, or vice versa. Switch modes in the selector tab first |
| Tracking mode has no policies to pick | Download one: `python3 hf/g1_moves_tracking_convert.py <ClipName> <category>` (see [docs/HUGGINGFACE_GUIDE.md](docs/HUGGINGFACE_GUIDE.md) section 3.4) |
| Real-robot step isn't working | Not verifiable in a dev sandbox. Needs the physical G1, a hoist, and a second person. See [docs/REAL_ROBOT_UPLOAD.md](docs/REAL_ROBOT_UPLOAD.md) |

## License

MIT for this repo's own code, see [LICENSE](LICENSE). Bundled third-party
assets (the G1 robot description, the `exptech/g1-moves` motion data) carry
their own separate licenses, listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
