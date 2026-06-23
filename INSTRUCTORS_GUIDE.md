# Instructor's Guide

This is a **non-coder** workshop — participants never open an editor. Your
job is narration and pacing across three stages, ending with an optional
real-robot demo that only you (the instructor) drive.

---

## Pre-flight (do this before the room fills up)

1. Confirm WSL2 environment still works:
   ```bash
   wsl.exe -d Ubuntu-22.04 -u root -- bash -c "source /root/venvs/kimodo/bin/activate && python3 -c 'import torch, mujoco, streamlit; print(torch.cuda.is_available())'"
   ```
   Should print `True`.
2. Run the headless smoke test (catches a broken policy/asset before
   anyone's watching):
   ```bash
   wsl.exe -d Ubuntu-22.04 -u root -- bash -c "source /root/venvs/kimodo/bin/activate && cd /mnt/e/.../g1-sim2sim-workshop && python3 tests/smoke_test.py"
   ```
3. Launch both `g1_sim` and `g1_ui` once yourself, end to end, on the
   actual room machine — not just on your laptop. WSLg display forwarding
   and GPU passthrough are machine-specific; verify on-site.
4. If you plan to demo Kimodo generation live: run it once beforehand so
   the model weights are warm in the OS file cache (the *first* run after
   a reboot is slower than subsequent ones).
5. If a real-robot demo is planned: do the full safety setup from
   `G1_RL_Training_and_Deployment_Guide.md` §5.1 well before participants
   arrive, not as a live "let's see if this works" moment.

---

## Timing (suggested — flex to the room)

| Stage | Time | What happens |
|---|---|---|
| Intro + framing | 5 min | What's a policy vs. a motion clip; why we validate in sim first |
| Stage 1: get a motion/policy | 15-20 min | Pre-staged walkthrough, HF browse (mock), one live Kimodo generation demo |
| Stage 2: live knobs | 25-35 min | Groups A/B/C, in order, with presets first then free exploration |
| Stage 3: real robot (optional) | 15-30 min | Instructor-only, full safety ritual, audience watches |
| Wrap-up | 5 min | What changed, what didn't, why sim2sim matters |

---

## Stage 1 notes — "Get a policy"

- Start with the **pre-staged** policy/motion selection — it always works,
  no internet/GPU dependency. Don't open with the riskiest path.
- The **HF browse** panel is mock by default — make this explicit so no
  one thinks they're browsing the real internet. If you want a real-mode
  demo, you (the instructor) flip `MOCK_HF=0` yourself beforehand; don't
  improvise a real gated model ID in front of the room (see
  `hf/hf_browse.py`'s docstring — same caution as the literacy workshop).
- The **Kimodo generate** panel is a real GPU job — budget ~1-2 minutes of
  dead air while it runs, and narrate what's happening (text encoder load,
  then diffusion sampling) rather than standing there silently. Do this
  once, live, as a demonstration — not as something every participant
  runs in parallel (it would queue on one GPU).
- If a participant's generated clip looks broken (jittery, off-skeleton),
  that's expected variance from a generative model — use it as a teaching
  moment about why sim2sim validation exists, then fall back to the
  pre-staged clip.

## Two ways robots learn motion (talking point for Stage 1)

This workshop ships **12 motion clips** but **1 motion policy**. That's not
an oversight — it's the most honest thing you can show a non-coder about
why general robot skill is hard. Use it as a deliberate talking point, not
just a fact to skip past.

**Approach 1 — Reward-driven RL ("learn to walk from scratch").** No human
reference motion at all. You write a reward function (stay alive, track
the commanded velocity, don't waste energy) and let thousands of simulated
copies explore until a walking gait emerges that maximizes it. This is how
the workshop's one policy (`g1_walk_policy.pt`) was made. **Output: a
general controller that reacts to commands it was never explicitly
shown** — this is the entire "Drive the robot" experience. It generalizes
and recovers from pushes, but you cannot reward-shape your way to "now do
a karate chop" — expressive, specific motion isn't something this approach
produces naturally.

**Approach 2 — Motion imitation / retargeting ("teach it to copy a
human").** A reference motion already exists, from real mocap or a
generative model. Two sub-options:
- **Kinematic playback** — just replay the reference poses, no physics,
  no reactivity. This is *all 12* of our clips. Cheap: a retarget +
  quaternion-fix script, no GPU-hours. The robot is a puppet here, not
  moving itself.
- **Imitation-trained tracking policy** — train an RL policy whose reward
  is "match this one reference motion's joints" instead of "track a
  commanded velocity." Produces a policy that performs *that one specific
  routine* but stays robust to physics along the way — not a vx/vy/yaw
  joystick, a "perform this robustly" controller. This is the workshop's
  **third selector mode** (`"Imitation-tracking policy"` in the mode
  radio) — not hypothetical, not a future extension: pick it, watch a
  karate routine play out under real physics, and try the **push** button
  in Group C on it, same as the live policy. See
  `docs/HUGGINGFACE_GUIDE.md` §3.4 and `sim/motion_tracking.py` for how it
  works and `python3 hf/g1_moves_tracking_convert.py` for downloading more
  routines.

**Say this:** "Clips are cheap to add — one script, no GPU time. A policy
needs real training, hours-to-days on a GPU, or finding someone else's
already-trained checkpoint. That cost difference is *why* this workshop
has a dozen clips and far fewer policies — you're looking straight at the
asymmetry in the file counts on disk."

## Stage 2 notes — "Live knobs"

- Run Groups A, B, C **in that order** — driving (intuitive) before
  temperament (more abstract) before breaking it (payoff).
- Use the preset buttons first in each group so the room sees the
  intended extremes, *then* let people set their own slider values.
- The "push" button in Group C is the bridge to sim2sim validation: a
  policy that falls over from a moderate push here is a policy that fails
  the gate and does not go on the real robot. Say this explicitly.
- If someone asks "can the clip respond to my speed slider" while in
  motion-clip mode — no, by design (see `KNOBS_CHEATSHEET.md`). This is a
  deliberate, honest UI choice, not a missing feature.
- The push button also works in **imitation-tracking** mode, not just live
  policy — it's a real PD-driven sim, unlike a recorded clip. Worth a
  side-by-side: push a recorded clip (nothing happens, no physics) then
  push the tracking policy mid-routine (it visibly reacts) to make the
  "this is a different category than both other modes" point land.

## Stage 3 notes — "Real robot" (optional, instructor-only)

- This step is **never** participant-driven. Read
  `docs/REAL_ROBOT_UPLOAD.md` and
  `G1_RL_Training_and_Deployment_Guide.md` §5 and §7 yourself, beforehand,
  in full.
- Non-negotiables, restated because they're worth repeating: robot
  hoisted, two people present, a kill switch always in reach, never skip
  the sim2sim gate (Stage 2's push test is part of that gate).
- It's fine to skip this stage entirely if there's no hardware access that
  day — the workshop's pedagogical payoff (Stages 1-2) doesn't depend on
  it.

---

## Common questions

**"Why can't I just edit the YAML directly?"** — You can (it's a normal
text file), but the workshop is designed so no one needs to. The point is
showing that *the same robot* feels completely different just from
slider values — no code, no restart.

**"Is the push knob doing the same thing the leg-following Z1 workshop
does?"** — No, different exercise: Z1's reflexes workshop tunes a
*response delay*. The G1 push tests *recovery from a disturbance* — a
different failure mode (falling over vs. lagging behind).

**"What happens to my generated motion after the workshop?"** — It stays
in `motions/generated/` on this machine. Nothing is uploaded anywhere
automatically.

---

## Troubleshooting quick-reference

See the main `README.md`'s troubleshooting table for technical issues. If
the room loses momentum waiting on a Kimodo generation, switch to the
pre-staged motion clip and circle back to the generated one once it's
ready — don't let one slow GPU job stall the whole session.
