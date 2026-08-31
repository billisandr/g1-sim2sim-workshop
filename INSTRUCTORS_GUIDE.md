# Instructor's Guide

This is a non-coder workshop. Participants never open an editor. Your job is
narration and pacing across three stages, ending with an optional real-robot
demo that only you, the instructor, drive.

## Pre-flight, before the room fills up

1. Confirm the WSL2 environment still works:

   ```bash
   wsl.exe -d Ubuntu-22.04 -u root -- bash -c "source /root/venvs/kimodo/bin/activate && python3 -c 'import torch, mujoco, streamlit; print(torch.cuda.is_available())'"
   ```

   Should print `True`.

2. Run the headless smoke test. It catches a broken policy or asset before
   anyone's watching:

   ```bash
   wsl.exe -d Ubuntu-22.04 -u root -- bash -c "source /root/venvs/kimodo/bin/activate && cd /path/to/g1-sim2sim-workshop && python3 tests/smoke_test.py"
   ```

3. Launch both `g1_sim` and `g1_ui` yourself, end to end, on the actual room
   machine, not just your laptop. WSLg display forwarding and GPU
   passthrough are machine-specific, so verify on-site rather than trusting
   it worked elsewhere.

4. Planning to demo Kimodo generation live? Run it once beforehand so the
   model weights are warm in the OS file cache. The first run after a
   reboot is noticeably slower than the ones after it.

5. Planning a real-robot demo? Do the full safety setup from
   [docs/REAL_ROBOT_UPLOAD.md](docs/REAL_ROBOT_UPLOAD.md) well before
   participants arrive. Don't treat it as a live "let's see if this works"
   moment in front of the room.

## Timing (suggested, flex to the room)

| Stage | Time | What happens |
|---|---|---|
| Intro and framing | 5 min | What's a policy vs. a motion clip, and why we validate in sim first |
| Stage 1: get a motion/policy | 15-20 min | Pre-staged walkthrough, HF browse (mock), one live Kimodo generation demo |
| Stage 2: live knobs | 25-35 min | Groups A/B/C, in order, presets first then free exploration |
| Stage 3: real robot (optional) | 15-30 min | Instructor-only, full safety procedure, audience watches |
| Wrap-up | 5 min | What changed, what didn't, why sim2sim matters |

## Stage 1 notes: "Get a policy"

Start with the pre-staged policy or motion selection. It always works, with
no internet or GPU dependency, so don't open with the riskiest path.

The HF browse panel runs in mock mode by default. Make that explicit so no
one thinks they're browsing the real internet. If you want a real-mode demo,
flip `MOCK_HF=0` yourself beforehand rather than improvising a real gated
model ID in front of the room. See `hf/hf_browse.py`'s docstring for the
same caution.

The Kimodo generate panel runs a real GPU job, so budget one to two minutes
of dead air while it runs, and narrate what's happening (text encoder load,
then diffusion sampling) instead of standing there silently. Do this once,
live, as a demonstration, not as something every participant runs in
parallel. It would just queue on one GPU.

If a participant's generated clip looks broken, jittery or off-skeleton,
that's expected variance from a generative model. Use it as a teaching
moment about why sim2sim validation exists, then fall back to the
pre-staged clip.

### Two ways robots learn motion (a talking point for Stage 1)

This workshop ships twelve motion clips but one motion policy. That's not
an oversight. It's the most honest thing you can show a non-coder about why
general robot skill is hard, so treat it as a deliberate talking point
rather than a fact to skip past.

**Approach 1: reward-driven RL, "learn to walk from scratch."** No human
reference motion at all. You write a reward function (stay alive, track
the commanded velocity, don't waste energy) and let thousands of simulated
copies explore until a walking gait emerges that maximizes it. This is how
the workshop's one policy, `g1_walk_policy.pt`, was made. The output is a
general controller that reacts to commands it was never explicitly shown.
That's the entire "drive the robot" experience. It generalizes and
recovers from pushes, but you can't reward-shape your way to "now do a
karate chop." Expressive, specific motion just isn't something this
approach produces naturally.

**Approach 2: motion imitation or retargeting, "teach it to copy a
human."** A reference motion already exists, from real mocap or a
generative model. Two sub-options:

- **Kinematic playback**: just replay the reference poses, no physics, no
  reactivity. This is all twelve of our clips. It's cheap, a retarget plus a
  quaternion-fix script, no GPU-hours. The robot is a puppet here, not
  moving itself.
- **Imitation-trained tracking policy**: train an RL policy whose reward is
  "match this one reference motion's joints" instead of "track a commanded
  velocity." This produces a policy that performs that one specific
  routine but stays robust to physics along the way, a "perform this
  robustly" controller rather than a vx/vy/yaw joystick. This is the
  workshop's third selector mode ("Imitation-tracking policy" in the mode
  radio button), fully wired in, not hypothetical. Pick it, watch a karate
  routine play out under real physics, and try the push button in Group C
  on it, same as the live policy. See
  `docs/HUGGINGFACE_GUIDE.md` section 3.4 and `sim/motion_tracking.py` for
  how it works, and `python3 hf/g1_moves_tracking_convert.py` for
  downloading more routines.

Say this out loud: "Clips are cheap to add, one script, no GPU time. A
policy needs real training, hours to days on a GPU, or finding someone
else's already-trained checkpoint. That cost difference is why this
workshop has a dozen clips and far fewer policies. You're looking straight
at the asymmetry in the file counts on disk."

## Stage 2 notes: "Live knobs"

Run Groups A, B, C in that order: driving is intuitive, temperament is more
abstract, breaking it is the payoff.

Use the preset buttons first in each group so the room sees the intended
extremes, then let people set their own slider values.

The push button in Group C is the bridge to sim2sim validation. A policy
that falls over from a moderate push here is a policy that fails the gate
and doesn't go on the real robot. Say this explicitly.

If someone asks "can the clip respond to my speed slider" while in
motion-clip mode, the answer is no, by design (see `KNOBS_CHEATSHEET.md`).
That's a deliberate, honest UI choice, not a missing feature.

The push button also works in imitation-tracking mode, not just live
policy, since it's a real PD-driven sim, unlike a recorded clip. Worth a
side-by-side demo: push a recorded clip (nothing happens, no physics), then
push the tracking policy mid-routine (it visibly reacts). That contrast is
what makes the "this is a different category than both other modes" point
land.

## Stage 3 notes: "Real robot" (optional, instructor-only)

This step is never participant-driven. Read
[docs/REAL_ROBOT_UPLOAD.md](docs/REAL_ROBOT_UPLOAD.md) yourself, beforehand,
in full.

The non-negotiables are worth repeating even though you've read them
before: robot hoisted, two people present, a kill switch always in reach,
never skip the sim2sim gate. Stage 2's push test is part of that gate.

It's fine to skip this stage entirely if there's no hardware access that
day. The workshop's pedagogical payoff in Stages 1 and 2 doesn't depend on
it.

## Common questions

**"Why can't I just edit the YAML directly?"** You can, it's a normal text
file, but the workshop is designed so no one needs to. The point is showing
that the same robot feels completely different just from slider values,
with no code and no restart.

**"What happens to my generated motion after the workshop?"** It stays in
`motions/generated/` on this machine. Nothing gets uploaded anywhere
automatically.

## Troubleshooting quick-reference

See the main README's troubleshooting table for technical issues. If the
room loses momentum waiting on a Kimodo generation, switch to the
pre-staged motion clip and circle back to the generated one once it's
ready. Don't let one slow GPU job stall the whole session.
