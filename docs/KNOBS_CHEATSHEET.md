# Knobs Cheat Sheet

One card per knob: what it does, what to say, what to watch for. For use
while running the Streamlit UI (`g1_ui`) live in front of a room.

## Selector tab

**Live policy vs. recorded motion clip vs. imitation-tracking policy.** Say:
"A live policy is a brain that reacts to your vx/vy/yaw commands. A clip is
a recording. It can't react, only play back. A tracking policy is also a
brain, but trained to perform one specific routine instead of listening to
your commands. Watch the push button still work on it, unlike the clip."
Watch what knobs disable when you switch modes.

**Hugging Face search.** Say: "This searches example data, not the real
internet, by default. It's safe to click around. We'd only flip to real
mode for a live demo with an instructor's own account token."

**Kimodo generate.** Say: "This is really running a GPU model right now,
which is why it takes a minute. We're not faking the wait." Don't have
everyone click it at once. It's a one-at-a-time demo.

## Group A: drive the robot (policy mode only)

| Knob | What it does | What to watch |
|---|---|---|
| Forward speed (vx) | Commanded forward velocity, m/s | Above ~1.0, watch the gait shorten and quicken |
| Sideways speed (vy) | Commanded lateral velocity, m/s | Small values only, since this policy was trained mostly for forward walking |
| Turn rate (yaw) | Commanded turning velocity, rad/s | Combine with vx for an arc; alone, it spins in place |

Say: "You are the joystick. The legs don't know about 'forward,' they only
know the three numbers you just set."

## Group B: tune the temperament

| Knob | What it does | What to watch |
|---|---|---|
| pd_scale | Multiplies joint stiffness (kp/kd together) | Below ~0.6, legs go floppy and may sag or fall. Above ~1.5, jerky, may oscillate |
| action_scale | Scales how far each policy output moves a joint | Low: small, calm steps. High: big, twitchy steps, can become unstable |

Say: "Same brain, same commands. Only how hard it reacts to its own
decisions changed. This is the gap between a sluggish and an aggressive
robot, with the policy itself untouched."

Presets (Balanced/Floppy/Stiff/Calm gait/Twitchy gait) are quick one-click
demonstrations of the extremes. Use them to bracket the slider's range
before letting people set their own values.

## Group C: break it on purpose (live policy and imitation-tracking modes, not recorded clips)

| Knob | What it does | What to watch |
|---|---|---|
| Push strength | An instantaneous torso impulse, applied on button press | Low (~50): barely a stumble-recovery. High (~300+): may knock it over |

Say: "This is the same kind of disturbance a real push or a curb would
cause. If it can't recover here in sim, it's not going on the real robot."
This is the connective tissue to the sim2sim validation gate. A policy that
falls under a push here fails the gate, full stop. A recorded clip has no
physics to push against, so the button disables there, on purpose.

## Honest caveats to say out loud

- The Kimodo "generate" demo only makes sense as a once-per-stage live
  moment, not something everyone runs in parallel. It's a real GPU job, not
  an instant one.
- A Kimodo motion clip and the locomotion policy use different skeletons
  under the hood. If the UI shows an error switching between them, that's
  the system correctly refusing to play a clip on the wrong skeleton, not a
  bug to paper over.
- Mock Hugging Face search results are illustrative, not real model IDs.
  Say so if anyone asks to actually download one.
