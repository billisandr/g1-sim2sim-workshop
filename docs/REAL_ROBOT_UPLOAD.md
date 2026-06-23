# Real-Robot Upload

This workshop's sim2sim stage (knobs + MuJoCo viewer) is the validation
gate. Once an instructor is satisfied a policy is stable — walks cleanly,
recovers from pushes, doesn't degrade under reasonable knob ranges — the
upload-to-real-robot step is **entirely** governed by
[`G1_RL_Training_and_Deployment_Guide.md`](../../../G1_RL_Training_and_Deployment_Guide.md)
at the repo root, Sections 5 and 7. This doc does not restate that
guide's safety ritual — read it there. It only summarizes the two upload
*paths* so an instructor can pick one per session, and links onward.

**Not verifiable in a typical dev sandbox.** Like the other two G1
workshops in this project, this step needs the physical G1, a hoist, and a
second person present. Nothing here can be smoke-tested without hardware.

---

## Path A — Run from your PC over Ethernet (recommended first)

The deployment script runs on the instructor's laptop, exchanging
sensor/command messages with the robot over a direct Ethernet cable
(~50 Hz). No file copying needed — the exported policy is already on the
machine running the script.

See root guide §5.2 (network setup) and §5.3 Option A.

## Path B — Upload to the robot's onboard Jetson via SSH, run there

The G1 EDU carries a Jetson Orin on its internal network, commonly at
`192.168.123.164`. From the instructor's PC:

```bash
scp -r unitree_rl_gym unitree@192.168.123.164:~/
scp <your_exported_policy>.pt unitree@192.168.123.164:~/unitree_rl_gym/deploy/
ssh unitree@192.168.123.164
```

One-time setup on the Jetson itself (CPU PyTorch is fine — this network is
tiny): see root guide §5.3 Option B for the exact pip install sequence.

Onboard execution makes the robot self-contained (no tether), but debug
first runs with Path A, where the terminal is visible without a second
screen.

## Either path: the deployment ritual is identical from here

Both paths converge on the same sequence — hoist, debug mode (L2+R2), edit
`deploy/deploy_real/configs/g1.yaml`'s `policy_path`, then:

```bash
python deploy/deploy_real/deploy_real.py <interface> g1.yaml
```

Follow root guide §5.4 exactly, in order. §5.5 covers aborting. §7 covers
kill-switch conditions — read it before anyone is in the room.

## Picking a policy to upload

Whichever policy passed this workshop's sim2sim knobs stage. If
participants generated or browsed multiple candidates, the instructor
picks the one that looked most stable under push-disturbance testing in
`g1_sim` — that judgment call is the point of the live-knobs exercise.
