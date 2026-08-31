# Real-Robot Upload

This workshop's sim2sim stage (knobs plus the MuJoCo viewer) is the
validation gate. A policy only moves on to hardware once an instructor has
watched it walk cleanly, recover from pushes, and hold up across the knob
ranges people actually tried in the room.

This step needs the physical G1, a hoist, and a second person, so none of it
is something you can smoke-test in a dev sandbox. It's also instructor-only.
Participants never touch this stage.

## Safety setup, every single run

Do this before power-on, not just the first time:

1. Robot suspended on the gantry, feet just off the floor.
2. At least a 4x4 meter clear area, no spectators inside it, cables routed
   so a fall doesn't snag them.
3. Two people: one at the keyboard, one holding the remote controller with
   a thumb ready on the damping/stop sequence.
4. Battery charged on both the robot and the remote.
5. Agree out loud, before powering on, who says "stop" and what happens the
   moment they do.

## Path A: run from your PC over Ethernet (recommended first)

The deployment script runs on the instructor's laptop and exchanges
sensor/command messages with the robot over a direct Ethernet cable, about
50 times a second. No file copying needed, since the exported policy is
already on the machine running the script.

Give your PC a static IP on the robot's subnet (Ubuntu: Settings > Network >
Wired > IPv4 > Manual), then confirm the link:

```bash
ifconfig               # note the interface name, e.g. enp3s0 or eth0
ping <robot-control-pc-ip>
```

The robot's onboard control computer should answer. Check your unit's own
documentation for its exact address rather than assuming a default, since
it can be reconfigured per lab.

## Path B: upload to the robot's onboard Jetson via SSH, run there

The G1 EDU carries a Jetson Orin on its internal network. From the
instructor's PC:

```bash
scp -r unitree_rl_gym unitree@<robot-jetson-ip>:~/
scp <your_exported_policy>.pt unitree@<robot-jetson-ip>:~/unitree_rl_gym/deploy/
ssh unitree@<robot-jetson-ip>
```

Default SSH credentials for the onboard Jetson ship in your G1 EDU's own
documentation, not here. If you haven't already, change the default
password after your first login rather than leaving it in place.

One-time setup on the Jetson itself (CPU PyTorch is fine, since inference
on this small network needs no GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
cd ~/unitree_sdk2_python && pip install -e .
cd ~/unitree_rl_gym && pip install -e .
```

Onboard execution makes the robot self-contained, with no tether, but debug
your first runs with Path A, where the terminal is visible without a second
screen.

## Either path, the deployment ritual is identical from here

1. Hoist the robot so its feet sit just off the ground. Power it on and
   wait for it to settle into zero-torque mode, where the limbs swing
   freely.
2. On the remote controller, press L2 + R2. The robot enters debug mode,
   and the joints go to a damped state. This shuts down Unitree's built-in
   motion service (`ai_sport`) so your policy can command the motors
   directly without the factory controller fighting it. Side effect: the
   normal app and remote walking functions go dead in debug mode. Reboot
   the robot afterward to return it to factory behavior. That's expected,
   not a fault.
3. Edit `deploy/deploy_real/configs/g1.yaml` so `policy_path` points at
   your exported policy. Leave the PD gains and joint mappings alone.
4. Run the deploy script on whichever machine you picked in Path A or B:

   ```bash
   cd unitree_rl_gym
   python deploy/deploy_real/deploy_real.py <interface> g1.yaml
   ```

5. The script walks you through states with remote-button prompts: first
   it drives the joints to the default start posture (the pose training
   assumed), then it waits for your confirmation before starting the
   policy. Follow the on-screen prompts, since they map to controller
   buttons.
6. With the policy running and the robot still hoisted, watch the legs.
   They should make calm, walking-in-air motions in response to gentle
   joystick velocity commands. Violent oscillation means stop immediately
   (see Aborting, below).
7. Lower the hoist until the feet just take weight, keeping slack minimal.
   Let it balance in place. Only after several boring, stable minutes
   across several sessions should you add more slack, try small velocity
   commands, and eventually move to free standing with a spotter.

## Aborting

- The deploy script's stop control (it names the button on-screen) drops
  the robot into a damped state.
- The person holding the remote can always command damping directly.
- Worst case, the hoist holds it. That's why the hoist is non-negotiable.

## Kill switches: when to stop and rethink

Set these tripwires before a session starts, while everyone is calm:

- Joints buzz or oscillate violently at policy start: stop. That's usually
  a policy/config mismatch, an edited gain, or a skipped sim2sim step.
  Re-run the MuJoCo validation and diff `g1.yaml` against git before trying
  again.
- Three consecutive hardware sessions with oscillation at policy start:
  stop hardware work entirely. The bug lives in the export or config
  pipeline, and more hardware attempts only risk the robot.
- Any session where the hoist catches a real fall: end the session there.
  Falls are useful data, but only if you actually stop to read them instead
  of pushing through.

## Picking a policy to upload

Whichever policy passed this workshop's sim2sim knobs stage. If
participants generated or browsed several candidates, the instructor picks
the one that looked most stable under push-disturbance testing in `g1_sim`.
That judgment call is the point of the live-knobs exercise.
