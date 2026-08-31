# Startup Reference

Full run reference for the G1 live-knobs workshop. Everything here runs
inside WSL2 (Ubuntu-22.04). There's no Docker component (see the README's
"Why WSL2 and not Docker" section for the reasoning).

## 1. Prerequisites

- WSL2 distro `Ubuntu-22.04` installed, with a working Python 3.10 venv
  that already has `torch`, Kimodo, and `bitsandbytes` installed. If this
  doesn't exist yet, it's a separate, one-time install, outside the scope
  of this workshop's own setup script.
- An NVIDIA GPU visible inside WSL2 (`nvidia-smi` should work from within
  the distro).
- Hugging Face auth already done inside WSL2 (`hf auth login`), if you plan
  to use real (non-mock) Hugging Face browsing or download gated Kimodo
  checkpoints.

## 2. One-time environment setup

From a Windows terminal (PowerShell or Git Bash):

```bash
wsl.exe -d Ubuntu-22.04 -u root -- bash /path/to/g1-sim2sim-workshop/setup_wsl_env.sh
```

This installs `mujoco`, `streamlit`, `pyyaml`, and `huggingface_hub` into
the existing venv and verifies they import.

## 3. Shell aliases

The setup also expects these aliases in `/root/.bashrc` (added once,
during initial workshop setup):

```bash
export G1_SIM2SIM_DIR=/path/to/g1-sim2sim-workshop
alias g1_sim="source /root/venvs/kimodo/bin/activate && export DISPLAY=:0 && python3 $G1_SIM2SIM_DIR/sim/g1_mujoco_liveknobs.py"
alias g1_ui="source /root/venvs/kimodo/bin/activate && streamlit run $G1_SIM2SIM_DIR/ui/workshop_ui.py --server.headless true --server.port 8501 --server.address 0.0.0.0"
```

If these aren't present, for example on a fresh distro, add them to
`/root/.bashrc` and start a new shell, or `source /root/.bashrc` in an
interactive shell. Aliases don't load in non-interactive scripts.

## 4. Running a session

Open two separate WSL2 terminals.

Terminal 1, the sim:

```bash
g1_sim
```

This opens a MuJoCo passive-viewer window via WSLg. It starts in policy
mode with the pre-staged walking policy and default knob values from
`config/g1_liveknobs.yaml`, and keeps running, re-reading
`sim/runtime_state.json` every control tick until you close the viewer
window.

Terminal 2, the knobs UI:

```bash
g1_ui
```

Prints a URL, `http://localhost:8501`. Open it in a browser. Sliders and
buttons write to `sim/runtime_state.json`, and Terminal 1's sim picks up
changes on the next tick, with no relaunch needed.

## 5. Manual invocation (without the aliases)

```bash
source /root/venvs/kimodo/bin/activate
export DISPLAY=:0
python3 sim/g1_mujoco_liveknobs.py --config config/g1_liveknobs.yaml
```

```bash
source /root/venvs/kimodo/bin/activate
streamlit run ui/workshop_ui.py --server.headless true --server.port 8501 --server.address 0.0.0.0
```

## 6. Headless smoke test (no display needed)

```bash
source /root/venvs/kimodo/bin/activate
python3 tests/smoke_test.py
```

Loads the pre-staged policy, steps 1000 ticks under a fixed forward-walk
command, and asserts the torso stays above a fall-height threshold. Use
this to confirm the policy/XML asset pair is still good without needing
WSLg or a viewer window.

## 7. Generating a new motion with Kimodo directly

```bash
source /root/venvs/kimodo/bin/activate
python3 kimodo_bridge/generate_motion.py "a person walks forward casually" --duration 3.0 --seed 0
```

Prints the path to the generated `.csv` under `motions/generated/`, plus a
log path and a short status report: prompt, model, seed, wall-clock
generation time. The Streamlit UI's "Generate" panel calls this same
function and shows the same report in an expander next to its "Generated:
..." message. Use the CLI directly only for debugging or for
pre-generating extra example clips.

Output and log filenames are derived from the prompt itself, for example
`a_person_walks_forward_casually_20260619_132017_seed0.csv`: a slugified
version of the prompt text, plus timestamp, plus seed, so files stay
identifiable without opening them.

Every invocation also writes a full stdout/stderr logfile to
`/root/g1_liveknobs_logs/kimodo_gen/` inside WSL2, not under the Windows
mount (see section 8 below on why). List recent runs with:

```bash
ls -t /root/g1_liveknobs_logs/kimodo_gen/ | head
```

Override the location with the `G1_LIVEKNOBS_LOG_DIR` environment variable
if `/root` isn't appropriate on a given machine.

## 8. Working on the Linux filesystem during development

If you're iterating on the Python files themselves, not just running them,
copy the working tree to the Linux filesystem first
(`/root/g1_liveknobs_dev/` or similar) rather than editing directly under
the Windows mount. Cross-OS file access over 9p is slow, and it's
especially noticeable on repeated MuJoCo model loads. Sync finished changes
back to the repo when you're done.
