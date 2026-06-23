#!/usr/bin/env bash
# One-shot environment setup for the G1 live-knobs workshop.
#
# WSL2-only, no Docker (see g1-rl-sim2sim-workshop-PLAN.md §3.2): this
# workshop reuses the existing Kimodo venv at /root/venvs/kimodo rather than
# creating a second one, and runs natively under WSLg (GPU passthrough + X11
# GUI forwarding already proven working there for Kimodo). Run this from
# inside the WSL2 Ubuntu-22.04 distro, as root:
#
#   wsl.exe -d Ubuntu-22.04 -u root -- bash /mnt/e/.../g1-sim2sim-workshop/setup_wsl_env.sh
#
# or copy it to the Linux filesystem first (recommended for anything you'll
# re-run often — see PLAN.md §1.6 on 9p filesystem performance).

set -euo pipefail

VENV=/root/venvs/kimodo

if [ ! -x "$VENV/bin/python3" ]; then
  echo "ERROR: $VENV does not exist or has no python3." >&2
  echo "This script assumes the Kimodo venv from g1-rl-sim2sim-workshop-PLAN.md §1.5 already exists." >&2
  exit 1
fi

source "$VENV/bin/activate"

pip install --upgrade pip
pip install mujoco streamlit pyyaml huggingface_hub

python3 - <<'PYEOF'
import mujoco, streamlit, yaml, huggingface_hub
print("mujoco      ", mujoco.__version__)
print("streamlit   ", streamlit.__version__)
print("pyyaml      ", yaml.__version__)
print("huggingface_hub", huggingface_hub.__version__)
PYEOF

echo "Done. All deps import cleanly inside $VENV."
