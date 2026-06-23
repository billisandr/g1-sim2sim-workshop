"""
Thin wrapper around Kimodo's `kimodo_gen` CLI for the live-knobs workshop.

Wraps the verified command from g1-rl-sim2sim-workshop-PLAN.md §1.4:

    TEXT_ENCODER_4BIT=1 kimodo_gen "<prompt>" --model Kimodo-G1-RP-v1 \
        --duration <seconds> --num_samples 1 --diffusion_steps 50 \
        --seed <seed> --output <output_stem>

Same environment as the rest of this workshop (/root/venvs/kimodo, WSL2,
no Docker/cross-environment interop — see PLAN.md §3.2), so this is a plain
subprocess call, not a remote/container bridge.

Kimodo's output skeleton (g1skel34, nq=36) is NOT the same as the
locomotion policy's skeleton (g1_12dof, nq=19) — see PLAN.md §3.4 and
config/g1_liveknobs.yaml's motion_clip section. Generated clips only work
in motion-clip playback mode, never as a policy.

Memory safety: the text encoder's load is a real, multi-GB transient even
with the 4-bit GPU patch (see PLAN.md §1.4). On a memory-constrained host,
running this while sim/g1_mujoco_liveknobs.py is also active has caused
full-machine freezes rather than a clean error — generate_motion() checks
for both low memory and a concurrently-running sim up front and refuses
with a clear message instead of risking that.

Every invocation writes a full stdout/stderr logfile plus a short status
report to LOG_DIR, which lives on the WSL2 Linux filesystem (not /mnt/e —
cross-OS 9p file access is slow for repeated I/O, see PLAN.md §1.6). The
same report is returned to the caller so the UI can show it next to the
"Generated: ..." message.
"""

import datetime
import os
import re
import subprocess
import time

import psutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
GENERATED_DIR = os.path.join(PROJECT_ROOT, "motions", "generated")

# Logs live on the WSL2 Linux filesystem, not /mnt/e — see PLAN.md §1.6.
# Override with G1_LIVEKNOBS_LOG_DIR if /root isn't appropriate on a given
# machine.
LOG_DIR = os.environ.get("G1_LIVEKNOBS_LOG_DIR", "/root/g1_liveknobs_logs/kimodo_gen")

DEFAULT_MODEL = "Kimodo-G1-RP-v1"
DEFAULT_DURATION = 3.0
DEFAULT_DIFFUSION_STEPS = 50

MIN_AVAILABLE_RAM_GB = 4.0
SIM_PROCESS_MARKER = "g1_mujoco_liveknobs.py"


class GenerationError(RuntimeError):
    pass


def _slugify(text, max_len=40):
    """'A person walks forward!' -> 'a_person_walks_forward'. Truncates at
    a word boundary rather than mid-word where possible."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len].rsplit("_", 1)[0]
    return slug or "motion"


def _sim_is_running():
    """True if sim/g1_mujoco_liveknobs.py looks like it's running anywhere
    on this machine. The encoder's load is a multi-GB RAM spike, and running
    it alongside the sim (MuJoCo + CPU policy inference) is the combination
    that has caused full-machine freezes on this project's hardware."""
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(SIM_PROCESS_MARKER in part for part in cmdline):
            return True
    return False


def _check_preflight():
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    if available_gb < MIN_AVAILABLE_RAM_GB:
        raise GenerationError(
            f"Only {available_gb:.1f} GB RAM available, need at least "
            f"{MIN_AVAILABLE_RAM_GB:.0f} GB free for the text encoder's load "
            f"transient. Close other apps/browser tabs and try again."
        )
    if _sim_is_running():
        raise GenerationError(
            "g1_mujoco_liveknobs.py looks like it's currently running. The "
            "encoder's load is a multi-GB RAM spike — running it alongside "
            "the sim has caused full-machine freezes on this hardware "
            "before. Close g1_sim, generate the motion, then relaunch g1_sim."
        )


def _write_log(log_path, cmd, result, report):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"command: {' '.join(cmd)}\n\n")
        f.write("--- stdout ---\n")
        f.write(result.stdout or "")
        f.write("\n--- stderr ---\n")
        f.write(result.stderr or "")
        f.write("\n\n=== REPORT ===\n")
        for key, value in report.items():
            f.write(f"{key}: {value}\n")


def generate_motion(
    prompt,
    duration=DEFAULT_DURATION,
    seed=0,
    model=DEFAULT_MODEL,
    diffusion_steps=DEFAULT_DIFFUSION_STEPS,
    output_dir=GENERATED_DIR,
):
    """
    Run kimodo_gen for `prompt`. Returns a dict:

        {"csv_path": ..., "log_path": ..., "report": {...}}

    `report` is a small dict (status, prompt, model, requested motion
    duration, diffusion steps, seed, wall-clock generation time) — the same
    one written to the end of the logfile at `log_path`.

    Blocking — encoder load+encode alone took ~37s in verification (see
    PLAN.md §1.4), before diffusion sampling. Callers driving this from a
    UI should show a spinner/progress message, not assume this returns
    quickly. Intended as a Stage-1 "watch it happen once, live" demo, not
    something every participant runs in parallel (per PLAN.md §3.3).

    Raises GenerationError up front, before spending the ~30-40s on the
    load, if there isn't enough free RAM or if the sim looks like it's
    running concurrently — see the module docstring. A logfile is still
    written for a failure that gets as far as actually invoking kimodo_gen.
    """
    _check_preflight()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{_slugify(prompt)}_{timestamp}_seed{seed}"
    output_path = os.path.join(output_dir, stem)
    log_path = os.path.join(LOG_DIR, f"{stem}.log")

    cmd = [
        "kimodo_gen",
        prompt,
        "--model", model,
        "--duration", str(duration),
        "--num_samples", "1",
        "--diffusion_steps", str(diffusion_steps),
        "--seed", str(seed),
        "--output", output_path,
    ]

    env = dict(os.environ)
    env["TEXT_ENCODER_4BIT"] = "1"

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    generation_time_s = round(time.time() - start, 1)

    csv_path = output_path + ".csv"
    success = result.returncode == 0 and os.path.exists(csv_path)

    report = {
        "status": "success" if success else "failed",
        "prompt": prompt,
        "model": model,
        "motion_duration_s": duration,
        "diffusion_steps": diffusion_steps,
        "seed": seed,
        "started_at": started_at,
        "generation_time_s": generation_time_s,
        "csv_path": csv_path if success else None,
    }
    _write_log(log_path, cmd, result, report)

    if result.returncode != 0:
        raise GenerationError(
            f"kimodo_gen failed (exit {result.returncode}) after "
            f"{generation_time_s}s — full log at {log_path}:\n"
            f"{result.stderr[-2000:]}"
        )
    if not os.path.exists(csv_path):
        raise GenerationError(
            f"kimodo_gen reported success but {csv_path} was not created "
            f"— full log at {log_path}."
        )

    return {"csv_path": csv_path, "log_path": log_path, "report": report}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = generate_motion(args.prompt, duration=args.duration, seed=args.seed)
    print(f"Generated: {result['csv_path']}")
    print(f"Log: {result['log_path']}")
    for key, value in result["report"].items():
        print(f"  {key}: {value}")
