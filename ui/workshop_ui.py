#!/usr/bin/env python3
"""
Workshop UI — live knobs for the G1 sim2sim demo.

A browser-based control panel (Streamlit) for non-coders. Sliders and
buttons write straight to sim/runtime_state.json; sim/g1_mujoco_liveknobs.py
re-reads that file every control tick, so the robot's behaviour changes
WITHOUT relaunching the simulation (same file-driven philosophy as
ros_z1_sim_marker-real-camera's workshop_ui.py, which writes to the ROS
parameter server instead — see g1-rl-sim2sim-workshop-PLAN.md §3.3).

Run from WSL2, in a separate terminal from the sim:

    g1_ui            # alias -> streamlit run .../ui/workshop_ui.py

then open the printed URL (http://localhost:8501).
"""

import json
import os
import sys
import time

import streamlit as st
import yaml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

HERO_VIDEO_NAME = "g1_dance_6sec.mp4"
HERO_VIDEO_PATH = os.path.join(THIS_DIR, "static", HERO_VIDEO_NAME)

from hf.hf_browse import search_policies, get_model_card  # noqa: E402

CONFIG_PATH = os.environ.get(
    "G1_LIVEKNOBS_CONFIG", os.path.join(PROJECT_ROOT, "config", "g1_liveknobs.yaml")
)

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

RUNTIME_STATE_PATH = os.path.join(PROJECT_ROOT, CFG["runtime_state_path"])
PRE_STAGED_DIR = os.path.join(PROJECT_ROOT, "motions", "pre_staged")
GENERATED_DIR = os.path.join(PROJECT_ROOT, "motions", "generated")
TRACKING_DIR = os.path.join(PROJECT_ROOT, "motions", "pre_staged", "tracking")

DEFAULT_RUNTIME_STATE = {
    "mode": "policy",
    "vx": 0.5,
    "vy": 0.0,
    "yaw": 0.0,
    "pd_scale": 1.0,
    "action_scale_override": None,
    "push_impulse": 150.0,
    "push_requested": False,
    "motion_path": None,
    "motion_paused": False,
    "motion_scrub_frame": None,
    "tracking_dir": None,
}

# -- exercise presets: (label, {key: value}, caption) --------------------
PRESETS_TEMPERAMENT = [
    ("Balanced", {"pd_scale": 1.0, "action_scale_override": None}, "default, stable walk"),
    ("Floppy", {"pd_scale": 0.4}, "joints go soft — watch it sag"),
    ("Stiff", {"pd_scale": 1.8}, "overly rigid — jerky correction"),
    ("Calm gait", {"action_scale_override": 0.10}, "small, careful steps"),
    ("Twitchy gait", {"action_scale_override": 0.50}, "large, jumpy steps"),
]


def load_state():
    try:
        with open(RUNTIME_STATE_PATH) as f:
            state = json.load(f)
        merged = dict(DEFAULT_RUNTIME_STATE)
        merged.update(state)
        return merged
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_RUNTIME_STATE)


def save_state(state):
    os.makedirs(os.path.dirname(RUNTIME_STATE_PATH), exist_ok=True)
    with open(RUNTIME_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def apply_changes(changes):
    state = load_state()
    state.update(changes)
    save_state(state)


def list_motion_clips():
    clips = []
    for d in (PRE_STAGED_DIR, GENERATED_DIR):
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".csv"):
                clips.append(os.path.relpath(os.path.join(d, fname), PROJECT_ROOT))
    return clips


def list_policies():
    policies = []
    if os.path.isdir(PRE_STAGED_DIR):
        for fname in sorted(os.listdir(PRE_STAGED_DIR)):
            if fname.endswith(".pt"):
                policies.append(os.path.relpath(os.path.join(PRE_STAGED_DIR, fname), PROJECT_ROOT))
    return policies


def list_tracking_policies():
    """Each subfolder of motions/pre_staged/tracking/ with both policy.pt
    and motion.npz is one selectable imitation-tracking routine."""
    dirs = []
    if os.path.isdir(TRACKING_DIR):
        for name in sorted(os.listdir(TRACKING_DIR)):
            full = os.path.join(TRACKING_DIR, name)
            if os.path.isfile(os.path.join(full, "policy.pt")) and os.path.isfile(os.path.join(full, "motion.npz")):
                dirs.append(os.path.relpath(full, PROJECT_ROOT))
    return dirs


def main():
    st.set_page_config(page_title="G1 Sim2Sim", page_icon=":robot_face:", layout="wide")

    title_col, video_col = st.columns([4, 1])
    with title_col:
        st.title("Make the G1 Walk or Dance?")
        st.caption("Sim2Sim — RL policy — Imitation Learning")
    with video_col:
        if os.path.isfile(HERO_VIDEO_PATH):
            # Requires the app to be launched with static serving enabled
            # (STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true) — see the g1_ui
            # alias. Plain in-flow element next to the title — it scrolls
            # away with the page like any other content, which is fine; no
            # more fighting position:fixed across Streamlit's containers.
            st.markdown(
                f"""
                <style>
                .g1-hero-video {{
                    width: 90%; aspect-ratio: 9 / 9; object-fit: cover;
                    border-radius: 16px; box-shadow: 0 5px 14px rgba(0,0,0,.35);
                }}
                </style>
                <video class="g1-hero-video" autoplay loop muted playsinline>
                    <source src="app/static/{HERO_VIDEO_NAME}" type="video/mp4">
                </video>
                """,
                unsafe_allow_html=True,
            )

    state = load_state()

    knobs = CFG["knobs"]

    selector_tab, drive_tab, temperament_tab, break_tab = st.tabs(
        ["Pick a motion/policy", "Drive the robot", "Tune the temperament", "Break it on purpose"]
    )

    # ===================== Policy / motion-clip selector ===================
    with selector_tab:
        st.header("What's the robot running?")
        mode = st.radio(
            "Mode",
            ["policy", "motion_clip", "motion_tracking"],
            index=["policy", "motion_clip", "motion_tracking"].index(state["mode"])
            if state["mode"] in ("policy", "motion_clip", "motion_tracking") else 0,
            format_func=lambda m: {
                "policy": "Live policy",
                "motion_clip": "Recorded motion clip",
                "motion_tracking": "Imitation-tracking policy",
            }[m],
            horizontal=True,
        )
        if mode != state["mode"]:
            apply_changes({"mode": mode})
            state = load_state()

        if mode == "policy":
            policies = list_policies()
            if policies:
                st.selectbox("Policy checkpoint", policies, key="policy_select")
            st.info(
                "A policy is a neural net trained to react to vx/vy/yaw commands. "
                "Use the **Drive the robot** tab."
            )
        elif mode == "motion_clip":
            clips = list_motion_clips()
            chosen = st.selectbox("Motion clip", clips, key="clip_select") if clips else None
            if chosen and chosen != state.get("motion_path"):
                apply_changes({"motion_path": chosen, "motion_scrub_frame": None})
            st.info(
                "A recorded clip is a fixed sequence of poses — it does not listen to "
                "vx/vy/yaw. Play/pause and scrub below instead."
            )
            paused = st.checkbox("Paused", value=state.get("motion_paused", False))
            if paused != state.get("motion_paused"):
                apply_changes({"motion_paused": paused})
            if st.button("Scrub to start"):
                apply_changes({"motion_scrub_frame": 0})
        else:
            tracking_dirs = list_tracking_policies()
            chosen = st.selectbox("Tracking policy", tracking_dirs, key="tracking_select") if tracking_dirs else None
            if chosen and chosen != state.get("tracking_dir"):
                apply_changes({"tracking_dir": chosen})
            if not tracking_dirs:
                st.warning(
                    "No tracking policies found under motions/pre_staged/tracking/. "
                    "Download one with: python3 hf/g1_moves_tracking_convert.py <ClipName> <category>"
                )
            st.info(
                "This is neither of the other two modes: it's a real PD-driven simulation "
                "(reacts to pushes, in **Break it on purpose**), but it was trained to perform "
                "**one specific reference routine** — it does not listen to vx/vy/yaw, the "
                "same as a recorded clip. Drag **pd_scale** in **Tune the temperament** to feel "
                "the same stiffness/floppiness knob apply to this mode too."
            )

        st.divider()
        st.subheader("Browse Hugging Face for a policy")
        st.caption(
            "Mock mode by default (MOCK_HF=1) — searches illustrative fixture data, "
            "not the real Hub. Set MOCK_HF=0 for the instructor's live demo."
        )
        tag_query = st.text_input("Search tags (comma-separated)", value="robotics,locomotion")
        if st.button("Search"):
            tags = [t.strip() for t in tag_query.split(",") if t.strip()]
            results = search_policies(tags)
            st.session_state["hf_results"] = results
        for m in st.session_state.get("hf_results", []):
            with st.expander(f"{m['id']}  ({'gated' if m['gated'] else 'open'})"):
                st.write(m["description"])
                st.write(f"Embodiment: {m['embodiment']}  |  action_dim: {m['action_dim']}")

        st.divider()
        st.subheader("Generate a new motion with Kimodo")
        st.caption(
            "Takes real time — encoder load+encode alone took ~37s in verification, "
            "before diffusion sampling. Best as a 'watch it happen once, live' demo, "
            "not something everyone runs in parallel."
        )
        st.warning(
            "**Close g1_sim before generating.** The text encoder's load is a multi-GB "
            "RAM spike — running it while the sim is also open has caused full-machine "
            "freezes on memory-constrained hardware. Generate the motion first, then "
            "relaunch g1_sim to play it back."
        )
        prompt = st.text_input("Prompt", value="a person walks forward casually")
        duration = st.slider("Duration (s)", 1.0, 6.0, 3.0, 0.5)
        if st.button("Generate (this will take a minute or two)"):
            from kimodo_bridge.generate_motion import generate_motion, GenerationError
            with st.spinner("Loading text encoder and sampling motion..."):
                try:
                    result = generate_motion(prompt, duration=duration)
                    st.success(f"Generated: {os.path.relpath(result['csv_path'], PROJECT_ROOT)}")
                    with st.expander("Generation report", expanded=True):
                        st.json(result["report"])
                        st.caption(f"Full log (in WSL2): {result['log_path']}")
                except GenerationError as e:
                    st.error(str(e))

    # ===================== Group A — Drive the robot ========================
    with drive_tab:
        if state["mode"] != "policy":
            st.warning("Switch to **Live policy** mode in the selector tab to use these knobs.")
        vx_lo, vx_hi, vx_step, vx_def = knobs["vx"]["range"]
        vx = st.slider(knobs["vx"]["label"], vx_lo, vx_hi, float(state.get("vx", vx_def)), vx_step,
                        disabled=state["mode"] != "policy")

        vy_lo, vy_hi, vy_step, vy_def = knobs["vy"]["range"]
        vy = st.slider(knobs["vy"]["label"], vy_lo, vy_hi, float(state.get("vy", vy_def)), vy_step,
                        disabled=state["mode"] != "policy")

        yaw_lo, yaw_hi, yaw_step, yaw_def = knobs["yaw"]["range"]
        yaw = st.slider(knobs["yaw"]["label"], yaw_lo, yaw_hi, float(state.get("yaw", yaw_def)), yaw_step,
                         disabled=state["mode"] != "policy")

        if state["mode"] == "policy":
            apply_changes({"vx": vx, "vy": vy, "yaw": yaw})

    # ===================== Group B — Tune the temperament ===================
    with temperament_tab:
        pd_lo, pd_hi, pd_step, pd_def = knobs["pd_scale"]["range"]
        pd_scale = st.slider(knobs["pd_scale"]["label"], pd_lo, pd_hi,
                              float(state.get("pd_scale", pd_def)), pd_step,
                              help=knobs["pd_scale"].get("help"))

        as_lo, as_hi, as_step, as_def = knobs["action_scale"]["range"]
        action_scale = st.slider(knobs["action_scale"]["label"], as_lo, as_hi,
                                  float(state.get("action_scale_override") or as_def), as_step,
                                  help=knobs["action_scale"].get("help"))

        apply_changes({"pd_scale": pd_scale, "action_scale_override": action_scale})

        st.write("**Quick presets:**")
        p_cols = st.columns(3)
        for i, (label, changes, cap) in enumerate(PRESETS_TEMPERAMENT):
            with p_cols[i % 3]:
                if st.button(label, key=f"temp_{label}", use_container_width=True):
                    apply_changes(changes)
                    st.toast(f"{label}: {cap}")
                    st.rerun()

    # ===================== Group C — Break it on purpose =====================
    with break_tab:
        pushable = state["mode"] in ("policy", "motion_tracking")
        if not pushable:
            st.warning("Switch to **Live policy** or **Imitation-tracking policy** mode to push the robot — "
                       "a recorded clip has no physics to push against.")
        push_lo, push_hi, push_step, push_def = knobs["push_impulse"]["range"]
        push_impulse = st.slider(knobs["push_impulse"]["label"], push_lo, push_hi,
                                  float(state.get("push_impulse", push_def)), push_step)
        apply_changes({"push_impulse": push_impulse})
        if st.button("PUSH", use_container_width=True, disabled=not pushable):
            apply_changes({"push_requested": True})
            st.toast("Push applied to the torso")

    st.divider()
    if st.button("Reset to defaults"):
        save_state(dict(DEFAULT_RUNTIME_STATE))
        st.toast("Reset to default values")
        st.rerun()


if __name__ == "__main__":
    main()
