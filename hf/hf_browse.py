"""
Hugging Face Hub browsing helper for the G1 live-knobs workshop.

Set MOCK_HF=1 (the default) to run fully offline against canned fixture
data — no Hugging Face account, token, or internet connection needed.
Set MOCK_HF=0 for the real Hugging Face Hub API (the user's existing
`hf auth login` session is reused automatically — see
g1-rl-sim2sim-workshop-PLAN.md §1.2).

Ported from g1-policy-literacy-workshop/workshop/hf_helper.py — duplicated
rather than imported across workshops so this one stays self-contained.
"""

import os

MOCK_HF = os.environ.get("MOCK_HF", "1") != "0"

# ---------------------------------------------------------------------------
# Illustrative fixture data ONLY — these are NOT real Hugging Face repo IDs.
#
# Before any live demo with a real gated token, an instructor must look up
# the actual current model page and swap in the real ID. Never guess a real
# model ID/URL in front of participants — model pages move and gating terms
# change.
# ---------------------------------------------------------------------------
_MOCK_MODELS = [
    {
        "id": "example-org/g1-velocity-walk-rl",
        "tags": ["robotics", "locomotion", "g1", "rl-policy"],
        "gated": False,
        "license": "bsd-3-clause",
        "embodiment": "Unitree G1, 12 leg joints, no end-effector",
        "action_dim": 12,
        "description": (
            "Illustrative stand-in for a published G1 velocity-walking RL "
            "policy, similar in shape to unitree_rl_gym's pre-trained checkpoint."
        ),
    },
    {
        "id": "example-org/groot-style-manipulation-vla",
        "tags": ["robotics", "manipulation", "vla", "gated-example"],
        "gated": True,
        "license": "example-gated-model-license",
        "embodiment": "dual-arm manipulator with parallel-jaw gripper",
        "action_dim": 32,
        "description": (
            "Illustrative stand-in for a gated vision-language-action "
            "foundation policy, similar in shape to NVIDIA's GR00T family. "
            "Not compatible with this G1 build (it has no grippers)."
        ),
    },
    {
        "id": "example-org/quadruped-velocity-walk",
        "tags": ["robotics", "locomotion", "quadruped"],
        "gated": False,
        "license": "mit",
        "embodiment": "quadruped, 12 leg joints, no end-effector",
        "action_dim": 12,
        "description": (
            "Illustrative stand-in for a published quadruped locomotion "
            "policy. Same action_dim as the G1 build by coincidence, but "
            "the wrong embodiment — a good example for the compatibility "
            "lesson from g1-policy-literacy-workshop."
        ),
    },
]


def search_policies(tags):
    """
    Search the Hugging Face Hub for models matching ALL of `tags`.

    tags: an iterable of tag strings, e.g. ["robotics", "locomotion"].

    Returns a list of dicts, each with at least "id", "tags", and "gated".
    In mock mode (default) this searches the fixture list above. In real
    mode it calls the live Hugging Face Hub API.
    """
    query_tags = {t.lower() for t in tags}

    if MOCK_HF:
        return [
            m for m in _MOCK_MODELS
            if query_tags.issubset({t.lower() for t in m["tags"]})
        ]

    from huggingface_hub import HfApi
    api = HfApi()
    results = []
    for model in api.list_models(tags=list(query_tags), limit=20):
        results.append({
            "id": model.id,
            "tags": model.tags or [],
            "gated": bool(getattr(model, "gated", False)),
        })
    return results


def get_model_card(model_id):
    """
    Return the embodiment/action-space metadata for `model_id`.

    In mock mode this looks up the fixture by id. Real-mode model-card
    inspection (reading config.json / README front-matter on the live Hub)
    is an instructor-led live-demo extension.
    """
    if MOCK_HF:
        for m in _MOCK_MODELS:
            if m["id"] == model_id:
                return m
        raise KeyError(
            f"No mock model card for {model_id!r}. "
            f"Pick one of the ids returned by search_policies()."
        )

    raise NotImplementedError(
        "Real-mode model card inspection is an instructor-led live demo "
        "— not required for the live-knobs workshop flow."
    )


def download_model(model_id, local_dir):
    """
    Download `model_id`'s repo files into `local_dir`.

    Mock mode refuses (there is nothing real to download for a fixture
    entry). Real mode uses the caller's existing `hf auth login` session.
    """
    if MOCK_HF:
        raise NotImplementedError(
            f"{model_id!r} is mock fixture data, not a real Hugging Face "
            f"repo — there is nothing to download. Set MOCK_HF=0 to browse "
            f"and download real models."
        )

    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model_id, local_dir=local_dir)
