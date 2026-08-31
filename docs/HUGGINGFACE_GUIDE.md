# Hugging Face: How It Works, and How We Used It Here

This doc has two purposes: (1) explain Hugging Face and its Hub well enough
that someone with no prior exposure can navigate it confidently, and (2)
document, in detail, how this workshop's `exptech/g1-moves` motion clips
were found, vetted, and wired in — including the dead ends, so nobody
re-walks them. The dead ends took longer than the part that worked, and
the lessons from them are the most reusable part of this document.

---

## 1. What Hugging Face actually is

Hugging Face is a company and a website (huggingface.co) built around the
**Hub** — a hosting platform for three kinds of repositories:

- **Models** (`huggingface.co/<org>/<name>`) — trained weights, usually
  with a README "model card" describing what the model does, how it was
  trained, and how to load it.
- **Datasets** (`huggingface.co/datasets/<org>/<name>`) — data for
  training or evaluation, with a README "dataset card."
- **Spaces** (`huggingface.co/spaces/<org>/<name>`) — small hosted demo
  apps (Gradio/Streamlit), not relevant to this workshop.

Every repo of every kind is, underneath, **a git repository**. That one
fact explains almost everything else in this document.

---

## 2. How a Hugging Face repo works, mechanically

### 2.1 It's git, plus one extra layer (LFS)

Cloning a HF repo is exactly `git clone https://huggingface.co/<org>/<name>`
(prefix with `datasets/` for a dataset repo). Commits, branches, history —
all normal git. The one wrinkle: large binary files (model weights,
`.onnx`, `.pt`, video previews) are virtually never committed directly.
They're tracked via **Git LFS** (Large File Storage) — a `.gitattributes`
file declares which extensions are LFS-tracked, and git stores a small
**pointer file** in the actual commit (literally just a few lines of text:
a version string, a SHA-256 oid, and a byte size) while the real binary
content lives in a separate LFS storage backend.

**This is the single most important mechanical fact in this whole
document.** A plain `git clone` without LFS support, or an LFS clone where
the *server's* LFS storage is missing/broken, gets you those pointer
files — small, valid-looking text files with the *exact* name of the real
file (`model.onnx`, `policy.pt`, etc.) but containing none of the actual
content. This bit us twice in this session (§5).

### 2.2 The README is structured data, not just prose

Every repo's `README.md` (datasets also get a `DATASET_CARD.md`) starts
with a **YAML frontmatter block** between `---` markers:

```yaml
---
license: mit
tags: [reinforcement-learning, robotics, unitree-g1]
library_name: isaacgym
---
```

This is what populates the repo's tag chips, license badge, and search
filters on the Hub's website. The prose below it is the human-readable
model/dataset card — but as we found in §5.3, **the prose can be wrong or
stale relative to the actual files**. Don't trust the README's claimed
shapes/formats without checking the real files when it matters.

### 2.3 URL patterns you'll actually use

For a repo `<org>/<name>` (prefix `datasets/` for datasets):

| URL pattern | What it gives you |
|---|---|
| `huggingface.co/<org>/<name>` | The repo's web page (README rendered, file browser) |
| `huggingface.co/<org>/<name>/raw/main/<path>` | Raw file content (good for small text files like README.md) |
| `huggingface.co/<org>/<name>/resolve/main/<path>` | **Redirects to the actual LFS-resolved binary** if the file is LFS-tracked and the LFS object actually exists. This is the URL to `curl`/`urllib` for downloading a real file. |
| `huggingface.co/<org>/<name>/blob/main/<path>` | The web page's file-preview view (HTML, not raw content) |
| `huggingface.co/api/models/<org>/<name>` | JSON metadata: file list (`siblings`), tags, license, download/like counts |
| `huggingface.co/api/datasets/<org>/<name>` | Same, for datasets |

**Use `resolve/main/...`, not `raw/main/...`, for any file that might be
LFS-tracked.** `raw` does not resolve LFS pointers in all cases; `resolve`
is built specifically to do that redirect (and is also what fails loudly
when the LFS object is missing, which is the signal you want).

### 2.4 Three ways to actually pull content down

1. **`git clone` + `git lfs pull`** — full repo, full history. Heaviest,
   but gives you everything including files you didn't ask for.
2. **`huggingface_hub` Python library** (`hf_hub_download`,
   `snapshot_download`) — the officially recommended way; handles LFS
   transparently, supports auth tokens for gated repos. This project's
   `hf/hf_browse.py` already wraps this for *searching*; for actual
   downloads of *known* files, `download_and_convert()` in
   `hf/g1_moves_convert.py` (§4) uses a plain `urllib.request.urlretrieve`
   against a `resolve/main/...` URL instead, since we only ever need one
   specific known file, not the SDK's discovery features.
3. **Direct `curl`/`wget` against a `resolve/main/...` URL** — simplest
   for a single known file, no library dependency. What we used for all
   the investigation in §5 and the actual clip downloads in §4.

### 2.5 Gated and private repos

Some repos (especially from NVIDIA, Meta, etc.) require accepting a
license click-through on the website before they're downloadable — these
return 401/403 until you authenticate with a token that has accepted the
gate. This project's `hf/hf_browse.py` already has a mock-mode convention
for this (`MOCK_HF=1` default, see that file's docstring) — **never guess
a real gated repo ID**, always have a human verify it actually exists and
accept its terms first. Not relevant to `exptech/g1-moves` (it's
ungated, CC-BY-4.0), but relevant if you extend this to other sources.

---

## 3. Deep dive: `exptech/g1-moves`

**URL:** https://huggingface.co/datasets/exptech/g1-moves
**License:** CC-BY-4.0 (free to use, attribution required)
**Size:** 64 clips, 29.5 minutes total at 60 FPS, captured from real human
performers in Austin, TX using MOVIN TRACIN markerless motion capture
(LiDAR + vision) and retargeted onto a Unitree G1 model via
[movin_sdk_python](https://github.com/MOVIN3D/movin_sdk_python).

### 3.1 Categories (confirmed by checking the actual folder structure, not just the README's claimed counts)

| Category | Folder | Count | Examples |
|---|---|---|---|
| Dance | `dance/` | 28 | `J_Dance0_StepTouch` ... `J_Dance23_MidnightSun`, `J_ShortDance13`-`16`, `B_DadDance`, `B_WiggleDance` |
| Karate | `karate/` | 27 | `M_Move1`-`20`, `M_ShortMove12`-`16`, `B_BowKarate`, `B_AttackKarate` |
| Bonus | `bonus/` | 6 (dataset card says 5, one was apparently added since) | `B_Fence1`, `B_Fence2`, `B_HandsChop`, `B_HandsUp`, `V_PullOver`, `V_Rocamena` |

### 3.2 Per-clip pipeline structure

Each clip lives at `<category>/<clip_name>/` with four pipeline stages:

```
<category>/<clip>/
  capture/      raw mocap: .bvh (51-joint humanoid skeleton, 60fps, Y-up), preview .gif/.mp4, .fbx for Blender/Maya/UE/Unity
  retarget/     retargeted onto the G1: .pkl (python pickle), .csv (same data, no header), preview .gif/.mp4
  training/     .npz with forward-kinematics-derived training data (joint pos/vel, body pos/quat/lin_vel/ang_vel in world frame)
  policy/       a per-clip trained MOTION-IMITATION policy: .onnx, .pt, agent.yaml, env.yaml, training_log.csv
```

**We only used the `retarget/<clip>.csv` files.** The `policy/` files are
a different thing entirely — see §3.4.

### 3.3 The retargeted CSV format (what we actually load)

36 columns, no header, one row per frame at 60 FPS:

| Columns | Content |
|---|---|
| 0-2 | Root position (x, y, z), meters |
| 3-6 | Root orientation as a quaternion, **in xyzw order** |
| 7-35 | 29 joint angles, radians, in a fixed order (legs → waist → left arm → right arm) |

**The critical gotcha:** MuJoCo's `qpos` convention for a free joint is
**wxyz** (scalar-first), not xyzw (scalar-last). Loading this dataset's
quaternion columns directly into `qpos` without reordering produces a
valid-*looking* quaternion (same magnitude) that represents the **wrong
rotation** — the robot's orientation comes out scrambled, not an obvious
crash, just visibly wrong. `hf/g1_moves_convert.py`'s
`convert_qpos_xyzw_to_wxyz()` does this one reorder and nothing else.

The 29-joint order (verified joint-for-joint, in order, against
`assets/g1_description/g1_29dof.xml`'s own joint listing — see §5.4 for
how that was confirmed, not assumed):

```
0  left_hip_pitch     15 left_shoulder_pitch
1  left_hip_roll      16 left_shoulder_roll
2  left_hip_yaw       17 left_shoulder_yaw
3  left_knee          18 left_elbow
4  left_ankle_pitch   19 left_wrist_roll
5  left_ankle_roll    20 left_wrist_pitch
6  right_hip_pitch    21 left_wrist_yaw
7  right_hip_roll     22 right_shoulder_pitch
8  right_hip_yaw      23 right_shoulder_roll
9  right_knee         24 right_shoulder_yaw
10 right_ankle_pitch  25 right_elbow
11 right_ankle_roll   26 right_wrist_roll
12 waist_yaw          27 right_wrist_pitch
13 waist_roll         28 right_wrist_yaw
14 waist_pitch
```

### 3.4 The per-clip "policy" files — now wired in as a third sim-loop mode

Each clip also ships a **motion-imitation policy** (`<clip>_policy.pt`/
`.onnx`) — an RL policy trained (via a framework called `mjlab`, per the
`env.yaml`'s `spec_fn: mjlab.asset_zoo.robots.unitree_g1.g1_constants.get_spec`)
to track *that one specific reference clip*. This is a fundamentally
different kind of policy than the one this workshop's "Live policy" mode
already supports:

- Our existing policy (`motions/pre_staged/g1_walk_policy.pt`) takes
  **velocity commands** (vx/vy/yaw) and walks accordingly — a general
  locomotion controller.
- These per-clip policies take **no velocity commands at all** — they're
  trained to reproduce one fixed reference trajectory, correcting for
  disturbances along the way.

This is now a third, fully wired sim-loop mode (`"motion_tracking"` in
`sim/runtime_state.json`) — not hypothetical anymore. `sim/motion_tracking.py`
has the full story: the checkpoint is a raw RSL-RL training checkpoint (not
the TorchScript export the other policy uses — `torch.jit.load()` fails on
it), and its real 160-dim observation had to be decoded dimension-for-
dimension from mjlab's actual source rather than trusted from the YAML
(58 = reference joint pos/vel, 3+6 = anchor position/orientation relative
to the robot, 3+3 = body velocities, 29+29 = the robot's own relative
joint pos/vel, 29 = last action). Two non-obvious fixes were needed to get
a stable sim, both because `g1_29dof.xml` (built for *kinematic* clip
playback, never actuated before this) ships with no `<option>` and zero
joint armature: the explicit-Euler default integrator is unstable at
these PD gains (switch to `implicitfast`, matching `env.yaml`), and zero
armature means far less effective inertia than the policy was trained
against (set `m.dof_armature` from the same `env.yaml`'s per-joint
values). Get one of these wrong and the sim doesn't error — it quietly
explodes to NaN within ~10 ticks or collapses into a stable-but-wrong
crouch, so a real headless physics run (`tests/smoke_test_motion_tracking.py`),
not just "obs shapes match", is what actually catches it.

Download a tracking policy with:
```bash
python3 hf/g1_moves_tracking_convert.py <ClipName> <category>
```
which fetches `policy/<ClipName>_policy.pt` and `training/<ClipName>.npz`
into `motions/pre_staged/tracking/<ClipName>/` — picked up automatically by
the UI's tracking-policy selector, the same "drop a file in, no per-clip
code" convention `hf/g1_moves_convert.py` already uses for clips.

---

## 4. What we actually built: `hf/g1_moves_convert.py`

```bash
source /root/venvs/kimodo/bin/activate
cd g1-sim2sim-workshop
python3 hf/g1_moves_convert.py <ClipName> <category>
# e.g.: python3 hf/g1_moves_convert.py J_Dance2_Salsa dance
```

This downloads `<category>/<ClipName>/retarget/<ClipName>.csv` from the
dataset, reorders the quaternion columns, and writes
`motions/pre_staged/g1_moves_<ClipName>.csv` — which then shows up
automatically in the workshop UI's motion-clip dropdown (`list_motion_clips()`
in `ui/workshop_ui.py` just lists every `.csv` in `motions/pre_staged/`
and `motions/generated/`; no per-clip UI code is needed).

The skeleton it needs (`assets/g1_description/g1_29dof.xml`, the real
Unitree 29-DOF G1, already present in this repo alongside the original
locomotion policy) is resolved automatically by
`sim/g1_mujoco_liveknobs.py`'s `_resolve_skeleton_xml()`,
which matches the filename prefix `g1_moves_` against
`config/g1_liveknobs.yaml`'s `motion_clip.skeleton_by_prefix` list. This
matters because **Kimodo's motion clips and these clips both happen to
have `nq=36`** despite being completely different, incompatible skeletons
(Kimodo's own 34-joint animation rig vs. the G1's real 29 actuated
joints) — nq alone can't tell them apart, hence the filename-based lookup
instead of trying to infer it from the data.

Currently converted (9 clips, in `motions/pre_staged/`):

| File | Category | Duration |
|---|---|---|
| `g1_moves_M_ShortMove12.csv` | karate | 7.8s |
| `g1_moves_M_ShortMove13.csv` | karate | 9.0s |
| `g1_moves_M_ShortMove15.csv` | karate | 6.8s |
| `g1_moves_J_ShortDance14_Disco.csv` | dance | 14.3s |
| `g1_moves_J_ShortDance16_JazzWalk.csv` | dance | 10.6s |
| `g1_moves_B_WiggleDance.csv` | dance | 37.3s |
| `g1_moves_B_HandsUp.csv` | bonus | 7.4s |
| `g1_moves_V_Rocamena.csv` | bonus | 4.8s |
| `g1_moves_B_Fence1.csv` | bonus | 27.0s |

55 clips remain unconverted — adding any of them is the one-line command
above, no further code changes needed.

---

## 5. Lessons learned (the dead ends — read before repeating them)

This session investigated three other Hugging Face/GitHub sources before
landing on `g1-moves`. None of them panned out, for three completely
different reasons. Knowing *why* each failed saves re-deriving it.

### 5.1 `ioai-tech/onnx_policy` (GitHub, not HF) — LFS objects don't exist server-side

A GitHub repo claiming ONNX policies for G1/X2 robots, with a README that
gave exact, plausible-looking metadata (`num_observations: 47`,
`num_actions: 12`, matching kp/kd/default-angles bit-for-bit identical to
our own policy's config). Looked extremely promising. **Every single LFS
object in the entire repo 404s on GitHub's LFS server** — confirmed three
independent ways (the GitHub Contents API, the LFS media CDN, and a literal
`git lfs install && git clone`, which fails with `fatal: ... smudge filter
lfs failed`). The repo's git tree only contains ~130-byte LFS pointer text
files. This is unfixable client-side — the maintainer never pushed (or
later deleted) the actual binaries. **Lesson: when a repo's binary file is
suspiciously small (low hundreds of bytes) for what it claims to be,
check if it's an LFS pointer (`file <path>` will say "ASCII text"; a real
binary won't) before spending any more time on it.**

### 5.2 Org name typos look like network hangs

A `git clone` against `IO-AI-TECH` (hyphenated) instead of the real
`ioai-tech` (no hyphens) didn't fail fast — it hung indefinitely on DNS/
connection resolution rather than erroring immediately. **Lesson: if a
clone or download hangs with no progress for an unreasonable time, verify
the org/repo name resolves at all (`curl -s api/users/<name>` or similar)
before assuming it's a slow network.**

### 5.3 `hardware-pathon-ai/unitree-g1-phase1-locomotion` (real HF model) — README disagrees with the actual file

A real, public, ungated, MIT-licensed model with a `policy_jit.pt`
(TorchScript, loadable, same format our sim loop already expects).
README claimed "29 continuous actions." **Inspecting the actual
`torch.jit.load()`'d module's weight tensor shapes directly** showed the
real input/output is **77 in, 22 out** — neither the README's "29 total
DOF" nor its "15 active DOF" claim. The README and the file disagree with
each other, and no joint-name list is published to know what those 22
actions even map to. **Lesson: never trust a model card's claimed
shapes — load the actual file and read `named_parameters()`'s tensor
shapes directly. It takes seconds and is unambiguous, unlike prose.**

### 5.4 `exptech/g1-moves` — the one that worked, and why it was trustworthy

Real download counts (3203), a complete and *internally consistent*
documentation set (`DATASET_CARD.md` gives an exact joint table; `env.yaml`
per clip gives exact PD gains referencing a real, named training
framework), and — critically — we **independently verified** the claimed
29-joint order against `g1_29dof.xml`'s actual joint listing (`grep -oP`
the XML, don't just trust either side's prose) before trusting it. They
matched exactly, joint-for-joint. That independent cross-check, not the
dataset's reputation or download count, is what actually justified
proceeding.

---

## 6. Reusable prompt: do this again for a different HF/GitHub source

Paste the block below (filling in the bracketed parts) to brief a fresh
agent session with no prior context on replicating this exact
investigate-then-integrate workflow for a *different* motion/policy
source. It encodes the due-diligence steps from §5 as explicit
instructions, not just a description of what happened.

> I want to add motion clips or a policy from `[HF_OR_GITHUB_URL]` to the
> `g1-sim2sim-workshop` repo. Read `docs/HUGGINGFACE_GUIDE.md` in that repo
> first, since it documents how Hugging Face repos work mechanically and
> the exact due-diligence steps that already caught three dead-end sources
> in this project (broken LFS
> hosting, a hung clone from a typo'd org name, and a model card that
> disagreed with its own file's actual tensor shapes). Don't skip these
> checks just because the source looks reputable.
>
> Before writing any integration code:
> 1. Fetch the repo's metadata via the HF API (`/api/models/<org>/<name>`
>    or `/api/datasets/<org>/<name>`) or `gh api repos/<org>/<name>` for
>    GitHub. Confirm the org/repo name resolves (a hang, not just an
>    error, can mean a typo'd name).
> 2. List the actual files (the `siblings` field for HF, or `gh api
>    repos/<org>/<name>/contents/<path>` for GitHub). Identify which files
>    are LFS-tracked (check `.gitattributes` or just try downloading one).
> 3. Download ONE real file via a `resolve/main/...` URL (HF) or the LFS
>    media CDN / a real `git clone` + `git lfs pull` (GitHub). Check with
>    `file <path>` that it's NOT "ASCII text" (i.e., not an unresolved LFS
>    pointer) before trusting its size or content.
> 4. If it's a model/policy: load it for real (`torch.jit.load`, `onnx`
>    inspection, etc.) and read the **actual** input/output tensor shapes
>    from the loaded object — not the README's claimed shapes. Compare
>    against this project's existing policy contract: 12 actuated leg
>    joints, 47-dim observation (`config/g1_liveknobs.yaml`'s `policy:`
>    section), TorchScript-loadable via `torch.jit.load`. A mismatch means
>    real adapter work, not a quick swap — say so explicitly rather than
>    assuming compatibility.
> 5. If it's motion/animation data: check the documented joint order and
>    quaternion convention (scalar-first wxyz vs. scalar-last xyzw) against
>    whatever MuJoCo skeleton XML you intend to play it back on. Confirm
>    by `grep`-ing the actual XML's joint list and comparing it against the
>    dataset's documented order yourself — don't take either side's word
>    for it. Confirm `nq` matches the data's column count, but remember nq
>    matching is NOT sufficient proof of skeleton compatibility (two
>    different skeletons can coincidentally share the same nq — resolve
>    skeleton identity by filename/source, the way
>    `config/g1_liveknobs.yaml`'s `skeleton_by_prefix` already does for
>    Kimodo vs. g1-moves clips).
> 6. Only after all of the above pass, write the actual integration code
>    (a converter script under `hf/`, a new `skeletons` entry in
>    `config/g1_liveknobs.yaml` if it's a new skeleton, and a matching
>    `skeleton_by_prefix` rule) and verify end-to-end in `g1_sim` — start
>    the sim in its default policy mode, then live-switch
>    `sim/runtime_state.json` to point at the new clip/policy while
>    watching the sim's own stdout for its confirmation print statements
>    (not just "no crash" — confirm the actual skeleton/clip name and
>    frame count it reports match what you expect).
>
> Report back what you found at each step, including anything that didn't
> check out — a source that fails this due diligence is a legitimate
> answer, not a failure to find something usable.
