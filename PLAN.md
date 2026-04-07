# Plan: Improve Canvas World Model VLM Scoring

## Context

The canvas world model predicts next-frame images for candidate robot actions (move right, move left, hold). A VLM (Gemma 4 E4B) compares the predicted frames to decide which action best accomplishes a task. 

**Current problems:**
1. VLM only sees predicted base camera — missing the wrist camera which shows the claw and what's in front of it
2. Task description says "center in camera view" but there are two cameras — confusing
3. VLM can't reliably distinguish predicted frames with only one viewpoint

**Goal:** Send both camera views to the VLM with clear labels, use directional language (right/left/hold), and change the task to be about pointing the claw at the object.

**No randomization** for now — keep image order fixed so reports are easier to read and debug.

---

## Changes

### 1. Extract both camera views from predictions

**File:** `canvas-robot-control/control/world_model.py` — `predict_batch()`

Currently `extract_workspace_view()` returns only the top 224x224 (base camera). Change to return **both** views:
- Base camera view: top 224x224 of predicted frame (overhead)  
- Wrist camera view: bottom 224x224 of predicted frame (claw view)

Return format: `list[tuple[np.ndarray, np.ndarray]]` — 3 pairs of (base, wrist).

### 2. Update scorer with both cameras, clear labels, no randomization

**File:** `canvas-robot-control/scorers/gemma4_comparative.py` — `score_frames()`

Send **8 images** to Gemma 4 in fixed order:

```
These are images from a robot with two cameras:
- Overhead camera: shows the arm and table from above
- Wrist camera: mounted on the wrist, shows the claw and what's in front of it

Current view:
- Image 1: overhead camera NOW
- Image 2: wrist camera NOW

If the arm moves RIGHT:
- Image 3: predicted overhead camera
- Image 4: predicted wrist camera

If the arm moves LEFT:
- Image 5: predicted overhead camera
- Image 6: predicted wrist camera

If the arm HOLDS STILL:
- Image 7: predicted overhead camera
- Image 8: predicted wrist camera

Task: {task}

Which action should the arm take? Answer with just: RIGHT, LEFT, or HOLD.
```

Parse RIGHT/LEFT/HOLD from response. Map to move+/move-/hold scores.

### 3. Update callers and report

**File:** `canvas-robot-control/run_control_with_report.py`

- Pass both cameras to scorer via `set_current_observation(base, wrist)`
- Pass paired predicted views to `score_frames()`
- **HTML report shows all images per step:** current base+wrist, predicted base+wrist for each action, VLM raw response, chosen action
- Task: `"move the arm so the claw is pointed at the blue kong"` (or similar)

**File:** `canvas-robot-control/run_control.py`

- Same changes to pass both cameras through

---

## Files to Modify

| File | Change |
|------|--------|
| `canvas-robot-control/control/world_model.py` | `predict_batch()` returns (base, wrist) pairs |
| `canvas-robot-control/control/canvas_utils.py` | Add helper to extract both base and wrist views from predicted frame |
| `canvas-robot-control/scorers/gemma4_comparative.py` | Accept paired views, send 8 images with labeled prompt, parse RIGHT/LEFT/HOLD, no randomization |
| `canvas-robot-control/run_control_with_report.py` | Pass both cameras, update HTML report to show all views |
| `canvas-robot-control/run_control.py` | Pass both cameras to scorer |

---

## Verification

1. Run on hardware with kong toy placed to one side
2. Command: `python run_control_with_report.py --task "move the arm so the claw is pointed at the blue kong" --scorer gemma4 --max-steps 20 --success-threshold 95 --save-frames`
3. Review HTML report: all 8 images visible per step, VLM responses logged, action decisions clear
4. Verify arm moves toward the kong consistently
5. Move kong to opposite side, run again, verify arm reverses direction
