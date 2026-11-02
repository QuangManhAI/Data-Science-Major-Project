# YOLO Multi-Class Bounding Box Label Dropping in Vision Client — Bug Report

- **Motivation/Background**: The vision system was upgraded from 1 class (`person`) to a fine-tuned 17-class model including `patient`, `doctor`, `wheelchair`, `hospital_bed`, `stretcher_cart`, etc. However, during runtime inference, all bounding boxes in the Godot viewport and debug inspection panel still rendered as `"person"`.
- **Purpose**: Document symptoms, root causes, remediations, and test verification for the class metadata dropping issue in the Godot vision client and Python detection pipeline.
- **Overview Pipeline**: Identified in visual HUD inspection panel; investigated JSON payload between `yolo_server.py`, `vision_client.gd`, and `view_overlay.gd`; resolved class field preservation.
- **Detailed Plan**: §1 Symptom & Error Traceback; §2 Root Cause Analysis; §3 Solution & Remediation; §4 Verification Evidence; §5 Related Links.
- **References**: [`agents/templates/BUG_TEMPLATE.md`](../../agents/templates/BUG_TEMPLATE.md), [`sim_vision/godot/scripts/vision_client.gd`](../../sim_vision/godot/scripts/vision_client.gd), [`sim_vision/python/detector.py`](../../sim_vision/python/detector.py).
- **Created**: 2026-11-02T12:26:00+07:00
- **Last Updated**: 2026-11-02T12:26:00+07:00

---

## Metadata

- **Bug ID**: `BUG-02`
- **Title**: YOLO Multi-Class Bounding Box Label Dropping in Vision Client
- **Status**: Resolved
- **Severity**: Medium
- **Category**: Vision / Perception Pipeline
- **Target Component**: [`sim_vision/godot/scripts/vision_client.gd`](../../sim_vision/godot/scripts/vision_client.gd), [`sim_vision/godot/scripts/view_overlay.gd`](../../sim_vision/godot/scripts/view_overlay.gd), [`sim_vision/python/detector.py`](../../sim_vision/python/detector.py)

---

## 1. Symptom & Error Traceback

In the runtime simulation:
1. Objects such as hospital beds, wheelchairs, doctors, and patients were successfully detected by YOLO (`yolo_server.py`).
2. However, the HUD overlays rendered every single detected object with the label `"person"`, discarding medical asset classes.
3. Policy server was unable to differentiate between patient (target) and doctor (obstacle) from Godot's visual detections.

Trace from inspection HUD:
```text
[ViewOverlay] Rendering box (0.32, 0.45, 0.65, 0.88): label="person", conf=0.91
[VisionClient] Dropped raw fields: class_id, class_name
```

---

## 2. Root Cause Analysis (5-Whys)

1. **Why did all bounding box overlays show "person"?**
   Because `view_overlay.gd` was displaying `det.get("label", "person")`, and `det` did not contain a valid `class_name` or `label`.
2. **Why was `class_name` missing from the detection dictionary?**
   In `sim_vision/godot/scripts/vision_client.gd:_parse_response()`, the JSON response parsing loop extracted only `x1`, `y1`, `x2`, `y2`, and `conf`, omitting `class_id` and `class_name`.
3. **Why did the parser omit them?**
   `vision_client.gd` was originally authored for a single-class YOLO model where every detection was unconditionally assumed to be `person`.
4. **Why did Python unit tests fail when fixing `detector.py`?**
   In `sim_vision/python/detector.py:normalize_boxes()`, the helper function signature was modified to accept multi-class lists, but legacy test suites called it with 4 positional arguments `(boxes, w, h, orig_shape)`, raising `TypeError`.

---

## 3. Solution & Remediation

### 3.1 Preserve Class Metadata in `vision_client.gd`
Updated JSON parser in [`sim_vision/godot/scripts/vision_client.gd`](../../sim_vision/godot/scripts/vision_client.gd):
```gdscript
var class_id: int = int(d.get("class_id", 0))
var class_name: String = str(d.get("class_name", "patient" if class_id == 0 else "doctor"))
parsed.append({
    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
    "conf": conf,
    "class_id": class_id,
    "class_name": class_name,
    "label": class_name
})
```

### 3.2 Dynamic Class Styling in `view_overlay.gd`
Updated [`sim_vision/godot/scripts/view_overlay.gd`](../../sim_vision/godot/scripts/view_overlay.gd) with class-specific colors:
- Patient: Amber (`#FFB300`)
- Doctor / Staff: Electric Blue (`#2979FF`)
- Medical Equipment (bed, wheelchair, stretcher): Cyan (`#00E5FF`)
- Obstacle: Orange/Red (`#FF5252`)

### 3.3 Backward Compatibility in `detector.py`
Updated `normalize_boxes()` in [`sim_vision/python/detector.py`](../../sim_vision/python/detector.py) to support both:
- Dictionary-based multi-class format: `[{"x1": ..., "y1": ..., "class_name": ...}]`
- Positional array format from legacy unit test callers.

---

## 4. Verification Evidence

### 4.1 Python Unit Tests
```powershell
.\.venv\Scripts\python.exe -m pytest sim_vision/python/tests/test_detector.py -v
```

**Output:**
```text
sim_vision/python/tests/test_detector.py::test_normalize_boxes_xyxy PASSED
sim_vision/python/tests/test_detector.py::test_normalize_boxes_multiclass PASSED
sim_vision/python/tests/test_detector.py::test_detector_inference PASSED
============================== 3 passed in 0.82s ==============================
```

### 4.2 Godot GUT UI Tests
```powershell
& "C:\document\Study documents\Robo-helper-in-Hospital\sim_vision\godot\bin\Godot_v4.3-stable_win64.exe" --headless --path sim_vision/godot -s addons/gut/gut_cmdln.gd -gtest=res://tests/test_inspection_panel.gd
```

**Output:**
```text
run_tests: Running script res://tests/test_inspection_panel.gd
  * test_pure_vision_defaults: Passed
  * test_multiclass_bounding_box_colors: Passed
  * test_five_sector_hud_display: Passed
  * test_lidar_hidden_in_pure_vision: Passed
  [4/4] tests passed.
```

---

## 5. Related Links

- Bug Directory Index: [`docs/bugs/README.md`](README.md)
- Decision Log D-058: [`docs/AI-Human-logs/Vuong.md`](../AI-Human-logs/Vuong.md)
- Core Scripts:
  - [`sim_vision/godot/scripts/vision_client.gd`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/scripts/vision_client.gd)
  - [`sim_vision/godot/scripts/view_overlay.gd`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/scripts/view_overlay.gd)
  - [`sim_vision/python/detector.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/python/detector.py)
