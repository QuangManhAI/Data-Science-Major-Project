# Bug Reports & Post-Mortem Records

| Field | Value |
| :--- | :--- |
| **Document Type** | Documentation Subsystem Index |
| **Status** | Active Index |
| **Owner** | AI Agent & Human Practitioner |
| **Scope** | Defects & Root Cause Analyses |
| **Created** | 2026-09-06T21:05:00+07:00 |
| **Last Updated** | 2026-11-02T12:26:00+07:00 |
| **Reference** | [docs/README.md](../README.md), [agents/templates/BUG_TEMPLATE.md](../../agents/templates/BUG_TEMPLATE.md) |

---

## Active Bug Reports

| Bug ID | Title | Status | Severity | Category | Target Component |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`BUG-01`](BUG_01_PATIENT_ENGAGEMENT_AND_5M_FALSE_ARRIVAL.md) | Seated Patient 5m False Arrival Trigger & Multi-NPC Dialogue Crosstalk | Resolved | High | Simulation / Dialogue | `hospital_npc.gd`, `main.gd` |
| [`BUG-02`](BUG_02_YOLO_CLASS_METADATA_DROPPING.md) | YOLO Multi-Class Bounding Box Label Dropping in Vision Client | Resolved | Medium | Vision Pipeline | `vision_client.gd`, `detector.py` |

---

## Purpose & Conventions

This directory tracks significant technical bugs, regression incidents, numerical instability issues, and environment gotchas.

### Rules for Bug Reports

1. Copy [agents/templates/BUG_TEMPLATE.md](../../agents/templates/BUG_TEMPLATE.md) into this directory:
   - e.g., `BUG_01_CUDA_OOM_DATALOADER.md`, `BUG_02_LABEL_LEAKAGE_VAL.md`.
2. Follow the 5-Whys methodology to identify the true root cause, not just the surface fix.
3. Record the exact regression test added to `tests/` that permanently prevents recurrence.
