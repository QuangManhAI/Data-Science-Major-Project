"""Interactive Real-Time Non-Headless Visual Renderer for HospitalPureVisionEnv.

Renders:
1. Left Panel (740x740): Top-Down 2D Hospital Floorplan with rooms, walls, doors,
   robot, translucent 75-deg camera FOV cone, 5 visual sectors, patient (with
   peripheral companion waypoint and tether), and patrolling doctor.
2. Right Top (640x360): Simulated Robot Front Camera View with YOLO bounding
   boxes (patient, doctor, doors) and 5-sector clearance HUD.
3. Right Bottom (640x380): Live RL Training Telemetry & PPO Metrics Panel.

Supports interactive controls:
- [SPACE]: Pause / Resume
- [+ / -]: Adjust playback speed
- [F]: Switch Floor view (Floor 1 / 2 / 3)
- [R]: Request early episode reset
- [Q / ESC]: Clean exit
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.models.ppo_agent import OMEGA_MAX, V_MAX
from src.training.camera_vision_model import CAMERA_HFOV_RAD
from src.training.hospital_env import (
    DESTINATION_POSITIONS,
    ROBOT_RADIUS_M,
    TaskPhase,
    _DOORS,
    _STRUCTURE,
    floor_of,
)

# Color Palette (BGR for OpenCV)
COLOR_BG_FLOOR = (30, 34, 43)          # Dark Slate
COLOR_WALL = (128, 114, 107)           # Structural Grey
COLOR_WALL_OUTLINE = (180, 165, 155)   # Light Grey
COLOR_DOOR = (80, 200, 120)            # Emerald Green
COLOR_ROBOT = (255, 229, 0)            # Electric Cyan / Yellow
COLOR_PATIENT = (0, 179, 255)          # Warm Amber
COLOR_DOCTOR = (255, 121, 41)          # Medical Blue
COLOR_GOAL_BEACON = (0, 230, 118)      # Neon Green
COLOR_TETHER_GOOD = (0, 230, 118)      # Green
COLOR_TETHER_STRETCH = (0, 165, 255)   # Orange
COLOR_TETHER_BROKEN = (60, 60, 255)    # Red
COLOR_TEXT_WHITE = (240, 240, 245)
COLOR_TEXT_DIM = (160, 160, 170)
COLOR_PANEL_BG = (22, 24, 29)
COLOR_CARD_BG = (32, 36, 45)


class VisualEnvRenderer:
    """Non-headless high-performance OpenCV GUI renderer for HospitalPureVisionEnv."""

    def __init__(
        self,
        window_name: str = "Hospital Helper Robot — Pure-Vision Visual Training Environment [MOCK TRAIN]",
        target_fps: float = 30.0,
        floor_override: Optional[int] = None,
    ) -> None:
        self.window_name = window_name
        self.target_fps = target_fps
        self.delay_ms = max(1, int(1000.0 / target_fps)) if target_fps > 0 else 1
        self.paused = False
        self.floor_override = floor_override
        self.trail: List[Tuple[float, float]] = []
        self.max_trail = 45

        # Canvas geometry
        self.width_map = 740
        self.height_map = 740
        self.width_side = 640
        self.height_side = 740
        self.total_width = self.width_map + self.width_side
        self.total_height = 740

        # Scale: 24m width -> ~680 px => S = 27.0 pixels / meter
        self.scale = 27.5
        self.origin_u = self.width_map // 2
        self.origin_v = self.height_map // 2

        # PPO Telemetry Cache
        self.last_ppo_stats: Dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "bc_loss": 0.0,
            "lr": 1e-4,
            "prior_weight": 0.0,
        }

        # Initialize window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(self.window_name, self.total_width, self.total_height)

    def world_to_screen(self, x: float, z: float, floor: int = 1) -> Tuple[int, int]:
        """Convert hospital world (x, z) to 2D pixel coordinates on active floor."""
        x_base = 100.0 if floor == 3 else (50.0 if floor == 2 else 0.0)
        x_rel = x - x_base
        u = int(self.origin_u + x_rel * self.scale)
        v = int(self.origin_v + z * self.scale)
        return u, v

    def update_ppo_metrics(self, stats: Dict[str, float]) -> None:
        """Update PPO training telemetry cache."""
        self.last_ppo_stats.update(stats)

    def update_ppo_stats(self, stats: Dict[str, float]) -> None:
        """Alias for update_ppo_metrics."""
        self.update_ppo_metrics(stats)

    def render_step(
        self,
        env: Any,
        action: np.ndarray,
        reward: float,
        step_idx: int,
        episode_idx: int,
        total_episodes: int,
        info: Dict[str, Any],
        is_mock_train: bool = True,
    ) -> bool:
        """Render a single frame of the training environment and handle user hotkeys.

        Returns False if the user pressed Q/ESC to quit, otherwise True.
        """
        canvas = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)

        # Determine active floor
        robot_floor = floor_of(env.robot_x)
        display_floor = self.floor_override if self.floor_override is not None else robot_floor

        # 1. Render Left Panel (World Floorplan)
        map_canvas = canvas[:, :self.width_map]
        map_canvas[:] = COLOR_BG_FLOOR
        self._draw_floorplan(map_canvas, env, display_floor)
        self._draw_entities(map_canvas, env, display_floor)

        # 2. Render Right Top Panel (Simulated Front Camera View)
        cam_canvas = canvas[:360, self.width_map:]
        self._draw_camera_view(cam_canvas, env)

        # 3. Render Right Bottom Panel (Live Telemetry & PPO HUD)
        hud_canvas = canvas[360:, self.width_map:]
        self._draw_telemetry_hud(
            hud_canvas,
            env,
            action,
            reward,
            step_idx,
            episode_idx,
            total_episodes,
            info,
            is_mock_train,
        )

        # Separator borders
        cv2.line(canvas, (self.width_map, 0), (self.width_map, self.total_height), (60, 65, 75), 2)
        cv2.line(canvas, (self.width_map, 360), (self.total_width, 360), (60, 65, 75), 2)

        # Show frame
        cv2.imshow(self.window_name, canvas)

        # Interactive key handling
        key = cv2.waitKey(self.delay_ms if not self.paused else 50) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            return False
        if key == 32:  # SPACE
            self.paused = not self.paused
        elif key in (ord("+"), ord("=")):
            self.delay_ms = max(1, self.delay_ms // 2)
        elif key in (ord("-"), ord("_")):
            self.delay_ms = min(200, self.delay_ms * 2)
        elif key in (ord("f"), ord("F")):
            # Cycle floor 1 -> 2 -> 3 -> Auto
            if self.floor_override is None:
                self.floor_override = 1
            elif self.floor_override == 1:
                self.floor_override = 2
            elif self.floor_override == 2:
                self.floor_override = 3
            else:
                self.floor_override = None

        while self.paused:
            p_key = cv2.waitKey(50) & 0xFF
            if p_key == 32:
                self.paused = False
                break
            if p_key in (ord("q"), ord("Q"), 27):
                return False

        return True

    def _draw_floorplan(self, img: np.ndarray, env: Any, floor: int) -> None:
        """Draw hospital rooms, walls, doorways, and beacons for active floor."""
        # 1. Subtle room tints and labels
        rooms_f1 = [
            ("Phong 101: Cap cuu", -11.5, -8.5, -4.5, -2.2, (45, 26, 31)),
            ("Phong 110: Xet nghiem", -3.5, -8.5, 3.5, -2.2, (51, 42, 22)),
            ("Phong 112: Khoa noi", 4.5, -8.5, 11.5, -2.2, (51, 27, 40)),
            ("Nha thuoc benh vien", -11.5, 2.2, -4.5, 8.5, (34, 45, 22)),
            ("Sanh cho & Tiep don", -3.5, 2.2, 5.5, 8.5, (26, 43, 51)),
            ("Thang may & WC", 7.5, -2.0, 11.5, 2.0, (56, 38, 26)),
        ]
        for name, x1, z1, x2, z2, bgr in rooms_f1:
            u1, v1 = self.world_to_screen(x1, z1, floor)
            u2, v2 = self.world_to_screen(x2, z2, floor)
            cv2.rectangle(img, (u1, v1), (u2, v2), bgr, -1)
            cv2.rectangle(img, (u1, v1), (u2, v2), (55, 60, 72), 1)
            cv2.putText(
                img,
                name,
                (u1 + 8, v1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                COLOR_TEXT_DIM,
                1,
                cv2.LINE_AA,
            )

        # 2. Central corridor centerline & bounds
        c_u1, c_v1 = self.world_to_screen(-12.0, -2.0, floor)
        c_u2, c_v2 = self.world_to_screen(12.0, 2.0, floor)
        cv2.rectangle(img, (c_u1, c_v1), (c_u2, c_v2), (38, 42, 52), -1)

        # 3. Structural Walls from _STRUCTURE
        if floor in _STRUCTURE:
            walls = _STRUCTURE[floor]
            for w in walls:
                wu1, wv1 = self.world_to_screen(float(w[0]), float(w[1]), floor)
                wu2, wv2 = self.world_to_screen(float(w[2]), float(w[3]), floor)
                cv2.rectangle(img, (wu1, wv1), (wu2, wv2), COLOR_WALL, -1)
                cv2.rectangle(img, (wu1, wv1), (wu2, wv2), COLOR_WALL_OUTLINE, 1)

        # 4. Doors from _DOORS
        for (f, side), (centres, _) in _DOORS.items():
            if f == floor:
                z_line = -2.0 if side == "north" else 2.0
                for c in centres:
                    du, dv = self.world_to_screen(c, z_line, floor)
                    cv2.circle(img, (du, dv), 5, COLOR_DOOR, -1)
                    cv2.circle(img, (du, dv), 7, (20, 80, 40), 1)

        # 5. Goal Beacon
        goal_x, goal_z = env.goal_x, env.goal_z
        if floor_of(goal_x) == floor:
            gu, gv = self.world_to_screen(goal_x, goal_z, floor)
            cv2.circle(img, (gu, gv), 9, COLOR_GOAL_BEACON, 2)
            cv2.circle(img, (gu, gv), 4, COLOR_GOAL_BEACON, -1)
            cv2.putText(
                img,
                f"GOAL: {getattr(env, 'goal_key', 'target')}",
                (gu + 12, gv + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                COLOR_GOAL_BEACON,
                1,
                cv2.LINE_AA,
            )

        # Floor indicator watermark
        floor_label = f"FLOOR {floor} (Corridor / Multi-Room)"
        cv2.putText(
            img,
            floor_label,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_TEXT_WHITE,
            2,
            cv2.LINE_AA,
        )

    def _draw_entities(self, img: np.ndarray, env: Any, floor: int) -> None:
        """Draw Robot, FOV cone, 5 Sectors, Patient, Tether, and Doctor."""
        rx, rz, ryaw = env.robot_x, env.robot_z, env.robot_yaw
        r_fl = floor_of(rx)

        if r_fl == floor:
            ru, rv = self.world_to_screen(rx, rz, floor)

            # 1. Breadcrumb trail
            self.trail.append((rx, rz))
            if len(self.trail) > self.max_trail:
                self.trail.pop(0)
            for i, (tx, tz) in enumerate(self.trail):
                tu, tv = self.world_to_screen(tx, tz, floor)
                alpha = (i + 1) / float(len(self.trail))
                cv2.circle(img, (tu, tv), 2, (int(255 * alpha), int(229 * alpha), 0), -1)

            # 2. Translucent Camera FOV Cone (75 deg, 7.0m)
            fov_rad = CAMERA_HFOV_RAD
            fov_range_px = int(7.0 * self.scale)
            cone_overlay = img.copy()

            half_fov = fov_rad / 2.0
            angle_left = ryaw - half_fov
            angle_right = ryaw + half_fov

            # Forward is (-sin(yaw), -cos(yaw))
            pt_left = (
                int(ru - math.sin(angle_left) * fov_range_px),
                int(rv - math.cos(angle_left) * fov_range_px),
            )
            pt_right = (
                int(ru - math.sin(angle_right) * fov_range_px),
                int(rv - math.cos(angle_right) * fov_range_px),
            )
            cone_poly = np.array([[ru, rv], pt_left, pt_right], dtype=np.int32)
            cv2.fillPoly(cone_overlay, [cone_poly], (50, 65, 80))
            cv2.addWeighted(cone_overlay, 0.40, img, 0.60, 0, img)
            cv2.polylines(img, [cone_poly], True, (120, 140, 170), 1, cv2.LINE_AA)

            # 3. 5 Visual Sector Radial Rays (FL, L, C, R, FR)
            sectors = getattr(env, "last_sectors", np.zeros(5, dtype=np.float32))
            sector_angles = [-30.0, -15.0, 0.0, 15.0, 30.0]
            sec_len_px = int(4.5 * self.scale)

            for i, deg in enumerate(sector_angles):
                s_ang = ryaw + math.radians(deg)
                su = int(ru - math.sin(s_ang) * sec_len_px)
                sv = int(rv - math.cos(s_ang) * sec_len_px)

                # Sector color based on clearance h
                val = float(sectors[i]) if i < len(sectors) else 0.0
                if val < 0.25:
                    col = (0, 220, 100)  # Green
                elif val < 0.55:
                    col = (0, 200, 255)  # Orange
                else:
                    col = (60, 60, 255)  # Red

                cv2.line(img, (ru, rv), (su, sv), col, 2, cv2.LINE_AA)
                cv2.circle(img, (su, sv), 3, col, -1)

            # 4. Robot Chassis & Heading Indicator
            r_rad_px = max(6, int(ROBOT_RADIUS_M * self.scale))
            cv2.circle(img, (ru, rv), r_rad_px, COLOR_ROBOT, -1)
            cv2.circle(img, (ru, rv), r_rad_px + 2, (20, 20, 25), 2)

            fwd_len_px = r_rad_px + 12
            fwd_u = int(ru - math.sin(ryaw) * fwd_len_px)
            fwd_v = int(rv - math.cos(ryaw) * fwd_len_px)
            cv2.arrowedLine(img, (ru, rv), (fwd_u, fwd_v), (0, 0, 0), 2, tipLength=0.35)

            cv2.putText(
                img,
                "ROBOT",
                (ru - 18, rv - r_rad_px - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                COLOR_ROBOT,
                1,
                cv2.LINE_AA,
            )

        # 5. Patients & Escort Formation
        patients = getattr(env, "patients", [])
        if patients:
            for i, p in enumerate(patients):
                px, pz = p["x"], p["z"]
                if floor_of(px) == floor:
                    pu, pv = self.world_to_screen(px, pz, floor)
                    p_rad_px = max(5, int(0.25 * self.scale))
                    is_active = (i == getattr(env, "active_patient_idx", None))
                    is_served = p.get("served", False)
                    if is_served:
                        col = (100, 150, 100)  # Muted green for completed
                        label = f"{p['id']} [SERVED]"
                    elif is_active:
                        col = COLOR_PATIENT    # Bright amber for active escort
                        label = f"{p['id']} [LOCKED]"
                    else:
                        col = (0, 140, 220)    # Soft orange for waiting
                        label = f"{p['id']}"

                    cv2.circle(img, (pu, pv), p_rad_px, col, -1)
                    cv2.circle(img, (pu, pv), p_rad_px + 2, (15, 20, 30), 2)
                    cv2.putText(
                        img,
                        label,
                        (pu - 24, pv - p_rad_px - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        col,
                        1,
                        cv2.LINE_AA,
                    )
        else:
            # Fallback for single-patient environments
            px, pz = env.patient_x, env.patient_z
            if floor_of(px) == floor:
                pu, pv = self.world_to_screen(px, pz, floor)
                p_rad_px = max(5, int(0.25 * self.scale))
                cv2.circle(img, (pu, pv), p_rad_px, COLOR_PATIENT, -1)
                cv2.circle(img, (pu, pv), p_rad_px + 2, (15, 20, 30), 2)
                cv2.putText(
                    img,
                    "PATIENT",
                    (pu - 22, pv - p_rad_px - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    COLOR_PATIENT,
                    1,
                    cv2.LINE_AA,
                )

        # If in GUIDE_ESCORT: draw companion target waypoint and visual tether
        if env.task_phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF) and r_fl == floor:
            ru, rv = self.world_to_screen(rx, rz, floor)
            px, pz = env.patient_x, env.patient_z
            if floor_of(px) == floor:
                pu, pv = self.world_to_screen(px, pz, floor)
                # Visual Tether
                d_p = math.hypot(px - rx, pz - rz)
                tether_col = (
                    COLOR_TETHER_GOOD
                    if d_p <= 2.0
                    else (COLOR_TETHER_STRETCH if d_p <= 3.5 else COLOR_TETHER_BROKEN)
                )
                cv2.line(img, (ru, rv), (pu, pv), tether_col, 2, cv2.LINE_AA)

                # Companion Waypoint Target (+1.0m fwd, +0.55m right)
                fwd_x, fwd_z = -math.sin(ryaw), -math.cos(ryaw)
                right_x, right_z = math.cos(ryaw), -math.sin(ryaw)
                c_tgt_x = rx + fwd_x * 1.0 + right_x * 0.55
                c_tgt_z = rz + fwd_z * 1.0 + right_z * 0.55
                ctu, ctv = self.world_to_screen(c_tgt_x, c_tgt_z, floor)
                cv2.circle(img, (ctu, ctv), 6, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    img,
                    "target",
                    (ctu + 6, ctv + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 6. Patrolling Doctor
        if getattr(env, "doctor_active", False):
            dx, dz = env.doctor_x, env.doctor_z
            if floor_of(dx) == floor:
                du, dv = self.world_to_screen(dx, dz, floor)
                cv2.circle(img, (du, dv), 6, COLOR_DOCTOR, -1)
                cv2.circle(img, (du, dv), 12, (200, 100, 30), 1)
                cv2.putText(
                    img,
                    "DOCTOR",
                    (du - 20, dv - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    COLOR_DOCTOR,
                    1,
                    cv2.LINE_AA,
                )

    def _draw_camera_view(self, img: np.ndarray, env: Any) -> None:
        """Render simulated first-person RGB camera frame with YOLO boxes and sector HUD."""
        w, h = 640, 360
        # Simulated interior corridor background (Ceiling dark, floor grey, horizon gradient)
        cv2.rectangle(img, (0, 0), (w, 180), (35, 38, 45), -1)    # Ceiling / Wall
        cv2.rectangle(img, (0, 180), (w, h), (48, 52, 60), -1)    # Walkable Floor
        cv2.line(img, (0, 180), (w, 180), (65, 70, 80), 1)       # Horizon line

        # 1. Check patient pinhole detection
        obs = env._get_obs()
        pat_seen = obs[5] > 0.5
        pat_cx = float(obs[6])
        pat_h = float(obs[7])
        pat_conf = float(obs[8])

        if pat_seen and pat_h > 0.04:
            # Map normalized cx in [-0.5, 0.5] and height to pixel coordinates
            box_w = max(24, int(pat_h * 0.42 * w))
            box_h = max(32, int(pat_h * h))
            center_x = int((pat_cx + 0.5) * w)
            center_y = int(180 + (pat_h * 0.15) * h)

            bx1 = max(0, center_x - box_w // 2)
            by1 = max(10, center_y - box_h // 2)
            bx2 = min(w - 1, center_x + box_w // 2)
            by2 = min(h - 10, center_y + box_h // 2)

            cv2.rectangle(img, (bx1, by1), (bx2, by2), COLOR_PATIENT, 2)
            d_est = 1.20 / max(pat_h, 0.05)
            label = f"patient {pat_conf:.2f} | h={pat_h:.2f} (d~{d_est:.1f}m)"
            cv2.rectangle(img, (bx1, by1 - 18), (bx1 + len(label) * 8, by1), COLOR_PATIENT, -1)
            cv2.putText(
                img,
                label,
                (bx1 + 4, by1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 2. Check doctor pinhole detection
        doc_seen = obs[9] > 0.5
        doc_cx = float(obs[10])
        doc_h = float(obs[11])
        doc_conf = float(obs[12])

        if doc_seen and doc_h > 0.04:
            box_w = max(24, int(doc_h * 0.40 * w))
            box_h = max(32, int(doc_h * h))
            center_x = int((doc_cx + 0.5) * w)
            center_y = int(180 + (doc_h * 0.15) * h)

            bx1 = max(0, center_x - box_w // 2)
            by1 = max(10, center_y - box_h // 2)
            bx2 = min(w - 1, center_x + box_w // 2)
            by2 = min(h - 10, center_y + box_h // 2)

            cv2.rectangle(img, (bx1, by1), (bx2, by2), COLOR_DOCTOR, 2)
            label = f"doctor {doc_conf:.2f} | h={doc_h:.2f}"
            cv2.rectangle(img, (bx1, by1 - 18), (bx1 + len(label) * 8, by1), COLOR_DOCTOR, -1)
            cv2.putText(
                img,
                label,
                (bx1 + 4, by1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 3. Check doorway detection (obs[25..28])
        door_dist_norm = float(obs[25])
        if door_dist_norm > 0.01:
            door_d = door_dist_norm * 10.0
            angle_door = math.atan2(float(obs[26]), float(obs[27]))
            if abs(angle_door) < (CAMERA_HFOV_RAD / 2.0):
                norm_cx = -math.tan(angle_door) / math.tan(CAMERA_HFOV_RAD / 2.0)
                center_x = int((norm_cx * 0.5 + 0.5) * w)
                door_w_px = max(30, int((2.0 / max(door_d, 0.5)) * 40))
                door_h_px = max(50, int((2.4 / max(door_d, 0.5)) * 60))
                dbx1 = max(0, center_x - door_w_px // 2)
                dby1 = max(10, 180 - door_h_px // 2)
                dbx2 = min(w - 1, center_x + door_w_px // 2)
                dby2 = min(h - 10, 180 + door_h_px // 2)
                cv2.rectangle(img, (dbx1, dby1), (dbx2, dby2), COLOR_DOOR, 2)
                label = f"doorway (d~{door_d:.1f}m)"
                cv2.putText(
                    img,
                    label,
                    (dbx1 + 4, dby1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    COLOR_DOOR,
                    1,
                    cv2.LINE_AA,
                )

        # 4. Top Visual Clearance HUD (5 Sectors: FL, L, C, R, FR)
        sectors = getattr(env, "last_sectors", np.zeros(5, dtype=np.float32))
        sec_names = ["FL (-30)", "L (-15)", "C (0)", "R (+15)", "FR (+30)"]
        bar_w = 90
        start_x = (w - (bar_w * 5 + 40)) // 2

        for i in range(5):
            bx = start_x + i * (bar_w + 10)
            val = float(sectors[i]) if i < len(sectors) else 0.0
            col = (0, 220, 100) if val < 0.25 else ((0, 200, 255) if val < 0.55 else (60, 60, 255))
            cv2.rectangle(img, (bx, 10), (bx + bar_w, 28), (25, 28, 35), -1)
            fill_w = int(val * bar_w)
            if fill_w > 0:
                cv2.rectangle(img, (bx, 10), (bx + fill_w, 28), col, -1)
            cv2.rectangle(img, (bx, 10), (bx + bar_w, 28), (70, 75, 85), 1)
            cv2.putText(
                img,
                f"{sec_names[i]}: {val:.2f}",
                (bx + 4, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                COLOR_TEXT_WHITE,
                1,
                cv2.LINE_AA,
            )

        # Camera Header Badge
        cv2.putText(
            img,
            "CAMERA VIEW (Pinhole RGB + YOLO Perception)",
            (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )

    def _draw_telemetry_hud(
        self,
        img: np.ndarray,
        env: Any,
        action: np.ndarray,
        reward: float,
        step_idx: int,
        episode_idx: int,
        total_episodes: int,
        info: Dict[str, Any],
        is_mock_train: bool,
    ) -> None:
        """Render live RL training statistics, telemetry bars, and interactive hotkeys."""
        w, h = 640, 380
        img[:] = COLOR_PANEL_BG

        # Banner: Mock Training vs Production
        banner_bg = (15, 60, 110) if is_mock_train else (20, 70, 30)
        cv2.rectangle(img, (12, 10), (w - 12, 38), banner_bg, -1)
        cv2.rectangle(img, (12, 10), (w - 12, 38), (0, 180, 255) if is_mock_train else (0, 220, 100), 1)
        banner_text = (
            "*** MOCK TRAINING MODE (NO CHECKPOINTS SAVED) ***"
            if is_mock_train
            else "--- PRODUCTION TRAINING (CHECKPOINTS SAVED) ---"
        )
        cv2.putText(
            img,
            banner_text,
            (w // 2 - 190, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 240, 255) if is_mock_train else (200, 255, 200),
            1,
            cv2.LINE_AA,
        )

        # 1. Episode & Phase Info
        y = 62
        cv2.putText(
            img,
            f"Episode: {episode_idx} / {total_episodes}",
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Step: {step_idx} / 1500 ({step_idx * 0.2:.1f}s)",
            (210, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )

        phase_names = {
            TaskPhase.SEARCH_PATIENT: ("SEARCH_PATIENT", (255, 120, 40)),
            TaskPhase.APPROACH_PATIENT: ("APPROACH_PATIENT", (0, 180, 255)),
            TaskPhase.GUIDE_ESCORT: ("GUIDE_ESCORT", (0, 220, 100)),
            TaskPhase.CALL_STAFF: ("CALL_STAFF", (60, 60, 255)),
        }
        if getattr(env, "returning_to_lobby", False):
            p_name, p_col = ("RETURN_ELEVATOR", (180, 80, 220))
        else:
            p_name, p_col = phase_names.get(env.task_phase, ("UNKNOWN", (150, 150, 150)))
        cv2.rectangle(img, (450, y - 16), (w - 16, y + 4), p_col, -1)
        cv2.putText(
            img,
            p_name,
            (456, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # 2. Kinematics & Velocity Gauges
        y += 32
        v_act, omega_act = float(action[0]), float(action[1])
        v_real = (v_act + 1.0) * 0.5 * V_MAX
        omega_real = omega_act * OMEGA_MAX

        # V speed gauge
        cv2.putText(
            img,
            f"Linear Speed v: {v_real:4.2f} m/s",
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(img, (220, y - 12), (380, y), (35, 38, 48), -1)
        v_fill = int((v_real / V_MAX) * 160)
        cv2.rectangle(img, (220, y - 12), (220 + v_fill, y), COLOR_ROBOT, -1)
        cv2.rectangle(img, (220, y - 12), (380, y), (80, 85, 95), 1)

        # Omega gauge
        cv2.putText(
            img,
            f"Yaw Rate w: {omega_real:+4.2f} rad/s",
            (400, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )

        # 3. Reward & Mission Performance Card
        y += 34
        cv2.rectangle(img, (12, y), (w - 12, y + 80), COLOR_CARD_BG, -1)
        cv2.rectangle(img, (12, y), (w - 12, y + 80), (55, 60, 72), 1)

        r_team = float(info.get("cumulative_team_reward", 0.0))
        helped = int(info.get("patients_helped", 0))
        cols = int(info.get("collisions", 0))
        tether_d = float(np.clip(info.get("tether_dist", 0.0), 0.0, 9.9))

        cv2.putText(
            img,
            f"Step Reward: {reward:+5.2f}",
            (24, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Team Reward: {r_team:+6.1f}",
            (24, y + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 230, 118) if r_team >= 0 else (80, 80, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            f"Patients Escorted: {helped} / 6",
            (230, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            COLOR_PATIENT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Collisions: {cols}",
            (230, y + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (100, 100, 255) if cols > 0 else COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            f"Escort Tether: {tether_d:3.1f}m",
            (440, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 230, 118) if tether_d <= 2.0 else (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Spotting: {'YES' if info.get('has_det', 0.0) > 0.5 else 'NO'}",
            (440, y + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )

        # 4. PPO Loss & Optimizer Telemetry
        y += 98
        cv2.putText(
            img,
            "PPO Training & Network Updates:",
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )
        y += 20
        stats = self.last_ppo_stats
        cv2.putText(
            img,
            f"Policy Loss: {stats['policy_loss']:+6.4f}  |  Value Loss: {stats['value_loss']:6.4f}  |  Entropy: {stats['entropy']:5.3f}",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )
        y += 20
        cv2.putText(
            img,
            f"BC Loss: {stats['bc_loss']:6.4f}  |  BC Weight: {stats['prior_weight']:4.2f}  |  LR: {stats['lr']:.1e}",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )

        # 5. Hotkeys Footer
        y += 32
        cv2.line(img, (12, y), (w - 12, y), (50, 55, 65), 1)
        y += 18
        controls_text = "[SPACE] Pause/Play  |  [+] Speed Up  |  [-] Slow Down  |  [F] Floor  |  [Q] Quit"
        cv2.putText(
            img,
            controls_text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    def close(self) -> None:
        """Close window and free GUI resources."""
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
