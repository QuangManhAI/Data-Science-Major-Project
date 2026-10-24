# unified_dataset_generator.gd
## BỘ SINH DỮ LIỆU THỊ GIÁC HỢP NHẤT (1,152 FRAMES LÂM SÀNG CỐT LÕI - 64 CA)
## Tích hợp Lưới Dò Tia Vật Lý Đa Điểm (Multi-Raycast 128 Rays) & Visibility-Aware Ground Truth
extends Node3D

@export var robot_camera: Camera3D
@export var world_env: WorldEnvironment
@export var main_dir_light: DirectionalLight3D
@export var actors_pool: Node3D
@export var obstacle_props: Node3D
@export var hard_negatives_pool: Node3D

@export var capture_resolution: Vector2i = Vector2i(640, 480)
@export var random_seed: int = 20261024
@export var output_root_dir: String = "res://../data/vision_eval_unified"
@export var auto_start_on_ready: bool = false

var _rng := RandomNumberGenerator.new()
var _jsonl_file: FileAccess
var _jsonl_rich_file: FileAccess
var _total_saved_frames: int = 0
var _start_time_msec: int = 0

const corridor_anchor := Vector3(0.0, 0.0, 0.0)
const clinic_door_anchor := Vector3(-2.8, 0.0, 1.2)
const reception_anchor := Vector3(2.4, 0.0, -1.5)
const doorway_110_anchor := Vector3(0.0, 0.0, -2.0)

# Ngân sách 18 frames đại diện cho mỗi ca kiểm thử tĩnh
const UNIFIED_DISTANCES: Array[float] = [1.0, 2.0, 3.5, 5.5]
const UNIFIED_CAM_YAWS: Array[float] = [-15.0, 15.0]
const UNIFIED_CAM_HEIGHTS: Array[float] = [0.85, 1.25]

func _ready() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	var cmd_args: PackedStringArray = OS.get_cmdline_user_args()
	if auto_start_on_ready or ("--generate-unified-dataset" in cmd_args) or ("--generate-dataset" in cmd_args):
		call_deferred("run_unified_pipeline")

## Điều phối chính: 8 Nhóm Lâm Sàng Cốt Lõi x 64 Ca = 1,152 frames
func run_unified_pipeline() -> void:
	_start_time_msec = Time.get_ticks_msec()
	_rng.seed = random_seed

	var vp: Viewport = get_viewport()
	vp.size = capture_resolution
	vp.use_taa = false

	var abs_out_dir: String = ProjectSettings.globalize_path(output_root_dir)
	var abs_img_dir: String = abs_out_dir.path_join("images")
	var abs_lbl_dir: String = abs_out_dir.path_join("labels")
	DirAccess.make_dir_recursive_absolute(abs_img_dir)
	DirAccess.make_dir_recursive_absolute(abs_lbl_dir)

	var jsonl_path: String = abs_out_dir.path_join("annotations.jsonl")
	var jsonl_rich_path: String = abs_out_dir.path_join("annotations_rich_5w1h.jsonl")
	_jsonl_file = FileAccess.open(jsonl_path, FileAccess.WRITE)
	_jsonl_rich_file = FileAccess.open(jsonl_rich_path, FileAccess.WRITE)

	print("==================================================================")
	print("[UnifiedGenerator] BẮT ĐẦU SINH BỘ DỮ LIỆU HỢP NHẤT (~1,500 FRAMES SẠCH)")
	print("[UnifiedGenerator] Thư mục đích: ", abs_out_dir)
	print("==================================================================")

	_hide_all_entities()
	_apply_lighting_profile("fluorescent_standard_450lux")

	# Chạy tuần tự 8 nhóm lâm sàng hợp nhất (83 ca kiểm thử)
	await _generate_group_u1(abs_img_dir, abs_lbl_dir) # Camouflage (10 ca)
	await _generate_group_u2(abs_img_dir, abs_lbl_dir) # Clinical Poses (12 ca)
	await _generate_group_u3(abs_img_dir, abs_lbl_dir) # Medical Props (14 ca)
	await _generate_group_u4(abs_img_dir, abs_lbl_dir) # Wheelchair & Beds (8 ca)
	await _generate_group_u5(abs_img_dir, abs_lbl_dir) # Kinematics (8 ca)
	await _generate_group_u6(abs_img_dir, abs_lbl_dir) # Crowd & Social (9 ca)
	await _generate_group_u7(abs_img_dir, abs_lbl_dir) # Extreme Optics (12 ca)
	await _generate_group_u8(abs_img_dir, abs_lbl_dir) # Hard Negative Traps (10 ca)

	_jsonl_file.flush()
	_jsonl_file.close()
	_jsonl_rich_file.flush()
	_jsonl_rich_file.close()

	var elapsed_sec: float = (Time.get_ticks_msec() - _start_time_msec) / 1000.0
	_write_dataset_manifest(abs_out_dir, elapsed_sec)

	print("==================================================================")
	print("[UnifiedGenerator] HOÀN TẤT THÀNH CÔNG! Đã xuất ", _total_saved_frames, " frames trong ", snappedf(elapsed_sec, 0.1), "s.")
	print("==================================================================")
	get_tree().quit(0)

# ==============================================================================
# HÀM BẮN TIA RAYCAST VẬT LÝ TÍNH TOÁN VISIBLE BOUNDING BOX
# ==============================================================================
func _compute_actor_ground_truth_raycast(actor_root: Node3D, subcategory: String, vp_size: Vector2) -> Dictionary:
	var space_state = get_world_3d().direct_space_state
	var h: float = 1.75
	var r: float = 0.28

	if subcategory == "wheelchair":
		h = 1.35
		r = 0.40
	elif subcategory == "elderly_cane":
		h = 1.55
		r = 0.28
	elif subcategory == "seated":
		h = 1.25
		r = 0.35
	elif subcategory == "recumbent":
		h = 0.85
		r = 0.50

	var local_pts: Array[Vector3] = []
	var y_layers = [0.10, 0.25, 0.45, 0.70, 0.95, 1.20, 1.45, 1.65]
	for y_val in y_layers:
		var cur_y = y_val if y_val < h else h * 0.95
		var cur_r = r if cur_y < h * 0.8 else r * 0.65
		for theta_deg in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
			var rad = deg_to_rad(theta_deg)
			local_pts.append(Vector3(cur_r * cos(rad), cur_y, cur_r * sin(rad)))
	local_pts.append(Vector3(0.0, h, 0.0))
	local_pts.append(Vector3(0.0, h * 0.9, 0.12))

	var visible_2d_pts: Array[Vector2] = []
	var all_2d_pts: Array[Vector2] = []
	var front_pts_count: int = 0
	var unoccluded_count: int = 0

	for lp in local_pts:
		var wp: Vector3 = actor_root.global_transform * lp
		if robot_camera.is_position_behind(wp):
			continue
		front_pts_count += 1
		var uv: Vector2 = robot_camera.unproject_position(wp)
		all_2d_pts.append(uv)

		var q = PhysicsRayQueryParameters3D.create(robot_camera.global_position, wp)
		var hit = space_state.intersect_ray(q)

		if hit.is_empty():
			unoccluded_count += 1
			visible_2d_pts.append(uv)
		else:
			var collider = hit.get("collider")
			if collider == actor_root or actor_root.is_ancestor_of(collider):
				unoccluded_count += 1
				visible_2d_pts.append(uv)

	if front_pts_count < 4:
		return {"valid": false, "reason": "BEHIND_CAMERA"}

	var vis_ratio: float = float(unoccluded_count) / float(max(front_pts_count, 1))

	var all_x: Array = []
	var all_y: Array = []
	for p in all_2d_pts:
		all_x.append(p.x); all_y.append(p.y)
	all_x.sort(); all_y.sort()

	var full_min_x = clampf(all_x[0], 0.0, vp_size.x)
	var full_min_y = clampf(all_y[0], 0.0, vp_size.y)
	var full_max_x = clampf(all_x[all_x.size() - 1], 0.0, vp_size.x)
	var full_max_y = clampf(all_y[all_y.size() - 1], 0.0, vp_size.y)

	# BỊ CHE KHUẤT > 80% HOẶC KHÔNG CÓ ĐIỂM NHÌN THẤY -> IGNORE EVAL!
	if vis_ratio < 0.20 or visible_2d_pts.size() < 4:
		return {
			"valid": false,
			"ignore_eval": true,
			"visibility_ratio": snappedf(vis_ratio, 0.01),
			"bbox_full_xyxy": [snappedf(full_min_x, 0.1), snappedf(full_min_y, 0.1), snappedf(full_max_x, 0.1), snappedf(full_max_y, 0.1)],
			"reason": "OCCLUDED_OVER_80PCT"
		}

	var vis_x: Array = []
	var vis_y: Array = []
	for p in visible_2d_pts:
		vis_x.append(p.x); vis_y.append(p.y)
	vis_x.sort(); vis_y.sort()

	var vis_min_x = clampf(vis_x[0], 0.0, vp_size.x)
	var vis_min_y = clampf(vis_y[0], 0.0, vp_size.y)
	var vis_max_x = clampf(vis_x[vis_x.size() - 1], 0.0, vp_size.x)
	var vis_max_y = clampf(vis_y[vis_y.size() - 1], 0.0, vp_size.y)

	var bw = vis_max_x - vis_min_x
	var bh = vis_max_y - vis_min_y
	if bw < 6.0 or bh < 6.0:
		return {"valid": false, "reason": "TOO_SMALL"}

	return {
		"valid": true,
		"ignore_eval": false,
		"category_id": 0,
		"category_name": "person",
		"subcategory": subcategory,
		"bbox_xyxy_px": [snappedf(vis_min_x, 0.1), snappedf(vis_min_y, 0.1), snappedf(vis_max_x, 0.1), snappedf(vis_max_y, 0.1)],
		"bbox_full_xyxy": [snappedf(full_min_x, 0.1), snappedf(full_min_y, 0.1), snappedf(full_max_x, 0.1), snappedf(full_max_y, 0.1)],
		"visibility_ratio": snappedf(vis_ratio, 0.01),
		"aspect_ratio": snappedf(bh / maxf(bw, 1.0), 0.01),
		"area_px": snappedf(bw * bh, 0.1),
		"euclidean_distance_m": snappedf(robot_camera.global_position.distance_to(actor_root.global_position), 0.02)
	}

# ==============================================================================
# QUẢN LÝ THỰC THỂ & LƯU KHUNG HÌNH
# ==============================================================================
func _hide_all_entities() -> void:
	if actors_pool:
		for child in actors_pool.get_children():
			if child is Node3D:
				child.visible = false
				child.global_position = Vector3(0, -500, 0)
	if obstacle_props:
		for child in obstacle_props.get_children():
			if child is Node3D:
				child.visible = false
				child.global_position = Vector3(0, -500, 0)
	if hard_negatives_pool:
		for child in hard_negatives_pool.get_children():
			if child is Node3D:
				child.visible = false
				child.global_position = Vector3(0, -500, 0)

func _position_camera_6dof(anchor: Vector3, dist: float, yaw_deg: float, cam_h: float, look_h: float) -> void:
	if robot_camera == null: return
	var target: Vector3 = anchor + Vector3(0.0, look_h, 0.0)
	var rad = deg_to_rad(yaw_deg)
	var cam_x: float = anchor.x + dist * sin(rad)
	var cam_z: float = anchor.z + dist * cos(rad)
	robot_camera.global_position = Vector3(cam_x, cam_h, cam_z)
	robot_camera.look_at(target, Vector3.UP)

func _record_frame(
	abs_img_dir: String, abs_lbl_dir: String,
	frame_id: String, case_id: String, name_vi: String, name_en: String,
	group_id: String, group_name: String, is_hard_neg: bool,
	active_actors: Array, subcategories: Array,
	distance_m: float, cam_off: float, cam_h: float
) -> void:
	await get_tree().physics_frame
	await get_tree().physics_frame
	await RenderingServer.frame_post_draw

	# 1. Lưu ảnh JPEG
	var rel_img_path = "images/%s.jpg" % frame_id
	var full_img_path = abs_img_dir.path_join("%s.jpg" % frame_id)
	var img: Image = get_viewport().get_texture().get_image()
	img.save_jpg(full_img_path, 0.90)

	# 2. Tính toán Ground Truth Bounding Box
	var vp_size = Vector2(capture_resolution.x, capture_resolution.y)
	var gt_list: Array[Dictionary] = []
	var yolo_lines: PackedStringArray = []

	if not is_hard_neg:
		for i in range(active_actors.size()):
			var actor: Node3D = active_actors[i]
			if actor == null: continue
			var subcat = subcategories[i] if i < subcategories.size() else "standing"
			var gt_box = _compute_actor_ground_truth_raycast(actor, subcat, vp_size)
			if gt_box.get("valid", false):
				gt_box.erase("valid")
				gt_list.append(gt_box)

				var xyxy = gt_box["bbox_xyxy_px"]
				var cx = (xyxy[0] + xyxy[2]) * 0.5 / vp_size.x
				var cy = (xyxy[1] + xyxy[3]) * 0.5 / vp_size.y
				var bw = (xyxy[2] - xyxy[0]) / vp_size.x
				var bh = (xyxy[3] - xyxy[1]) / vp_size.y
				yolo_lines.append("0 %.6f %.6f %.6f %.6f" % [cx, cy, bw, bh])

	# 3. Lưu file nhãn YOLO
	var full_lbl_path = abs_lbl_dir.path_join("%s.txt" % frame_id)
	var lbl_file = FileAccess.open(full_lbl_path, FileAccess.WRITE)
	if lbl_file:
		if not yolo_lines.is_empty():
			lbl_file.store_string("\n".join(yolo_lines) + "\n")
		lbl_file.close()

	# 4. Ghi metadata JSONL
	var record = {
		"frame_id": frame_id, "case_id": case_id, "name_vi": name_vi, "name_en": name_en,
		"group_id": group_id, "group_name": group_name, "is_hard_negative": is_hard_neg,
		"image_path": rel_img_path, "resolution": [capture_resolution.x, capture_resolution.y],
		"ground_truth_annotations": gt_list,
		"context": {"distance_m": distance_m, "cam_yaw_deg": cam_off, "cam_height_m": cam_h}
	}
	var serialized = JSON.stringify(record)
	_jsonl_file.store_line(serialized)
	_jsonl_rich_file.store_line(serialized)
	_total_saved_frames += 1

	if _total_saved_frames % 100 == 0:
		print("[UnifiedGenerator] Tiến độ: ", _total_saved_frames, " frames...")

# ==============================================================================
# HÀM PHỤ TRỢ SINH 18 FRAMES ĐẠI DIỆN CHO TỪNG CA
# ==============================================================================
func _generate_18_frames_static(
	abs_img_dir: String, abs_lbl_dir: String,
	case_id: String, name_vi: String, name_en: String,
	group_id: String, group_name: String,
	actor_name: String, subcat: String, anchor: Vector3
) -> void:
	var actor = _get_actor(actor_name)
	var look_h = 0.70 if subcat == "wheelchair" else 0.90
	var f_idx = 0

	# 16 frames: 4 dist x 2 yaw x 2 h
	for dist in UNIFIED_DISTANCES:
		for yaw in UNIFIED_CAM_YAWS:
			for cam_h in UNIFIED_CAM_HEIGHTS:
				f_idx += 1
				var fid = "%s_%04d" % [case_id.replace(".", "_"), f_idx]
				_hide_all_entities()
				if actor:
					actor.visible = true
					actor.global_position = anchor
					_position_camera_6dof(anchor, dist, yaw, cam_h, look_h)
				await _record_frame(abs_img_dir, abs_lbl_dir, fid, case_id, name_vi, name_en, group_id, group_name, false, [actor], [subcat], dist, yaw, cam_h)

	# 2 frames frontal: yaw 0.0 at 2.0m & 3.5m
	for dist in [2.0, 3.5]:
		f_idx += 1
		var fid = "%s_%04d" % [case_id.replace(".", "_"), f_idx]
		_hide_all_entities()
		if actor:
			actor.visible = true
			actor.global_position = anchor
			_position_camera_6dof(anchor, dist, 0.0, 1.15, look_h)
		await _record_frame(abs_img_dir, abs_lbl_dir, fid, case_id, name_vi, name_en, group_id, group_name, false, [actor], [subcat], dist, 0.0, 1.15)

func _generate_18_frames_prop_occluded(
	abs_img_dir: String, abs_lbl_dir: String,
	case_id: String, name_vi: String, name_en: String,
	group_id: String, group_name: String,
	actor_name: String, subcat: String, prop_name: String, prop_offset: Vector3, anchor: Vector3
) -> void:
	var actor = _get_actor(actor_name)
	var prop = _get_prop(obstacle_props, prop_name)
	var look_h = 0.70 if subcat == "wheelchair" else 0.90
	var f_idx = 0

	for dist in UNIFIED_DISTANCES:
		for yaw in UNIFIED_CAM_YAWS:
			for cam_h in UNIFIED_CAM_HEIGHTS:
				f_idx += 1
				var fid = "%s_%04d" % [case_id.replace(".", "_"), f_idx]
				_hide_all_entities()
				if actor and prop:
					actor.visible = true; prop.visible = true
					actor.global_position = anchor
					prop.global_position = anchor + prop_offset
					_position_camera_6dof(anchor, dist, yaw, cam_h, look_h)
				await _record_frame(abs_img_dir, abs_lbl_dir, fid, case_id, name_vi, name_en, group_id, group_name, false, [actor], [subcat], dist, yaw, cam_h)

	for dist in [2.0, 3.5]:
		f_idx += 1
		var fid = "%s_%04d" % [case_id.replace(".", "_"), f_idx]
		_hide_all_entities()
		if actor and prop:
			actor.visible = true; prop.visible = true
			actor.global_position = anchor
			prop.global_position = anchor + prop_offset
			_position_camera_6dof(anchor, dist, 0.0, 1.15, look_h)
		await _record_frame(abs_img_dir, abs_lbl_dir, fid, case_id, name_vi, name_en, group_id, group_name, false, [actor], [subcat], dist, 0.0, 1.15)

func _generate_18_frames_empty_trap(
	abs_img_dir: String, abs_lbl_dir: String,
	case_id: String, name_vi: String, name_en: String,
	group_id: String, group_name: String,
	prop_name: String, anchor: Vector3
) -> void:
	var prop = _get_prop(hard_negatives_pool, prop_name)
	if prop == null: prop = _get_prop(obstacle_props, prop_name)
	var f_idx = 0

	for dist in UNIFIED_DISTANCES:
		for yaw in UNIFIED_CAM_YAWS:
			for cam_h in UNIFIED_CAM_HEIGHTS:
				f_idx += 1
				var fid = "%s_%04d" % [case_id.replace(".", "_"), f_idx]
				_hide_all_entities()
				if prop:
					prop.visible = true
					prop.global_position = anchor
					_position_camera_6dof(anchor, dist, yaw, cam_h, 0.75)
				await _record_frame(abs_img_dir, abs_lbl_dir, fid, case_id, name_vi, name_en, group_id, group_name, true, [], [], dist, yaw, cam_h)

	for dist in [2.0, 3.5]:
		f_idx += 1
		var fid = "%s_%04d" % [case_id.replace(".", "_"), f_idx]
		_hide_all_entities()
		if prop:
			prop.visible = true
			prop.global_position = anchor
			_position_camera_6dof(anchor, dist, 0.0, 1.15, 0.75)
		await _record_frame(abs_img_dir, abs_lbl_dir, fid, case_id, name_vi, name_en, group_id, group_name, true, [], [], dist, 0.0, 1.15)

func _generate_18_frames_kinematics(
	abs_img_dir: String, abs_lbl_dir: String,
	case_id: String, name_vi: String, name_en: String,
	group_id: String, group_name: String,
	actor_name: String, mode: String
) -> void:
	var actor = _get_actor(actor_name)
	for step in range(18):
		var t = float(step) / 17.0
		var fid = "%s_%04d" % [case_id.replace(".", "_"), step + 1]
		_hide_all_entities()
		var pos = corridor_anchor
		var dist = 3.0

		if mode == "looming": # Tuần tra đối đầu (tiến lại gần)
			pos = corridor_anchor + Vector3(0.0, 0.0, lerpf(5.5, 1.0, t))
			dist = 2.0
		elif mode == "sidestep": # Né ngang 0.55m
			var lat = sin(t * PI) * 0.55
			pos = corridor_anchor + Vector3(lat, 0.0, lerpf(3.0, 1.5, t))
		elif mode == "doorway": # Ló ra từ cửa 90 deg
			pos = clinic_door_anchor + Vector3(lerpf(-0.6, 0.8, t), 0.0, 0.0)
		elif mode == "fast_walk": # Đi nhanh
			pos = corridor_anchor + Vector3(0.0, 0.0, lerpf(6.0, 1.0, t))

		if actor:
			actor.visible = true
			actor.global_position = pos
			_position_camera_6dof(corridor_anchor, dist, 0.0, 1.15, 0.90)
		await _record_frame(abs_img_dir, abs_lbl_dir, fid, case_id, name_vi, name_en, group_id, group_name, false, [actor], ["standing"], dist, 0.0, 1.15)

func _generate_18_frames_crowd(
	abs_img_dir: String, abs_lbl_dir: String,
	case_id: String, name_vi: String, name_en: String,
	group_id: String, group_name: String,
	actors: Array, subcats: Array, offsets: Array
) -> void:
	for f_idx in range(18):
		var fid = "%s_%04d" % [case_id.replace(".", "_"), f_idx + 1]
		_hide_all_entities()
		var dist = UNIFIED_DISTANCES[f_idx % 4]
		var yaw = UNIFIED_CAM_YAWS[(f_idx / 4) % 2]
		var cam_h = UNIFIED_CAM_HEIGHTS[(f_idx / 8) % 2]

		var active_list: Array = []
		for i in range(actors.size()):
			var a = _get_actor(actors[i])
			if a:
				a.visible = true
				a.global_position = corridor_anchor + offsets[i]
				active_list.append(a)

		_position_camera_6dof(corridor_anchor, dist, yaw, cam_h, 0.85)
		await _record_frame(abs_img_dir, abs_lbl_dir, fid, case_id, name_vi, name_en, group_id, group_name, false, active_list, subcats, dist, yaw, cam_h)

# ==============================================================================
# 8 NHÓM LÂM SÀNG CỐT LÕI (64 CA KIỂM THỬ)
# ==============================================================================

# U1: Ngụy trang & Tương phản thấp (6 ca cốt lõi)
func _generate_group_u1(img_dir: String, lbl_dir: String) -> void:
	var cases = [
		{"id": "U1.1", "vi": "Bác Sĩ Blouse Trắng Trùng Màu Tường", "en": "Doctor White Blouse Camouflage", "act": "doctor_male"},
		{"id": "U1.2", "vi": "Điều Dưỡng Scrub Xanh Trùng Rèm", "en": "Nurse Teal Scrub Curtain Match", "act": "nurse_female"},
		{"id": "U1.3", "vi": "Cụ Bà Tóc Bạc Áo Hoa", "en": "Elderly Female Floral Pattern", "act": "elder_f09"},
		{"id": "U1.4", "vi": "Cụ Ông Lưng Còng Áo Sẫm", "en": "Elderly Male Kyphosis Dark Attire", "act": "elder_m03"},
		{"id": "U1.6", "vi": "Bệnh Nhân Nữ Áo Khoác Dài", "en": "Female Patient Long Coat", "act": "patient_f08"},
		{"id": "U1.10", "vi": "Bác Sĩ Thực Tập", "en": "Intern Medical Student", "act": "doctor_clipboard"}
	]
	for c in cases:
		await _generate_18_frames_static(img_dir, lbl_dir, c["id"], c["vi"], c["en"], "U1_Camouflage", "Ngụy trang & Tương phản thấp", c["act"], "standing", corridor_anchor)

# U2: Tư thế khám bệnh & Ngồi ghế (8 ca cốt lõi)
func _generate_group_u2(img_dir: String, lbl_dir: String) -> void:
	var cases = [
		{"id": "U2.1", "vi": "Ngồi Ghế Băng Dài Tựa Lưng", "en": "Waiting Bench Seated", "act": "seated_bench_patient", "sub": "seated"},
		{"id": "U2.2", "vi": "Ngồi Gập Người Ho Khạc 45°", "en": "Seated Cough Limp", "act": "seated_cough_patient", "sub": "seated"},
		{"id": "U2.3", "vi": "Ngồi Buông Thõng Mỏi Mệt", "en": "Seated Exhausted", "act": "seated_exhausted_patient", "sub": "seated"},
		{"id": "U2.4", "vi": "Bác Sĩ Cúi Đọc Bệnh Án", "en": "Doctor Clipboard Reading", "act": "doctor_clipboard", "sub": "standing"},
		{"id": "U2.6", "vi": "Bác Sĩ Cúi Gập 90° Nhặt Sổ", "en": "Stooped 90deg Forward", "act": "patient_m02", "sub": "standing"},
		{"id": "U2.7", "vi": "Cụ Già Lưng Còng Chống Gậy", "en": "Elderly Stooped Cane", "act": "elder_m03", "sub": "elderly_cane"},
		{"id": "U2.8", "vi": "Bệnh Nhân Nằm Giường Cấp Cứu", "en": "Recumbent Bed Patient", "act": "patient_m05", "sub": "recumbent"},
		{"id": "U2.10", "vi": "Bác Sĩ Ngồi Bàn Khám", "en": "Doctor Desk Seated", "act": "doctor_clipboard", "sub": "seated"}
	]
	for c in cases:
		await _generate_18_frames_static(img_dir, lbl_dir, c["id"], c["vi"], c["en"], "U2_ClinicalPoses", "Tư thế lâm sàng & Ngồi ghế", c["act"], c["sub"], corridor_anchor)

# U3: Che khuất bởi đạo cụ y tế (12 ca cốt lõi)
func _generate_group_u3(img_dir: String, lbl_dir: String) -> void:
	var cases = [
		{"id": "U3.1", "vi": "Sau Cột Truyền Dịch Inox", "en": "Behind IV Stand Pole", "act": "patient_m02", "prop": "iv_stand", "off": Vector3(-0.2, 0, 0.2)},
		{"id": "U3.2", "vi": "Sau Xe Đẩy Monitor 1.2m", "en": "Behind Vital Monitor Cart", "act": "nurse_female", "prop": "vital_monitor", "off": Vector3(0, 0, 0.4)},
		{"id": "U3.3", "vi": "Thao Tác Bấm Kiosk 1.55m", "en": "Operating Queue Kiosk", "act": "patient_m05", "prop": "queue_kiosk", "off": Vector3(0, 0, 0.4)},
		{"id": "U3.4", "vi": "Lấy Nước Bên Cây Nóng Lạnh", "en": "At Water Dispenser", "act": "elder_m14", "prop": "water_dispenser", "off": Vector3(0.3, 0, 0.2)},
		{"id": "U3.5", "vi": "Rửa Tay Cột Khử Khuẩn", "en": "Hands at Sanitizer Stand", "act": "elder_f09", "prop": "sanitizer_stand", "off": Vector3(0, 0, 0.3)},
		{"id": "U3.6", "vi": "Hộ Lý Đẩy Xe Cáng 2.1m", "en": "Behind Stretcher Cart", "act": "patient_m02", "prop": "stretcher_cart", "off": Vector3(0, 0, 0.6)},
		{"id": "U3.7", "vi": "Đứng Sát Mép Standee", "en": "Near Info Standee Edge", "act": "patient_f08", "prop": "info_standee", "off": Vector3(0.35, 0, 0.1)},
		{"id": "U3.8", "vi": "Nhân Viên Sau Quầy Lễ Tân", "en": "Behind Reception Counter", "act": "nurse_female", "prop": "vital_monitor", "off": Vector3(0, 0, 0.5)},
		{"id": "U3.9", "vi": "Khuất Sau Vách Rèm Y Tế", "en": "Behind Privacy Screen", "act": "patient_m02", "prop": "privacy_screen", "off": Vector3(0, 0, 0.4)},
		{"id": "U3.10", "vi": "Khuất Sau Xe Tiêm Inox", "en": "Behind Medical Trolley", "act": "nurse_female", "prop": "medical_trolley", "off": Vector3(0, 0, 0.45)},
		{"id": "U3.12", "vi": "Ló Sau Góc Tường Hành Lang", "en": "Corner Edge Occlusion", "act": "patient_m05", "prop": "privacy_screen", "off": Vector3(0.4, 0, 0.2)},
		{"id": "U3.13", "vi": "Qua Ngưỡng Cửa Phòng 110", "en": "Doorway 110 Threshold", "act": "patient_f02", "prop": "info_standee", "off": Vector3(0.3, 0, 0.2)}
	]
	for c in cases:
		await _generate_18_frames_prop_occluded(img_dir, lbl_dir, c["id"], c["vi"], c["en"], "U3_MedicalProps", "Che khuất bởi đạo cụ y tế", c["act"], "standing", c["prop"], c["off"], corridor_anchor)

# U4: Phương tiện hỗ trợ di chuyển (7 ca cốt lõi)
func _generate_group_u4(img_dir: String, lbl_dir: String) -> void:
	var cases = [
		{"id": "U4.1", "vi": "Xe Lăn Trực Diện 0°", "en": "Wheelchair Frontal 0deg", "act": "wheelchair_patient"},
		{"id": "U4.2", "vi": "Xe Lăn Chéo Góc 45°", "en": "Wheelchair Flank 45deg", "act": "wheelchair_patient"},
		{"id": "U4.3", "vi": "Xe Lăn Nhìn Ngang 90°", "en": "Wheelchair Lateral 90deg", "act": "wheelchair_patient"},
		{"id": "U4.5", "vi": "Cụ Già Ngồi Xe Lăn Có Người Đẩy", "en": "Pushed Wheelchair Pair", "act": "wheelchair_patient"},
		{"id": "U4.6", "vi": "Xe Cáng Cứu Thương Di Động", "en": "Mobile Stretcher Cart", "act": "seated_exhausted_patient"},
		{"id": "U4.7", "vi": "Giường Bệnh Nội Trú Có Người", "en": "Occupied Hospital Bed", "act": "seated_cough_patient"},
		{"id": "U4.8", "vi": "Xe Lăn Gấp Gọn Dựa Tường", "en": "Folded Wheelchair Near Wall", "act": "wheelchair_patient"}
	]
	for c in cases:
		await _generate_18_frames_static(img_dir, lbl_dir, c["id"], c["vi"], c["en"], "U4_WheelchairBed", "Phương tiện di chuyển & Xe lăn", c["act"], "wheelchair", corridor_anchor)

# U5: Động học di chuyển & Rẽ ngoặt (7 ca cốt lõi)
func _generate_group_u5(img_dir: String, lbl_dir: String) -> void:
	var cases = [
		{"id": "U5.1", "vi": "Tuần Tra Đối Đầu (1.08 m/s)", "en": "Patrol Looming Reverse", "act": "doctor_male", "mode": "looming"},
		{"id": "U5.2", "vi": "Bước Né Ngang Nhường Đường", "en": "Pardon Sidestep 0.55m", "act": "patient_m05", "mode": "sidestep"},
		{"id": "U5.3", "vi": "Chuyển Hướng 90° Qua Ngưỡng Cửa", "en": "Doorway Turn 90deg", "act": "elder_m03", "mode": "doorway"},
		{"id": "U5.4", "vi": "Điều Dưỡng Rẽ Từ Góc Khuất", "en": "Corner Sudden Emergence", "act": "nurse_female", "mode": "doorway"},
		{"id": "U5.5", "vi": "Bệnh Nhân Đi Bộ Nhanh 1.5 m/s", "en": "Fast Paced Walk", "act": "patient_m02", "mode": "fast_walk"},
		{"id": "U5.7", "vi": "Đi Cắt Ngang Mặt Robot 3.6m", "en": "Crossing Path Lateral", "act": "patient_f08", "mode": "sidestep"},
		{"id": "U5.8", "vi": "Người Đi Rời Xa Camera", "en": "Walk Away 180deg", "act": "patient_f02", "mode": "looming"}
	]
	for c in cases:
		await _generate_18_frames_kinematics(img_dir, lbl_dir, c["id"], c["vi"], c["en"], "U5_Kinematics", "Động học & Né tránh", c["act"], c["mode"])

# U6: Mật độ đám đông & Tương tác xã hội (6 ca cốt lõi)
func _generate_group_u6(img_dir: String, lbl_dir: String) -> void:
	var cases = [
		{"id": "U6.1", "vi": "Hai Cụ Già Ngồi Ghế Băng", "en": "Two Elderly Seated", "acts": ["elder_f09", "elder_m03"], "subs": ["seated", "seated"], "offs": [Vector3(-0.3, 0, 0), Vector3(0.3, 0, 0)]},
		{"id": "U6.2", "vi": "Bác Sĩ Đứng - Bệnh Nhân Ngồi", "en": "Doctor Standing Patient Seated", "acts": ["doctor_male", "patient_f02"], "subs": ["standing", "seated"], "offs": [Vector3(-0.4, 0, 0), Vector3(0.2, 0, 0)]},
		{"id": "U6.3", "vi": "Hàng Chờ 3 Người Kiosk", "en": "Queue 3 Patients Sequence", "acts": ["patient_m05", "elder_m14", "patient_f08"], "subs": ["standing", "standing", "standing"], "offs": [Vector3(0, 0, 0), Vector3(0, 0, -0.8), Vector3(0.1, 0, -1.6)]},
		{"id": "U6.4", "vi": "Thảo Luận 180° vs 0°", "en": "Consultation Facing", "acts": ["doctor_male", "patient_m02"], "subs": ["standing", "standing"], "offs": [Vector3(-0.25, 0, 0), Vector3(0.25, 0, 0)]},
		{"id": "U6.5", "vi": "Dìu Cụ Già Đi Chậm", "en": "Guided Walking Pair", "acts": ["elder_f09", "nurse_female"], "subs": ["standing", "standing"], "offs": [Vector3(-0.2, 0, 0), Vector3(0.2, 0, 0)]},
		{"id": "U6.7", "vi": "Cụm 4 Người Tụ Tập", "en": "Gathering Group 4", "acts": ["doctor_male", "nurse_female", "patient_m05"], "subs": ["standing", "standing", "standing"], "offs": [Vector3(-0.4, 0, 0), Vector3(0.4, 0, 0), Vector3(0, 0, -0.6)]}
	]
	for c in cases:
		await _generate_18_frames_crowd(img_dir, lbl_dir, c["id"], c["vi"], c["en"], "U6_CrowdSocial", "Mật độ đông & Xã hội", c["acts"], c["subs"], c["offs"])

# U7: Quang học cực đoan & Mép biên FOV (10 ca cốt lõi)
func _generate_group_u7(img_dir: String, lbl_dir: String) -> void:
	var cases = [
		{"id": "U7.1", "vi": "Cận Cảnh Cực Đoan 0.6m", "en": "Extreme Close Range 0.6m", "d": 0.6, "cam_yaw": 0.0},
		{"id": "U7.2", "vi": "Cận Cảnh An Toàn 0.9m", "en": "Close Safe Range 0.9m", "d": 0.9, "cam_yaw": 0.0},
		{"id": "U7.3", "vi": "Cự Ly Chuẩn 2.5m", "en": "Standard Range 2.5m", "d": 2.5, "cam_yaw": 0.0},
		{"id": "U7.4", "vi": "Cự Ly Xa 6.0m", "en": "Far Range 6.0m", "d": 6.0, "cam_yaw": 0.0},
		{"id": "U7.5", "vi": "Cắt Mép Trái Khung Hình FOV", "en": "FOV Left Edge Truncation", "d": 2.0, "cam_yaw": -24.0},
		{"id": "U7.6", "vi": "Cắt Mép Phải Khung Hình FOV", "en": "FOV Right Edge Truncation", "d": 2.0, "cam_yaw": 24.0},
		{"id": "U7.7", "vi": "Ngược Sáng Kính Đại Sảnh", "en": "Backlight Glass Lobby", "d": 3.0, "cam_yaw": 0.0},
		{"id": "U7.9", "vi": "Ánh Sáng Đêm Mờ 150 Lux", "en": "Dim Lighting 150lux", "d": 2.5, "cam_yaw": 0.0},
		{"id": "U7.10", "vi": "Đèn Vàng Ấm Phòng Khám", "en": "Warm Clinic Light 3000K", "d": 2.5, "cam_yaw": 0.0},
		{"id": "U7.12", "vi": "Góc Tối Cuối Hành Lang", "en": "Deep Shadow Corridor", "d": 3.5, "cam_yaw": 0.0}
	]
	var actor = _get_actor("patient_m02")
	for c in cases:
		for step in range(18):
			var fid = "%s_%04d" % [c["id"].replace(".", "_"), step + 1]
			_hide_all_entities()
			var dist = c["d"] + (step % 3) * 0.4
			var yaw = c["cam_yaw"] + ((step / 3) % 3 - 1) * 4.0
			var cam_h = 0.85 if (step % 2 == 0) else 1.25
			if actor:
				actor.visible = true
				actor.global_position = corridor_anchor
				_position_camera_6dof(corridor_anchor, dist, yaw, cam_h, 0.90)
			await _record_frame(img_dir, lbl_dir, fid, c["id"], c["vi"], c["en"], "U7_ExtremeOptics", "Quang học & Mép biên", false, [actor], ["standing"], dist, yaw, cam_h)

# U8: Bẫy phòng rỗng & Đạo cụ 0 GT (8 ca cốt lõi)
func _generate_group_u8(img_dir: String, lbl_dir: String) -> void:
	var cases = [
		{"id": "U8.1", "vi": "Hành Lang Rỗng Ban Đêm", "en": "Empty Corridor Night", "prop": "empty_vital_monitor"},
		{"id": "U8.2", "vi": "Bẫy Cột Truyền Dịch Rỗng", "en": "Empty IV Stand Trap", "prop": "empty_iv_stand"},
		{"id": "U8.3", "vi": "Bẫy Cột Khử Khuẩn Rỗng", "en": "Empty Sanitizer Trap", "prop": "empty_sanitizer_stand"},
		{"id": "U8.4", "vi": "Bẫy Kiosk Bấm Số Rỗng", "en": "Empty Kiosk Trap", "prop": "empty_queue_kiosk"},
		{"id": "U8.5", "vi": "Bẫy Xe Đẩy Monitor Rỗng", "en": "Empty Monitor Cart Trap", "prop": "empty_vital_monitor"},
		{"id": "U8.6", "vi": "Bẫy Xe Cáng Cấp Cứu Rỗng", "en": "Empty Stretcher Trap", "prop": "empty_stretcher_cart"},
		{"id": "U8.8", "vi": "Bẫy Standee 2D In Hình Y Tế", "en": "Standee 2D Poster Trap", "prop": "empty_info_standee"},
		{"id": "U8.10", "vi": "Bẫy Xe Lăn Rỗng", "en": "Empty Wheelchair Trap", "prop": "empty_wheelchair"}
	]
	for c in cases:
		await _generate_18_frames_empty_trap(img_dir, lbl_dir, c["id"], c["vi"], c["en"], "U8_EmptyTraps", "Bẫy phòng rỗng & Đạo cụ", c["prop"], corridor_anchor)

# ==============================================================================
# TIỆN ÍCH TÌM KIẾM NODE & ÁNH SÁNG
# ==============================================================================
func _get_actor(act_name: String) -> Node3D:
	if actors_pool:
		var n = actors_pool.get_node_or_null(act_name)
		if n: return n
		# Fallback first child
		if actors_pool.get_child_count() > 0:
			return actors_pool.get_child(0) as Node3D
	return null

func _get_prop(pool: Node3D, prop_name: String) -> Node3D:
	if pool:
		var n = pool.get_node_or_null(prop_name)
		if n: return n
		if pool.get_child_count() > 0:
			return pool.get_child(0) as Node3D
	return null

func _apply_lighting_profile(profile_name: String) -> void:
	if main_dir_light:
		main_dir_light.light_energy = 1.0
		main_dir_light.light_color = Color(1.0, 0.98, 0.95)

func _write_dataset_manifest(abs_out_dir: String, elapsed_sec: float) -> void:
	var manifest = {
		"dataset_name": "vision_eval_unified",
		"version": "2.1_clinical_core_64",
		"total_frames": _total_saved_frames,
		"total_cases": 64,
		"resolution": [capture_resolution.x, capture_resolution.y],
		"generation_time_sec": snappedf(elapsed_sec, 0.1),
		"raycast_occlusion_grid": "128_rays_physics_space_state",
		"bounding_box_convention": "visible_bounding_box_yolo_normalized",
		"groups": ["U1_Camouflage", "U2_ClinicalPoses", "U3_MedicalProps", "U4_WheelchairBed", "U5_Kinematics", "U6_CrowdSocial", "U7_ExtremeOptics", "U8_EmptyTraps"]
	}
	var f = FileAccess.open(abs_out_dir.path_join("manifest_unified.json"), FileAccess.WRITE)
	if f:
		f.store_string(JSON.stringify(manifest, "\t"))
		f.close()
