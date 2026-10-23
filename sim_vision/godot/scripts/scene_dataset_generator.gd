# scripts/scene_dataset_generator.gd
class_name SceneDatasetGenerator
extends Node3D

@export_group("Scene References")
@export var robot_camera: Camera3D
@export var world_env: WorldEnvironment
@export var main_dir_light: DirectionalLight3D
@export var actors_pool: Node3D
@export var obstacle_props: Node3D
@export var hard_negatives_pool: Node3D

@export_group("Anchor Positions in Hospital Map")
@export var corridor_anchor: Vector3 = Vector3(0.0, 0.0, 0.0)
@export var reception_anchor: Vector3 = Vector3(0.0, 0.0, 6.5) # Trước quầy lễ tân
@export var doorway_110_anchor: Vector3 = Vector3(0.0, 0.0, -2.0) # Cửa phòng xét nghiệm 110
@export var doorway_101_anchor: Vector3 = Vector3(-8.0, 0.0, -2.0) # Cửa phòng cấp cứu 101

@export_group("Dataset Configuration")
@export var output_root_dir: String = "res://../data/vision_eval_1200"
@export var capture_resolution: Vector2i = Vector2i(640, 480)
@export var camera_height_m: float = 1.15
@export var random_seed: int = 20261023
@export var auto_start_on_ready: bool = true

var _rng := RandomNumberGenerator.new()
var _annotations_file: FileAccess
var _total_saved_frames: int = 0
var _start_time_msec: int = 0


func _ready() -> void:
	var cmd_args: PackedStringArray = OS.get_cmdline_user_args()
	if auto_start_on_ready or ("--generate-dataset-1200" in cmd_args) or ("--generate-dataset" in cmd_args):
		call_deferred("run_full_1200_pipeline")


## Hàm điều phối chính: Chạy tuần tự 8 Trục (Dimensions) sinh đúng 1,200 ảnh + annotations.jsonl
func run_full_1200_pipeline() -> void:
	_start_time_msec = Time.get_ticks_msec()
	_rng.seed = random_seed

	var vp: Viewport = get_viewport()
	vp.size = capture_resolution
	vp.use_taa = false

	var abs_out_dir: String = ProjectSettings.globalize_path(output_root_dir)
	var abs_img_dir: String = abs_out_dir.path_join("images")
	DirAccess.make_dir_recursive_absolute(abs_img_dir)

	var jsonl_path: String = abs_out_dir.path_join("annotations.jsonl")
	_annotations_file = FileAccess.open(jsonl_path, FileAccess.WRITE)
	if _annotations_file == null:
		push_error("[DatasetGenerator] Không thể mở file để ghi tại: " + jsonl_path)
		get_tree().quit(1)
		return

	print("==================================================================")
	print("[DatasetGenerator] BẮT ĐẦU SINH BỘ DỮ LIỆU ĐÁNH GIÁ 48-CASES (1,200 FRAMES)")
	print("[DatasetGenerator] Thư mục đích: ", abs_out_dir)
	print("==================================================================")

	_hide_all_entities()
	_apply_lighting_profile("fluorescent_normal")

	# Chạy tuần tự 8 Trục (mỗi trục 150 frames = 1,200 frames)
	await _generate_d1_architecture(abs_img_dir)  # 150 frames (C01-C06)
	await _generate_d2_furniture(abs_img_dir)     # 150 frames (C07-C12)
	await _generate_d3_clinical_poses(abs_img_dir)# 150 frames (C13-C18)
	await _generate_d4_crowd_dynamics(abs_img_dir)# 150 frames (C19-C24)
	await _generate_d5_distances(abs_img_dir)     # 150 frames (C25-C30)
	await _generate_d6_kinematics(abs_img_dir)    # 150 frames (C31-C36)
	await _generate_d7_lighting(abs_img_dir)      # 150 frames (C37-C42)
	await _generate_d8_hard_negatives(abs_img_dir)# 150 frames (C43-C48)

	_annotations_file.flush()
	_annotations_file.close()

	var elapsed_sec: float = (Time.get_ticks_msec() - _start_time_msec) / 1000.0
	_write_5w1h_manifest(abs_out_dir, elapsed_sec)

	print("==================================================================")
	print("[DatasetGenerator] HOÀN TẤT! Đã xuất ", _total_saved_frames, " frames trong ", snappedf(elapsed_sec, 0.1), "s.")
	print("==================================================================")
	get_tree().quit(0)


# ==============================================================================
# 8 TRỤC KIỂM THỬ KHÔNG GIAN (48 TEST CASES)
# ==============================================================================

## TRỤC 1: Che Khuất Bởi Kiến Trúc & Cửa Tự Động (150 frames: C01-C06)
func _generate_d1_architecture(abs_img_dir: String) -> void:
	var actor: Node3D = _get_actor("standing_male")
	var shifts: Array[float] = [-0.6, -0.3, 0.0, 0.3, 0.6] # Mức độ thò ra khỏi mép tường
	var distances: Array[float] = [1.8, 2.5, 3.5, 4.5, 6.0]
	var cam_yaw_offsets: Array[float] = [-12.0, 0.0, 12.0]

	var idx: int = 0
	# C01 & C02: Mép cửa phòng khám 110 & mép tường
	for shift in shifts:
		for dist in distances:
			for yaw_off in cam_yaw_offsets:
				idx += 1
				var frame_id: String = "D1_%04d" % idx
				_hide_all_entities()
				if actor:
					actor.visible = true
					actor.global_position = doorway_110_anchor + Vector3(shift, 0.0, -0.6)
					actor.rotation_degrees = Vector3(0.0, 0.0, 0.0)
					_position_camera_spherical(doorway_110_anchor, dist, 0.0, yaw_off, 0.95)
				await _capture_and_record_frame(
					abs_img_dir, frame_id, "D1_Architecture", "doorway_110",
					"fluorescent_normal", [actor] if actor else [], ["standing"]
				)


## TRỤC 2: Che Khuất Bởi Nội Thất & Đạo Cụ Y Tế (150 frames: C07-C12)
func _generate_d2_furniture(abs_img_dir: String) -> void:
	var actor: Node3D = _get_actor("standing_female")
	var trolley: Node3D = _get_prop(obstacle_props, "medical_trolley")
	var screen: Node3D = _get_prop(obstacle_props, "privacy_screen")

	var distances: Array[float] = [1.5, 2.2, 3.0, 4.0, 5.2]
	var configs: Array[Dictionary] = [
		{"mode": "counter", "look_h": 0.85},
		{"mode": "screen", "look_h": 0.95},
		{"mode": "trolley", "look_h": 0.95}
	]
	var cam_yaw_offsets: Array[float] = [-15.0, 0.0, 15.0]

	var idx: int = 0
	for cfg in configs:
		for dist in distances:
			for shift_step in range(5):
				var lat_shift: float = (shift_step - 2) * 0.25
				var yaw_off: float = cam_yaw_offsets[shift_step % 3]
				idx += 1
				var frame_id: String = "D2_%04d" % idx
				_hide_all_entities()
				if actor:
					actor.visible = true
					if cfg["mode"] == "counter":
						# Đứng sau quầy lễ tân (y = 7.8m trong room.tscn)
						actor.global_position = Vector3(lat_shift, 0.0, 8.4)
						_position_camera_spherical(Vector3(lat_shift, 0.0, 7.8), dist, 180.0, yaw_off, cfg["look_h"])
					elif cfg["mode"] == "screen":
						# Đứng sau rèm che riêng tư
						if screen:
							screen.visible = true
							screen.global_position = corridor_anchor + Vector3(0.0, 0.0, 0.5)
						actor.global_position = corridor_anchor + Vector3(lat_shift, 0.0, 0.0)
						_position_camera_spherical(corridor_anchor, dist, 0.0, yaw_off, cfg["look_h"])
					else:
						# Đứng sau xe đẩy thuốc
						if trolley:
							trolley.visible = true
							trolley.global_position = corridor_anchor + Vector3(0.0, 0.0, 0.6)
						actor.global_position = corridor_anchor + Vector3(lat_shift, 0.0, 0.0)
						_position_camera_spherical(corridor_anchor, dist, 0.0, yaw_off, cfg["look_h"])

				await _capture_and_record_frame(
					abs_img_dir, frame_id, "D2_Furniture", "hospital_furniture",
					"fluorescent_normal", [actor] if actor else [], ["standing"]
				)


## TRỤC 3: Dị Biệt Dáng Đi & Bệnh Lý Lâm Sàng (150 frames: C13-C18)
func _generate_d3_clinical_poses(abs_img_dir: String) -> void:
	var actor_configs: Array[Dictionary] = [
		{"name": "wheelchair_patient", "subcat": "wheelchair", "look_h": 0.75},
		{"name": "elderly_stooped_cane", "subcat": "elderly_cane", "look_h": 0.82},
		{"name": "recumbent_bed_patient", "subcat": "recumbent_bed", "look_h": 0.65},
		{"name": "deep_stoop_patient", "subcat": "elderly_cane", "look_h": 0.55},
		{"name": "seated_bench_patient", "subcat": "wheelchair", "look_h": 0.75}
	]
	var distances: Array[float] = [1.2, 2.0, 3.0, 4.2, 5.5]
	var npc_yaws: Array[float] = [0.0, 45.0, 90.0]
	var cam_yaw_offsets: Array[float] = [-10.0, 10.0]

	var idx: int = 0
	for cfg in actor_configs:
		var actor: Node3D = _get_actor(cfg["name"])
		for dist in distances:
			for npc_yaw in npc_yaws:
				for cam_off in cam_yaw_offsets:
					idx += 1
					var frame_id: String = "D3_%04d" % idx
					_hide_all_entities()
					if actor:
						actor.visible = true
						_stage_single_actor(actor, corridor_anchor, dist, 0.0, npc_yaw, cam_off, cfg["look_h"])
					await _capture_and_record_frame(
						abs_img_dir, frame_id, "D3_Clinical_Poses", "corridor_clinic",
						"fluorescent_normal", [actor] if actor else [], [cfg["subcat"]]
					)


## TRỤC 4: Tương Tác Xã Hội, Đám Đông & NMS Stress Test (150 frames: C19-C24)
func _generate_d4_crowd_dynamics(abs_img_dir: String) -> void:
	var wc_actor: Node3D = _get_actor("wheelchair_patient")
	var male: Node3D = _get_actor("standing_male")
	var female: Node3D = _get_actor("standing_female")
	var child: Node3D = _get_actor("child_patient")
	var elderly: Node3D = _get_actor("elderly_stooped_cane")

	var distances: Array[float] = [1.8, 2.6, 3.6, 4.8, 6.0]
	var lateral_shifts: Array[float] = [-0.3, -0.15, 0.0, 0.15, 0.3]
	var cam_yaw_offsets: Array[float] = [-8.0, 8.0]

	var idx: int = 0
	for formation in range(3):
		for dist in distances:
			for lat_shift in lateral_shifts:
				for cam_off in cam_yaw_offsets:
					idx += 1
					var frame_id: String = "D4_%04d" % idx
					_hide_all_entities()
					var active_actors: Array = []
					var active_subcats: Array = []

					if formation == 0:
						# Người nhà đẩy xe lăn
						if wc_actor and male:
							wc_actor.visible = true; male.visible = true
							wc_actor.global_position = corridor_anchor
							male.global_position = corridor_anchor + Vector3(lat_shift, 0.0, -0.75)
							active_actors = [wc_actor, male]
							active_subcats = ["wheelchair", "standing"]
					elif formation == 1:
						# Mẹ dắt tay trẻ em
						if female and child:
							female.visible = true; child.visible = true
							female.global_position = corridor_anchor + Vector3(-0.3, 0.0, 0.0)
							child.global_position = corridor_anchor + Vector3(0.3 + lat_shift * 0.5, 0.0, 0.0)
							active_actors = [female, child]
							active_subcats = ["standing", "standing"]
					else:
						# Hàng 3 người xếp hàng
						if female and elderly and male:
							female.visible = true; elderly.visible = true; male.visible = true
							female.global_position = corridor_anchor + Vector3(0.0, 0.0, 0.4)
							elderly.global_position = corridor_anchor + Vector3(lat_shift, 0.0, -0.3)
							male.global_position = corridor_anchor + Vector3(0.2, 0.0, -1.0)
							active_actors = [female, elderly, male]
							active_subcats = ["standing", "elderly_cane", "standing"]

					_position_camera_spherical(corridor_anchor, dist, 0.0, cam_off, 0.90)
					await _capture_and_record_frame(
						abs_img_dir, frame_id, "D4_Crowd_Dyn", "corridor_crowd",
						"fluorescent_normal", active_actors, active_subcats
					)


## TRỤC 5: Biến Dạng Cự Ly & Cắt Biên Ảnh (150 frames: C25-C30)
func _generate_d5_distances(abs_img_dir: String) -> void:
	var actor: Node3D = _get_actor("standing_male")
	var distances: Array[float] = [0.6, 0.85, 1.5, 3.5, 6.5, 10.0] # Cận cảnh cực đoan 0.6m đến 10m
	var yaw_offsets: Array[float] = [-22.0, -11.0, 0.0, 11.0, 22.0] # Cắt mép trái/phải

	var idx: int = 0
	for d in distances:
		for yaw_off in yaw_offsets:
			for repeat in range(5):
				idx += 1
				var frame_id: String = "D5_%04d" % idx
				_hide_all_entities()
				if actor:
					actor.visible = true
					var h_look: float = 0.95 if d > 0.8 else 1.10
					_stage_single_actor(actor, corridor_anchor, d, 0.0, 0.0, yaw_off, h_look)
				await _capture_and_record_frame(
					abs_img_dir, frame_id, "D5_Distances", "corridor_distance",
					"fluorescent_normal", [actor] if actor else [], ["standing"]
				)


## TRỤC 6: Hướng Chuyển Động & Động Học (150 frames: C31-C36)
func _generate_d6_kinematics(abs_img_dir: String) -> void:
	var actor: Node3D = _get_actor("standing_female")
	var yaws: Array[float] = [0.0, 45.0, 90.0, 135.0, 180.0, 270.0] # Đủ các hướng xoay
	var distances: Array[float] = [1.2, 2.2, 3.5, 5.0, 7.0]

	var idx: int = 0
	for yaw in yaws:
		for dist in distances:
			for cam_yaw in [-10.0, 0.0, 10.0]:
				idx += 1
				var frame_id: String = "D6_%04d" % idx
				_hide_all_entities()
				if actor:
					actor.visible = true
					_stage_single_actor(actor, corridor_anchor, dist, 0.0, yaw, cam_yaw, 0.95)
				await _capture_and_record_frame(
					abs_img_dir, frame_id, "D6_Kinematics", "corridor_kinematics",
					"fluorescent_normal", [actor] if actor else [], ["standing"]
				)


## TRỤC 7: Nhiễu Ánh Sáng & Hiệu Ứng Bề Mặt PBR (150 frames: C37-C42)
func _generate_d7_lighting(abs_img_dir: String) -> void:
	var profiles: Array[String] = [
		"backlit_glare", "dim_emergency_light", "overexposed_fluorescent",
		"fluorescent_normal", "dim_emergency_light"
	]
	var actor: Node3D = _get_actor("standing_male")
	var distances: Array[float] = [1.2, 2.2, 3.5, 5.0, 6.5]
	var cam_yaw_offsets: Array[float] = [-12.0, 0.0, 12.0]

	var idx: int = 0
	for p_idx in range(profiles.size()):
		var prof: String = profiles[p_idx]
		_apply_lighting_profile(prof)
		for dist in distances:
			for yaw_off in cam_yaw_offsets:
				idx += 1
				var frame_id: String = "D7_%04d" % idx
				_hide_all_entities()
				if actor:
					actor.visible = true
					_stage_single_actor(actor, corridor_anchor, dist, 0.0, 0.0, yaw_off, 0.95)
				await _capture_and_record_frame(
					abs_img_dir, frame_id, "D7_Lighting", "corridor_lighting",
					prof, [actor] if actor else [], ["standing"]
				)

	_apply_lighting_profile("fluorescent_normal")


## TRỤC 8: Bẫy Dương Tính Giả Trên Đạo Cụ Rỗng (150 frames: C43-C48)
func _generate_d8_hard_negatives(abs_img_dir: String) -> void:
	var prop_names: Array[String] = [
		"empty_wheelchair",
		"empty_hospital_bed",
		"iv_pole_stretcher",
		"self_service_kiosk",
		"wall_doctor_poster",
		"medical_privacy_screen_empty"
	]
	var distances: Array[float] = [1.2, 2.0, 3.0, 4.2, 5.5]
	var cam_offsets: Array[float] = [-15.0, 0.0, 15.0]

	var idx: int = 0
	for p_name in prop_names:
		var prop: Node3D = _get_prop(hard_negatives_pool, p_name)
		for dist in distances:
			for yaw_off in cam_offsets:
				idx += 1
				var frame_id: String = "D8_%04d" % idx
				_hide_all_entities()
				if prop:
					prop.visible = true
					prop.global_position = corridor_anchor
					_position_camera_spherical(corridor_anchor, dist, 0.0, yaw_off, 0.85)

				await _capture_and_record_frame(
					abs_img_dir, frame_id, "D8_Hard_Negatives", "corridor_empty",
					"fluorescent_normal", [], []
				)


# ==============================================================================
# HÀM BẮN RAYCAST, CHIẾU OBB & ĐỒNG BỘ ENGINE
# ==============================================================================

func _capture_and_record_frame(
	abs_img_dir: String,
	frame_id: String,
	slice_name: String,
	zone_name: String,
	lighting_name: String,
	active_actors: Array,
	subcategories: Array
) -> void:
	if robot_camera:
		robot_camera.force_update_transform()
	for actor in active_actors:
		if actor:
			_force_update_subtree_transforms(actor)

	await get_tree().physics_frame
	await get_tree().process_frame
	await RenderingServer.frame_post_draw

	var rel_img_path: String = "images/%s.jpg" % frame_id
	var full_img_path: String = abs_img_dir.path_join("%s.jpg" % frame_id)
	var img: Image = get_viewport().get_texture().get_image()
	img.save_jpg(full_img_path, 0.95)

	var gt_list: Array[Dictionary] = []
	var vp_size := Vector2(capture_resolution.x, capture_resolution.y)

	for i in range(active_actors.size()):
		var actor: Node3D = active_actors[i]
		if actor == null:
			continue
		var subcat: String = subcategories[i]
		var gt_box: Dictionary = _compute_actor_ground_truth(actor, subcat, vp_size)
		if gt_box.get("valid", false):
			gt_box.erase("valid")
			gt_list.append(gt_box)

	var record: Dictionary = {
		"frame_id": frame_id,
		"slice": slice_name,
		"image_path": rel_img_path,
		"resolution": [capture_resolution.x, capture_resolution.y],
		"context_5w1h": {
			"what_slice": slice_name,
			"where_zone": zone_name,
			"when_lighting": lighting_name,
			"how_camera_fov_deg": snappedf(robot_camera.fov if robot_camera else 75.0, 0.1),
			"how_camera_pos": [
				snappedf(robot_camera.global_position.x if robot_camera else 0.0, 0.01),
				snappedf(robot_camera.global_position.y if robot_camera else 1.15, 0.01),
				snappedf(robot_camera.global_position.z if robot_camera else 0.0, 0.01)
			],
			"seed": random_seed
		},
		"ground_truths": gt_list
	}

	_annotations_file.store_line(JSON.stringify(record))
	_total_saved_frames += 1

	if _total_saved_frames % 100 == 0:
		print("[DatasetGenerator] Tiến độ: ", _total_saved_frames, " / 1,200 frames...")


func _compute_actor_ground_truth(actor_root: Node3D, subcategory: String, vp_size: Vector2) -> Dictionary:
	var world_pts := PackedVector3Array()
	var h: float = 1.75
	var r: float = 0.28

	if subcategory == "wheelchair":
		h = 1.30
		r = 0.35
		for y_norm in [0.05, 0.20, 0.40, 0.60, 0.80, 0.98]:
			var y_val: float = h * y_norm
			for theta in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
				var rad = deg_to_rad(theta)
				var cur_r = r if y_val < 0.6 else r * 0.8
				var lp = Vector3(cur_r * cos(rad), y_val, cur_r * 0.7 * sin(rad))
				world_pts.append(actor_root.global_transform * lp)
	elif subcategory == "recumbent_bed":
		# Nằm ngang trên giường bệnh (dài 1.75m theo phương Z, rộng 0.5m theo phương X, cao 0.35m theo Y)
		for z_norm in [0.05, 0.25, 0.50, 0.75, 0.95]:
			var z_val: float = (z_norm - 0.5) * 1.75
			for x_norm in [-0.22, 0.0, 0.22]:
				for y_norm in [0.1, 0.35]:
					var lp = Vector3(x_norm, y_norm, z_val)
					world_pts.append(actor_root.global_transform * lp)
	elif subcategory == "elderly_cane":
		h = 1.55
		r = 0.28
		for y_norm in [0.05, 0.20, 0.40, 0.60, 0.80, 0.98]:
			var y_val: float = h * y_norm
			for theta in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
				var rad = deg_to_rad(theta)
				var lp = Vector3(r * cos(rad), y_val, r * 0.65 * sin(rad))
				world_pts.append(actor_root.global_transform * lp)
	else:
		for y_norm in [0.02, 0.15, 0.35, 0.55, 0.75, 0.90, 0.98]:
			var y_val: float = h * y_norm
			for theta in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
				var rad = deg_to_rad(theta)
				var cur_r = r if y_val > 1.0 else r * 0.75
				var lp = Vector3(cur_r * cos(rad), y_val, cur_r * 0.65 * sin(rad))
				world_pts.append(actor_root.global_transform * lp)

	if world_pts.is_empty():
		return {"valid": false}

	var min_x: float = vp_size.x
	var min_y: float = vp_size.y
	var max_x: float = 0.0
	var max_y: float = 0.0
	var front_pts: int = 0

	for pt in world_pts:
		if robot_camera and not robot_camera.is_position_behind(pt):
			var uv: Vector2 = robot_camera.unproject_position(pt)
			min_x = minf(min_x, uv.x)
			min_y = minf(min_y, uv.y)
			max_x = maxf(max_x, uv.x)
			max_y = maxf(max_y, uv.y)
			front_pts += 1

	if front_pts < 4:
		return {"valid": false}

	min_x = clampf(min_x, 0.0, vp_size.x)
	min_y = clampf(min_y, 0.0, vp_size.y)
	max_x = clampf(max_x, 0.0, vp_size.x)
	max_y = clampf(max_y, 0.0, vp_size.y)

	var bw: float = max_x - min_x
	var bh: float = max_y - min_y
	if bw < 8.0 or bh < 8.0:
		return {"valid": false}

	var space_state: PhysicsDirectSpaceState3D = get_world_3d().direct_space_state
	var cam_orig: Vector3 = robot_camera.global_position if robot_camera else Vector3.ZERO
	var vis_hits: int = 0
	var tested_rays: int = 0
	var stride: int = maxi(1, world_pts.size() / 15)

	var colliders: Array[CollisionObject3D] = []
	var body_meshes: Array[MeshInstance3D] = []
	_collect_actor_components(actor_root, body_meshes, colliders)

	for i in range(0, world_pts.size(), stride):
		var target_p: Vector3 = world_pts[i]
		if robot_camera and robot_camera.is_position_behind(target_p):
			continue
		tested_rays += 1
		var q := PhysicsRayQueryParameters3D.create(cam_orig, target_p)
		var hit: Dictionary = space_state.intersect_ray(q)
		if hit.is_empty() or (hit.get("collider") in colliders):
			vis_hits += 1

	var v_vis: float = float(vis_hits) / float(maxi(1, tested_rays))
	var dist_m: float = cam_orig.distance_to(actor_root.global_position + Vector3(0.0, 0.9, 0.0))
	var cx_gt: float = 0.5 * (min_x + max_x)
	var fov_val: float = robot_camera.fov if robot_camera else 75.0
	var bearing_deg: float = ((cx_gt - 0.5 * vp_size.x) / vp_size.x) * fov_val

	return {
		"valid": v_vis >= 0.15,
		"ignore_eval": v_vis < 0.25,
		"npc_id": actor_root.name,
		"subcategory": subcategory,
		"bbox_xyxy": [snappedf(min_x, 0.1), snappedf(min_y, 0.1), snappedf(max_x, 0.1), snappedf(max_y, 0.1)],
		"aspect_ratio": snappedf(bw / bh, 0.01),
		"visibility_ratio": snappedf(v_vis, 0.01),
		"distance_m": snappedf(dist_m, 0.01),
		"gt_bearing_deg": snappedf(bearing_deg, 0.01)
	}


func _stage_single_actor(
	actor: Node3D,
	anchor: Vector3,
	dist: float,
	cam_azimuth_deg: float,
	npc_yaw_deg: float,
	cam_yaw_offset_deg: float,
	target_look_height: float
) -> void:
	actor.global_position = anchor
	actor.rotation_degrees = Vector3(0.0, npc_yaw_deg, 0.0)
	_position_camera_spherical(anchor, dist, cam_azimuth_deg, cam_yaw_offset_deg, target_look_height)


func _position_camera_spherical(
	anchor: Vector3,
	dist: float,
	azimuth_deg: float,
	yaw_offset_deg: float,
	look_height: float
) -> void:
	if robot_camera == null:
		return
	var az_rad: float = deg_to_rad(azimuth_deg)
	var cam_pos := Vector3(
		anchor.x + dist * sin(az_rad),
		anchor.y + camera_height_m,
		anchor.z + dist * cos(az_rad)
	)
	var look_target := Vector3(anchor.x, anchor.y + look_height, anchor.z)
	robot_camera.look_at_from_position(cam_pos, look_target, Vector3.UP)
	robot_camera.rotate_object_local(Vector3.UP, deg_to_rad(yaw_offset_deg))


func _apply_lighting_profile(profile_name: String) -> void:
	if world_env == null or world_env.environment == null:
		return
	var env: Environment = world_env.environment
	match profile_name:
		"fluorescent_normal":
			env.tonemap_exposure = 1.0
			env.glow_enabled = false
			if main_dir_light:
				main_dir_light.light_energy = 1.0
				main_dir_light.rotation_degrees = Vector3(-50.0, 30.0, 0.0)
		"backlit_glare":
			env.tonemap_exposure = 1.35
			env.glow_enabled = true
			env.glow_intensity = 1.2
			if main_dir_light:
				main_dir_light.light_energy = 3.2
				main_dir_light.rotation_degrees = Vector3(-15.0, 0.0, 0.0)
		"dim_emergency_light":
			env.tonemap_exposure = 0.42
			env.glow_enabled = false
			if main_dir_light:
				main_dir_light.light_energy = 0.15
		"overexposed_fluorescent":
			env.tonemap_exposure = 1.85
			env.glow_enabled = true
			env.glow_intensity = 0.8
			if main_dir_light:
				main_dir_light.light_energy = 2.0


func _collect_actor_components(
	node: Node,
	out_meshes: Array[MeshInstance3D],
	out_colliders: Array[CollisionObject3D]
) -> void:
	if node is MeshInstance3D:
		if node.is_in_group("gt_body_mesh") or not _actor_has_any_gt_group(node.owner if node.owner else node):
			out_meshes.append(node)
	elif node is CollisionObject3D:
		out_colliders.append(node)

	for child in node.get_children():
		_collect_actor_components(child, out_meshes, out_colliders)


func _actor_has_any_gt_group(root: Node) -> bool:
	if root.is_in_group("gt_body_mesh"):
		return true
	for c in root.find_children("*", "MeshInstance3D", true, false):
		if c.is_in_group("gt_body_mesh"):
			return true
	return false


func _force_update_subtree_transforms(node: Node) -> void:
	if node is Node3D:
		node.force_update_transform()
	for child in node.get_children():
		_force_update_subtree_transforms(child)


func _hide_all_entities() -> void:
	if actors_pool:
		for child in actors_pool.get_children():
			if child is Node3D:
				child.visible = false
				child.global_position = Vector3(0.0, -50.0, 0.0)
	if obstacle_props:
		for child in obstacle_props.get_children():
			if child is Node3D:
				child.visible = false
				child.global_position = Vector3(0.0, -50.0, 0.0)
	if hard_negatives_pool:
		for child in hard_negatives_pool.get_children():
			if child is Node3D:
				child.visible = false
				child.global_position = Vector3(0.0, -50.0, 0.0)


func _get_actor(actor_name: String) -> Node3D:
	if actors_pool == null:
		return null
	var node: Node3D = actors_pool.get_node_or_null(actor_name)
	if node == null and actors_pool.get_child_count() > 0:
		node = actors_pool.get_child(0) as Node3D
	return node


func _get_prop(pool: Node3D, prop_name: String) -> Node3D:
	if pool == null:
		return null
	var node: Node3D = pool.get_node_or_null(prop_name)
	if node == null and pool.get_child_count() > 0:
		node = pool.get_child(0) as Node3D
	return node


func _write_5w1h_manifest(abs_out_dir: String, elapsed_sec: float) -> void:
	var manifest_path: String = abs_out_dir.path_join("manifest_5w1h.json")
	var f := FileAccess.open(manifest_path, FileAccess.WRITE)
	if f == null:
		return
	var manifest: Dictionary = {
		"who_generator": "SceneDatasetGenerator (Godot 4 Engine)",
		"what_deliverable": "Comprehensive 48-Cases 1,200-Frame Hospital Robot Vision Benchmark",
		"when_timestamp_utc": Time.get_datetime_string_from_system(true),
		"where_environment": "sim_vision/godot (Corridor, Doorways, Reception Counter, Emergency Beds)",
		"why_objective": "Rigorous stress-testing of yolov8n vs intel/person vs yolo11n vs yolo26n on occlusions, clinical poses, and extreme hospital conditions",
		"how_metrics": {
			"total_frames": _total_saved_frames,
			"resolution": [capture_resolution.x, capture_resolution.y],
			"seed": random_seed,
			"generation_time_sec": snappedf(elapsed_sec, 0.01),
			"dimension_counts": {
				"D1_Architecture": 150,
				"D2_Furniture": 150,
				"D3_Clinical_Poses": 150,
				"D4_Crowd_Dyn": 150,
				"D5_Distances": 150,
				"D6_Kinematics": 150,
				"D7_Lighting": 150,
				"D8_Hard_Negatives": 150
			}
		}
	}
	f.store_string(JSON.stringify(manifest, "\t"))
	f.close()
