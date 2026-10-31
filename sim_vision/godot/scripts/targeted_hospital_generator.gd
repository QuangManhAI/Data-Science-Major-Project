# targeted_hospital_generator.gd
## BỘ SINH DỮ LIỆU MỤC TIÊU (TARGETED DATASET GENERATOR - GODOT 4)
## Tập trung cao độ vào 3 lớp yếu: iv_stand (150 frames), water_dispenser (120 frames), patient (130 frames)
## Tạo góc nhìn robot đa dạng để tối ưu nhận diện cho mô hình YOLO26n Nano siêu nhỏ gọn
extends Node3D

@export var robot_camera: Camera3D
@export var room_node: Node3D
@export var capture_resolution: Vector2i = Vector2i(640, 480)
@export var output_root_dir: String = "res://../../data/vision/dataset/hospital_targeted_staging"
@export var target_total_frames: int = 400

const UNIFIED_CLASS_MAP = {
	"patient": 0,
	"doctor": 1,
	"wheelchair": 2,
	"queue_kiosk": 3,
	"reception_counter": 4,
	"sanitizer_stand": 5,
	"pccc_station": 6,
	"medical_waste_bin": 7,
	"water_dispenser": 8,
	"wc_station": 9,
	"elevator_doors": 10,
	"hospital_bed": 11,
	"stretcher_cart": 12,
	"iv_stand": 13,
	"medical_trolley": 14,
	"vital_monitor": 15,
	"hospital_chair": 16
}

const CLASS_NAMES = [
	"patient", "doctor", "wheelchair", "queue_kiosk", "reception_counter",
	"sanitizer_stand", "pccc_station", "medical_waste_bin", "water_dispenser",
	"wc_station", "elevator_doors", "hospital_bed", "stretcher_cart",
	"iv_stand", "medical_trolley", "vital_monitor", "hospital_chair"
]

var prop_catalog: Array[Dictionary] = []
var _total_saved: int = 0
var _class_box_counts: Dictionary = {}

func _ready() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0

	var cmd_args: PackedStringArray = OS.get_cmdline_args() + OS.get_cmdline_user_args()
	for i in range(cmd_args.size()):
		if cmd_args[i] == "--output-dir" and i + 1 < cmd_args.size():
			output_root_dir = cmd_args[i + 1]
		if cmd_args[i] == "--total-frames" and i + 1 < cmd_args.size():
			target_total_frames = int(cmd_args[i + 1])

	for c in CLASS_NAMES:
		_class_box_counts[c] = 0

	call_deferred("_run_targeted_pipeline")

func _build_catalog() -> void:
	prop_catalog.clear()
	if room_node == null:
		push_error("Room node is null!")
		return

	var furn = room_node.find_child("Furniture", true, false)
	if furn == null:
		push_error("Furniture node not found in Room!")
		return

	var node_mappings = [
		{"class": "reception_counter", "names": ["ReceptionCounter", "PharmacyCounter"]},
		{"class": "queue_kiosk", "names": ["LobbyQueueKiosk"]},
		{"class": "sanitizer_stand", "names": ["LobbySanitizerStand", "CorridorSanitizerStand"]},
		{"class": "pccc_station", "names": ["CorridorPccc"]},
		{"class": "medical_waste_bin", "names": ["LobbyWasteBin", "CorridorWasteBinEast", "CorridorWasteBinWest", "EmergencyWasteBin", "LabWasteBin", "ClinicWasteBin", "PharmacyWasteBin", "WcWasteBin"]},
		{"class": "water_dispenser", "names": ["LobbyWaterDispenser"]},
		{"class": "wc_station", "names": ["RestroomStation"]},
		{"class": "elevator_doors", "names": ["HospitalElevator"]},
		{"class": "hospital_bed", "names": ["EmergencyBed1", "EmergencyBed2", "ClinicExamBed"]},
		{"class": "stretcher_cart", "names": ["EmergencyStretcherCart"]},
		{"class": "iv_stand", "names": ["EmergencyIVStand", "ClinicIVStand"]},
		{"class": "medical_trolley", "names": ["EmergencyTrolley1", "EmergencyTrolley2", "ClinicTrolley", "LabTrolley"]},
		{"class": "vital_monitor", "names": ["EmergencyVitalMonitor", "ClinicVitalMonitor"]},
		{"class": "wheelchair", "names": ["LobbyWheelchair"]},
		{"class": "hospital_chair", "names": ["WaitingBenchWest", "WaitingBenchEast", "WaitingBenchWestBack", "WaitingBenchEastBack", "CorridorBenchEast"]}
	]

	for mapping in node_mappings:
		var cname = mapping["class"]
		var cid = UNIFIED_CLASS_MAP[cname]
		for n_name in mapping["names"]:
			var n = furn.find_child(n_name, true, false) as Node3D
			if n:
				var local_aabb = _compute_node_aabb(n)
				prop_catalog.append({
					"class_name": cname,
					"class_id": cid,
					"node_name": n_name,
					"node": n,
					"local_aabb": local_aabb,
					"center_offset": local_aabb.get_center(),
					"look_height": maxf(0.5, local_aabb.size.y * 0.5)
				})

	# Lập chỉ mục thêm các NPC
	var npcs = furn.find_children("*", "Node3D", true, false)
	for npc in npcs:
		var n_name = npc.name
		if n_name.begins_with("MedicalStaff") or n_name.begins_with("Doctor"):
			var aabb = _compute_node_aabb(npc)
			prop_catalog.append({
				"class_name": "doctor",
				"class_id": UNIFIED_CLASS_MAP["doctor"],
				"node_name": n_name,
				"node": npc,
				"local_aabb": aabb,
				"center_offset": aabb.get_center(),
				"look_height": 1.1
			})
		elif n_name.begins_with("WaitingPatient") or n_name.begins_with("Patient"):
			var aabb = _compute_node_aabb(npc)
			prop_catalog.append({
				"class_name": "patient",
				"class_id": UNIFIED_CLASS_MAP["patient"],
				"node_name": n_name,
				"node": npc,
				"local_aabb": aabb,
				"center_offset": aabb.get_center(),
				"look_height": 0.9
			})

	print("[TargetedGenerator] Đã lập chỉ mục ", prop_catalog.size(), " đối tượng y tế & nhân vật.")

func _compute_node_aabb(root_node: Node3D) -> AABB:
	var total_aabb := AABB()
	var found := false

	var meshes = root_node.find_children("*", "MeshInstance3D", true, false)
	for m in meshes:
		var mi = m as MeshInstance3D
		if mi and mi.mesh:
			var m_aabb = mi.mesh.get_aabb()
			var rel_xform = root_node.global_transform.affine_inverse() * mi.global_transform
			var x_aabb = rel_xform * m_aabb
			if not found:
				total_aabb = x_aabb
				found = true
			else:
				total_aabb = total_aabb.merge(x_aabb)

	if not found:
		var shapes = root_node.find_children("*", "CollisionShape3D", true, false)
		for s in shapes:
			var cs = s as CollisionShape3D
			if cs and cs.shape:
				var sz = Vector3(0.5, 0.5, 0.5)
				if cs.shape is BoxShape3D:
					sz = (cs.shape as BoxShape3D).size
				var s_aabb = AABB(-sz * 0.5, sz)
				var rel_xform = root_node.global_transform.affine_inverse() * cs.global_transform
				var x_aabb = rel_xform * s_aabb
				if not found:
					total_aabb = x_aabb
					found = true
				else:
					total_aabb = total_aabb.merge(x_aabb)

	if not found:
		total_aabb = AABB(Vector3(-0.3, 0.0, -0.3), Vector3(0.6, 1.2, 0.6))

	return total_aabb

func _run_targeted_pipeline() -> void:
	var start_ms = Time.get_ticks_msec()
	_build_catalog()

	var abs_out_dir: String = ProjectSettings.globalize_path(output_root_dir)
	var abs_img_dir: String = abs_out_dir.path_join("images")
	var abs_lbl_dir: String = abs_out_dir.path_join("labels")
	DirAccess.make_dir_recursive_absolute(abs_img_dir)
	DirAccess.make_dir_recursive_absolute(abs_lbl_dir)

	print("==================================================================")
	print("🎯 BẮT ĐẦU SINH DỮ LIỆU TARGETED (MỤC TIÊU: ", target_total_frames, " FRAMES)")
	print("📂 Thư mục xuất dữ liệu: ", abs_out_dir)
	print("==================================================================")

	var vp: Viewport = get_viewport()
	vp.size = capture_resolution

	# --------------------------------------------------------------------------
	# NHÓM 1: CỌC TRUYỀN DỊCH (iv_stand - Ngân sách: 150 frames)
	# --------------------------------------------------------------------------
	print("\n>>> NHÓM 1: Chụp tăng cường cọc truyền dịch (EmergencyIVStand + ClinicIVStand)...")
	var iv_props = prop_catalog.filter(func(p): return p["class_name"] == "iv_stand")
	var iv_target_count = 150
	var iv_saved = 0

	var iv_dists = [0.8, 1.1, 1.5, 2.0, 2.6]
	var iv_yaws = [-135.0, -90.0, -45.0, -25.0, 0.0, 25.0, 45.0, 90.0, 135.0, 180.0]
	var iv_heights = [0.55, 0.85, 1.15, 1.45] # Đa dạng độ cao camera robot

	for prop in iv_props:
		var target_pos: Vector3 = prop["node"].global_position + prop["center_offset"]
		for d in iv_dists:
			for y in iv_yaws:
				for h in iv_heights:
					if iv_saved >= iv_target_count or _total_saved >= target_total_frames: break
					var fid = "TGT_%04d" % (_total_saved + 1)
					_position_camera_around(target_pos, d, deg_to_rad(y), h, prop["look_height"])
					await _capture_and_label(abs_img_dir, abs_lbl_dir, fid, "Target_IVStand")
					iv_saved += 1
				if iv_saved >= iv_target_count or _total_saved >= target_total_frames: break
			if iv_saved >= iv_target_count or _total_saved >= target_total_frames: break
		if iv_saved >= iv_target_count or _total_saved >= target_total_frames: break

	# --------------------------------------------------------------------------
	# NHÓM 2: CÂY NƯỚC UỐNG (water_dispenser - Ngân sách: 120 frames)
	# --------------------------------------------------------------------------
	print("\n>>> NHÓM 2: Chụp tăng cường cây nước uống (LobbyWaterDispenser)...")
	var wd_props = prop_catalog.filter(func(p): return p["class_name"] == "water_dispenser")
	var wd_target_count = 120
	var wd_saved = 0

	var wd_dists = [1.2, 1.6, 2.2, 3.0, 3.8, 4.6]
	var wd_yaws = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 90.0]
	var wd_heights = [0.65, 0.95, 1.25]

	for prop in wd_props:
		var target_pos: Vector3 = prop["node"].global_position + prop["center_offset"]
		for d in wd_dists:
			for y in wd_yaws:
				for h in wd_heights:
					if wd_saved >= wd_target_count or _total_saved >= target_total_frames: break
					var fid = "TGT_%04d" % (_total_saved + 1)
					_position_camera_around(target_pos, d, deg_to_rad(y), h, prop["look_height"])
					await _capture_and_label(abs_img_dir, abs_lbl_dir, fid, "Target_WaterDispenser")
					wd_saved += 1
				if wd_saved >= wd_target_count or _total_saved >= target_total_frames: break
			if wd_saved >= wd_target_count or _total_saved >= target_total_frames: break
		if wd_saved >= wd_target_count or _total_saved >= target_total_frames: break

	# --------------------------------------------------------------------------
	# NHÓM 3: BỆNH NHÂN (patient - Ngân sách: 130 frames)
	# --------------------------------------------------------------------------
	print("\n>>> NHÓM 3: Chụp tăng cường bệnh nhân (WaitingPatients & Clinic/Emergency)...")
	var patient_props = prop_catalog.filter(func(p): return p["class_name"] == "patient")
	var patient_target_count = 130
	var patient_saved = 0

	var pt_dists = [1.0, 1.4, 2.0, 2.8]
	var pt_yaws = [-45.0, -20.0, 0.0, 20.0, 45.0, 90.0, 180.0]
	var pt_heights = [0.70, 1.05, 1.35]

	for prop in patient_props:
		var target_pos: Vector3 = prop["node"].global_position + prop["center_offset"]
		for d in pt_dists:
			for y in pt_yaws:
				for h in pt_heights:
					if patient_saved >= patient_target_count or _total_saved >= target_total_frames: break
					var fid = "TGT_%04d" % (_total_saved + 1)
					_position_camera_around(target_pos, d, deg_to_rad(y), h, prop["look_height"])
					await _capture_and_label(abs_img_dir, abs_lbl_dir, fid, "Target_Patient")
					patient_saved += 1
				if patient_saved >= patient_target_count or _total_saved >= target_total_frames: break
			if patient_saved >= patient_target_count or _total_saved >= target_total_frames: break
		if patient_saved >= patient_target_count or _total_saved >= target_total_frames: break

	# Manifest tổng kết
	var elapsed_sec = (Time.get_ticks_msec() - start_ms) / 1000.0
	_write_dataset_manifest(abs_out_dir, elapsed_sec)

	print("==================================================================")
	print("✅ HOÀN TẤT SINH BỘ DỮ LIỆU TARGETED THÀNH CÔNG!")
	print("📊 Tổng frames: ", _total_saved)
	print("📊 Thống kê hộp nhãn theo 17 classes: ", _class_box_counts)
	print("⏱️ Thời gian thực thi: ", elapsed_sec, " giây.")
	print("==================================================================")
	get_tree().quit(0)

func _position_camera_around(target: Vector3, dist: float, yaw_rad: float, cam_h: float, look_h: float) -> void:
	var cam_x = target.x + dist * sin(yaw_rad)
	var cam_z = target.z + dist * cos(yaw_rad)

	cam_x = clampf(cam_x, -11.5, 11.5)
	cam_z = clampf(cam_z, -8.5, 8.5)
	cam_h = clampf(cam_h, 0.45, 1.8)

	robot_camera.global_position = Vector3(cam_x, cam_h, cam_z)
	robot_camera.look_at(Vector3(target.x, target.y + look_h, target.z), Vector3.UP)

func _capture_and_label(abs_img_dir: String, abs_lbl_dir: String, frame_id: String, scenario_tag: String) -> void:
	await get_tree().physics_frame
	await get_tree().process_frame
	if DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw

	var full_img_path = abs_img_dir.path_join("%s.jpg" % frame_id)
	var img: Image = get_viewport().get_texture().get_image()
	if img:
		img.save_jpg(full_img_path, 0.92)

	var yolo_lines: PackedStringArray = []
	var detected_props: Array[String] = []
	var vp_w = float(capture_resolution.x)
	var vp_h = float(capture_resolution.y)

	var space_state = get_world_3d().direct_space_state

	for prop in prop_catalog:
		var node = prop["node"] as Node3D
		if node == null or not node.is_inside_tree() or not node.visible: continue

		var aabb: AABB = prop["local_aabb"]
		var local_pts: Array[Vector3] = [
			Vector3(aabb.position.x, aabb.position.y, aabb.position.z),
			Vector3(aabb.end.x, aabb.position.y, aabb.position.z),
			Vector3(aabb.position.x, aabb.end.y, aabb.position.z),
			Vector3(aabb.end.x, aabb.end.y, aabb.position.z),
			Vector3(aabb.position.x, aabb.end.y, aabb.end.z),
			Vector3(aabb.end.x, aabb.end.y, aabb.end.z),
			Vector3(aabb.position.x, aabb.position.y, aabb.end.z),
			Vector3(aabb.end.x, aabb.position.y, aabb.end.z),
			aabb.get_center()
		]

		var visible_screens: Array[Vector2] = []
		var front_count = 0

		for lp in local_pts:
			var wp = node.global_transform * lp
			if robot_camera.is_position_behind(wp):
				continue
			front_count += 1
			var uv = robot_camera.unproject_position(wp)
			if uv.x >= -30.0 and uv.x <= vp_w + 30.0 and uv.y >= -30.0 and uv.y <= vp_h + 30.0:
				var q = PhysicsRayQueryParameters3D.create(robot_camera.global_position, wp)
				var hit = space_state.intersect_ray(q)
				if hit.is_empty() or hit.get("collider") == node or node.is_ancestor_of(hit.get("collider")):
					visible_screens.append(uv)

		if visible_screens.size() >= 3 and front_count >= 3:
			var min_x = vp_w
			var max_x = 0.0
			var min_y = vp_h
			var max_y = 0.0

			for p in visible_screens:
				min_x = minf(min_x, p.x)
				max_x = maxf(max_x, p.x)
				min_y = minf(min_y, p.y)
				max_y = maxf(max_y, p.y)

			min_x = clampf(min_x, 0.0, vp_w)
			max_x = clampf(max_x, 0.0, vp_w)
			min_y = clampf(min_y, 0.0, vp_h)
			max_y = clampf(max_y, 0.0, vp_h)

			var bw = max_x - min_x
			var bh = max_y - min_y

			if bw >= 8.0 and bh >= 8.0:
				var cx = (min_x + max_x) * 0.5 / vp_w
				var cy = (min_y + max_y) * 0.5 / vp_h
				var norm_w = bw / vp_w
				var norm_h = bh / vp_h
				var cid = prop["class_id"]
				var cname = prop["class_name"]

				yolo_lines.append("%d %.6f %.6f %.6f %.6f" % [cid, cx, cy, norm_w, norm_h])
				_class_box_counts[cname] += 1
				detected_props.append(cname)

	var full_lbl_path = abs_lbl_dir.path_join("%s.txt" % frame_id)
	var lbl_file = FileAccess.open(full_lbl_path, FileAccess.WRITE)
	if lbl_file:
		if not yolo_lines.is_empty():
			lbl_file.store_string("\n".join(yolo_lines) + "\n")
		lbl_file.close()

	_total_saved += 1
	if _total_saved % 50 == 0:
		print("[TargetedGenerator] Đã tạo ", _total_saved, "/", target_total_frames, " frames (Gần nhất: ", scenario_tag, " -> ", detected_props, ")")

func _write_dataset_manifest(abs_out_dir: String, elapsed_sec: float) -> void:
	var total_boxes = 0
	for c in CLASS_NAMES:
		total_boxes += _class_box_counts[c]

	var manifest = {
		"dataset_name": "hospital_targeted_staging",
		"total_frames": _total_saved,
		"total_boxes": total_boxes,
		"generation_time_sec": elapsed_sec,
		"target_classes": ["iv_stand", "water_dispenser", "patient"],
		"class_counts": _class_box_counts
	}

	var mf = FileAccess.open(abs_out_dir.path_join("manifest.json"), FileAccess.WRITE)
	if mf:
		mf.store_string(JSON.stringify(manifest, "\t"))
		mf.close()
