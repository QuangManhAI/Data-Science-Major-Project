# props_dataset_generator.gd
## BỘ SINH DỮ LIỆU TRANG THIẾT BỊ BỆNH VIỆN TỰ ĐỘNG (1,600 FRAMES - 13 CLASSES)
## Tự động trích xuất Bounding Box từ 3D Mesh / Collision AABB & Chiếu camera 2D
extends Node3D

@export var robot_camera: Camera3D
@export var room_node: Node3D
@export var capture_resolution: Vector2i = Vector2i(640, 480)
@export var output_root_dir: String = "res://../../data/vision/dataset/hospital_props_1600_15class"
@export var target_total_frames: int = 1600

const CLASS_MAP = {
	"queue_kiosk": 0,
	"reception_counter": 1,
	"sanitizer_stand": 2,
	"pccc_station": 3,
	"medical_waste_bin": 4,
	"water_dispenser": 5,
	"wc_station": 6,
	"elevator_doors": 7,
	"hospital_bed": 8,
	"stretcher_cart": 9,
	"iv_stand": 10,
	"medical_trolley": 11,
	"vital_monitor": 12
}

const CLASS_NAMES = [
	"queue_kiosk", "reception_counter", "sanitizer_stand", "pccc_station",
	"medical_waste_bin", "water_dispenser", "wc_station", "elevator_doors",
	"hospital_bed", "stretcher_cart", "iv_stand", "medical_trolley", "vital_monitor"
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

	call_deferred("_run_props_pipeline")

func _clean_npcs() -> void:
	if room_node == null: return
	var furn = room_node.find_child("Furniture", true, false)
	if furn:
		for child in furn.get_children():
			var cname = child.name
			if cname.begins_with("MedicalStaff") or cname.begins_with("WaitingPatient") or (child is HospitalNpc):
				child.visible = false
				child.set_physics_process(false)
				child.set_process(false)
				child.global_position = Vector3(0, -9999, 0)
	print("[PropsGenerator] Đã làm sạch các NPC vãng lai khỏi scene bệnh viện.")

func _build_prop_catalog() -> void:
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
		{"class": "vital_monitor", "names": ["EmergencyVitalMonitor", "ClinicVitalMonitor"]}
	]

	for mapping in node_mappings:
		var cname = mapping["class"]
		var cid = CLASS_MAP[cname]
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

	print("[PropsGenerator] Đã lập chỉ mục ", prop_catalog.size(), " đối tượng trang thiết bị y tế trong phòng bệnh.")

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

func _run_props_pipeline() -> void:
	var start_ms = Time.get_ticks_msec()
	_clean_npcs()
	_build_prop_catalog()

	var abs_out_dir: String = ProjectSettings.globalize_path(output_root_dir)
	var abs_img_dir: String = abs_out_dir.path_join("images")
	var abs_lbl_dir: String = abs_out_dir.path_join("labels")
	DirAccess.make_dir_recursive_absolute(abs_img_dir)
	DirAccess.make_dir_recursive_absolute(abs_lbl_dir)

	print("==================================================================")
	print("🏥 BẮT ĐẦU SINH BỘ DATASET HOSPITAL PROPS & ITEMS (1,600 FRAMES / 13 CLASSES)")
	print("📂 Thư mục xuất dữ liệu: ", abs_out_dir)
	print("==================================================================")

	var vp: Viewport = get_viewport()
	vp.size = capture_resolution

	# 1. PHA 1: Chụp tập trung từng Props (Solo Inward Views: 1,160 frames)
	print("\n>>> PHA 1: Thu thập góc chụp chi tiết từng thiết bị (Inward Solo Views)...")
	var dists = [1.2, 1.8, 2.6, 3.6]
	var yaws = [-30.0, -15.0, 0.0, 15.0, 30.0]
	var heights = [0.85, 1.25]

	for prop in prop_catalog:
		var target_pos: Vector3 = prop["node"].global_position + prop["center_offset"]
		# Vector hướng từ đồ vật vào tâm phòng (0, y, 0)
		var center_pt = Vector3(0.0, target_pos.y, 0.0)
		if absf(target_pos.x) < 0.5 and absf(target_pos.z) < 0.5:
			center_pt = Vector3(0.0, target_pos.y, 1.0)
		var inward_dir = (center_pt - target_pos).normalized()
		var base_yaw = atan2(inward_dir.x, inward_dir.z)

		for d in dists:
			for y in yaws:
				for h in heights:
					if _total_saved >= 1160 or _total_saved >= target_total_frames: break
					var fid = "P_%04d" % (_total_saved + 1)
					_position_camera_inward(target_pos, d, base_yaw, y, h)
					await _capture_and_label(abs_img_dir, abs_lbl_dir, fid, "Solo_" + prop["class_name"])
				if _total_saved >= 1160 or _total_saved >= target_total_frames: break
			if _total_saved >= 1160 or _total_saved >= target_total_frames: break
		if _total_saved >= 1160 or _total_saved >= target_total_frames: break

	# 2. PHA 2: Chụp tổng quan cụm phòng (Multi-Object Views: 320 frames)
	print("\n>>> PHA 2: Thu thập cụm thiết bị theo khu vực (Multi-Object Overviews)...")
	var room_clusters = [
		{"name": "Lobby_Main", "center": Vector3(0.0, 0.9, 4.5), "look": Vector3(0.0, 0.8, 7.8)},
		{"name": "Lobby_West", "center": Vector3(-2.0, 1.1, 4.8), "look": Vector3(1.5, 0.8, 7.5)},
		{"name": "Lobby_East", "center": Vector3(2.0, 1.1, 4.8), "look": Vector3(-1.5, 0.8, 7.5)},
		{"name": "Emergency_Bay1", "center": Vector3(-7.5, 1.1, -4.5), "look": Vector3(-9.5, 0.8, -7.2)},
		{"name": "Emergency_Bay2", "center": Vector3(-4.0, 1.1, -4.5), "look": Vector3(-5.5, 0.8, -7.2)},
		{"name": "Corridor_East", "center": Vector3(1.5, 0.9, 0.0), "look": Vector3(5.5, 0.8, -1.5)},
		{"name": "Corridor_West", "center": Vector3(-1.5, 0.9, 0.0), "look": Vector3(-5.5, 0.8, -1.5)},
		{"name": "Clinic_Room", "center": Vector3(8.5, 1.1, -3.5), "look": Vector3(11.0, 0.8, -4.5)},
		{"name": "Pharmacy_Counter", "center": Vector3(-6.5, 1.0, 5.5), "look": Vector3(-8.8, 0.8, 7.5)},
		{"name": "Elevator_Lobby", "center": Vector3(8.5, 1.1, 3.5), "look": Vector3(11.8, 1.2, 4.6)},
		{"name": "Restroom_Approach", "center": Vector3(6.5, 1.0, 6.5), "look": Vector3(9.2, 1.1, 8.6)}
	]

	var step_cluster = 0
	while _total_saved < 1480 and _total_saved < target_total_frames:
		var cluster = room_clusters[step_cluster % room_clusters.size()]
		var angle_rad = deg_to_rad((step_cluster * 15.0))
		var offset_r = 0.8 + (step_cluster % 4) * 0.35
		var cam_pos = cluster["center"] + Vector3(sin(angle_rad) * offset_r, (step_cluster % 2) * 0.3, cos(angle_rad) * offset_r)
		cam_pos.x = clampf(cam_pos.x, -11.5, 11.5)
		cam_pos.z = clampf(cam_pos.z, -8.5, 8.5)
		cam_pos.y = clampf(cam_pos.y, 0.5, 1.6)

		robot_camera.global_position = cam_pos
		robot_camera.look_at(cluster["look"], Vector3.UP)

		var fid = "P_%04d" % (_total_saved + 1)
		await _capture_and_label(abs_img_dir, abs_lbl_dir, fid, "Cluster_" + cluster["name"])
		step_cluster += 1

	# 3. PHA 3: Mẫu âm tính hành lang / vách tường trống (Hard Negatives: 120 frames)
	print("\n>>> PHA 3: Thu thập mẫu âm tính tường/sàn trống (Hard Negatives)...")
	var empty_spots = [
		{"pos": Vector3(0.0, 0.9, 0.0), "look": Vector3(0.0, 0.9, -1.8)}, # Cửa trượt buồng bệnh đóng
		{"pos": Vector3(0.0, 1.2, 0.0), "look": Vector3(0.0, 2.8, 0.0)},  # Trần nhà & bóng đèn huỳnh quang
		{"pos": Vector3(2.0, 0.8, 0.0), "look": Vector3(2.0, 0.0, 1.5)},  # Sàn nhà gạch men trống
		{"pos": Vector3(-1.0, 1.0, 0.0), "look": Vector3(-1.0, 1.0, 1.8)} # Vách tường trống
	]

	for i in range(empty_spots.size()):
		var espot = empty_spots[i]
		for s in range(30):
			if _total_saved >= target_total_frames: break
			var yaw_rad = deg_to_rad((s - 15) * 3.0)
			robot_camera.global_position = espot["pos"] + Vector3(sin(yaw_rad) * 0.3, (s % 2) * 0.2, cos(yaw_rad) * 0.3)
			robot_camera.look_at(espot["look"], Vector3.UP)

			var fid = "P_%04d" % (_total_saved + 1)
			await _capture_and_label(abs_img_dir, abs_lbl_dir, fid, "HardNeg_EmptySurface")

	# 4. Ghi manifest tổng kết
	_write_dataset_manifest(abs_out_dir, (Time.get_ticks_msec() - start_ms) / 1000.0)

	print("==================================================================")
	print("✅ HOÀN TẤT SINH BỘ DATASET THÀNH CÔNG!")
	print("📊 Tổng frames: ", _total_saved)
	print("📊 Thống kê hộp nhãn theo 13 classes: ", _class_box_counts)
	print("⏱️ Thời gian thực thi: ", (Time.get_ticks_msec() - start_ms) / 1000.0, " giây.")
	print("==================================================================")
	get_tree().quit(0)

func _position_camera_inward(target: Vector3, dist: float, base_yaw: float, offset_deg: float, cam_h: float) -> void:
	var final_yaw = base_yaw + deg_to_rad(offset_deg)
	var cam_x = target.x + dist * sin(final_yaw)
	var cam_z = target.z + dist * cos(final_yaw)

	# Giữ camera luôn nằm bên trong phòng
	cam_x = clampf(cam_x, -11.5, 11.5)
	cam_z = clampf(cam_z, -8.5, 8.5)
	cam_h = clampf(cam_h, 0.45, 1.8)

	robot_camera.global_position = Vector3(cam_x, cam_h, cam_z)
	robot_camera.look_at(target + Vector3(0.0, 0.1, 0.0), Vector3.UP)

func _capture_and_label(abs_img_dir: String, abs_lbl_dir: String, frame_id: String, scenario_tag: String) -> void:
	await get_tree().physics_frame
	await get_tree().process_frame
	if DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw

	# 1. Lưu ảnh JPEG
	var full_img_path = abs_img_dir.path_join("%s.jpg" % frame_id)
	var img: Image = get_viewport().get_texture().get_image()
	img.save_jpg(full_img_path, 0.90)

	# 2. Quét toàn bộ 13 classes props xem có xuất hiện trong khung hình không
	var yolo_lines: PackedStringArray = []
	var detected_props: Array[String] = []
	var vp_w = float(capture_resolution.x)
	var vp_h = float(capture_resolution.y)

	var space_state = get_world_3d().direct_space_state

	for prop in prop_catalog:
		var node = prop["node"] as Node3D
		if node == null or not node.is_inside_tree(): continue

		var aabb: AABB = prop["local_aabb"]
		var local_pts: Array[Vector3] = [
			Vector3(aabb.position.x, aabb.position.y, aabb.position.z),
			Vector3(aabb.end.x, aabb.position.y, aabb.position.z),
			Vector3(aabb.position.x, aabb.end.y, aabb.position.z),
			Vector3(aabb.end.x, aabb.end.y, aabb.position.z),
			Vector3(aabb.position.x, aabb.position.y, aabb.end.z),
			Vector3(aabb.end.x, aabb.position.y, aabb.end.z),
			Vector3(aabb.position.x, aabb.end.y, aabb.end.z),
			Vector3(aabb.end.x, aabb.end.y, aabb.end.z),
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
				# Raycast check occlusion
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

			if bw >= 12.0 and bh >= 12.0:
				var cx = (min_x + max_x) * 0.5 / vp_w
				var cy = (min_y + max_y) * 0.5 / vp_h
				var norm_w = bw / vp_w
				var norm_h = bh / vp_h
				var cid = prop["class_id"]
				var cname = prop["class_name"]

				yolo_lines.append("%d %.6f %.6f %.6f %.6f" % [cid, cx, cy, norm_w, norm_h])
				_class_box_counts[cname] += 1
				detected_props.append(cname)

	# 3. Lưu nhãn YOLO
	var full_lbl_path = abs_lbl_dir.path_join("%s.txt" % frame_id)
	var lbl_file = FileAccess.open(full_lbl_path, FileAccess.WRITE)
	if lbl_file:
		if not yolo_lines.is_empty():
			lbl_file.store_string("\n".join(yolo_lines) + "\n")
		lbl_file.close()

	_total_saved += 1
	if _total_saved % 100 == 0:
		print("[PropsGenerator] Đã tạo ", _total_saved, "/", target_total_frames, " frames (Gần nhất: ", scenario_tag, " -> ", detected_props, ")")

func _write_dataset_manifest(abs_out_dir: String, elapsed_sec: float) -> void:
	var total_boxes = 0
	for c in CLASS_NAMES:
		total_boxes += _class_box_counts[c]

	var manifest = {
		"dataset_name": "hospital_props_and_items_13class",
		"version": "1.0_unified_flat",
		"created_at": Time.get_datetime_string_from_system(true),
		"total_frames": _total_saved,
		"total_instances": total_boxes,
		"generation_time_sec": snappedf(elapsed_sec, 0.1),
		"resolution": [capture_resolution.x, capture_resolution.y],
		"format": "YOLO_normalized_5col",
		"classes": CLASS_MAP,
		"class_counts": _class_box_counts,
		"notes": "Generated automatically via Godot 4 3D Physics Mesh Projection. Unified flat dataset."
	}

	var json_path = abs_out_dir.path_join("manifest.json")
	var f = FileAccess.open(json_path, FileAccess.WRITE)
	if f:
		f.store_string(JSON.stringify(manifest, "  "))
		f.close()
