# robot_patrol_test_generator.gd
## BỘ SINH DỮ LIỆU KIỂM THỬ HÀNH TRÌNH ROBOT TUẦN TRA THỰC TẾ (REAL ROBOT PATROL TESTBENCH)
## Robot di chuyển liên tục qua toàn bộ 7 phân khu của bệnh viện theo chuỗi Waypoints mượt mà.
## Camera gắn trên robot (h=1.18m, pitch=-5 deg, fov=70 deg) thu hình liên tục ở góc nhìn thực tế.
## Tự động tính toán Bounding Box 3D raycast cho toàn bộ 17 lớp đối tượng (bao gồm đạo cụ và NPC).
extends Node3D

@export var robot_camera: Camera3D
@export var room_node: Node3D
@export var capture_resolution: Vector2i = Vector2i(640, 480)
@export var output_root_dir: String = "res://../../data/vision/dataset/hospital_robot_patrol_testset"
@export var step_distance: float = 0.16 # ~0.16m mỗi frame -> ~450 frames cho toàn bộ 70m hành trình

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

# 28 Waypoints tuần tra khép kín 7 phân khu lâm sàng
const PATROL_WAYPOINTS: Array[Vector3] = [
	Vector3(0.0, 0.0, 4.0),    # WP0: Xuất phát Sảnh tiếp đón
	Vector3(2.5, 0.0, 5.5),    # WP1: Quầy tiếp đón & Kiosk
	Vector3(1.0, 0.0, 6.5),    # WP2: Phía trước Kiosk & Sanitizer
	Vector3(-2.5, 0.0, 6.2),   # WP3: Khu ghế chờ hàng sau
	Vector3(-4.5, 0.0, 5.0),   # WP4: Khu ghế chờ hàng trước & xe lăn sảnh
	Vector3(-6.0, 0.0, 3.5),   # WP5: Lối rẽ sang hành lang Tây
	Vector3(-8.8, 0.0, 2.5),   # WP6: Trước cửa Nhà Thuốc
	Vector3(-8.8, 0.0, 0.5),   # WP7: Hành lang Tây
	Vector3(-8.0, 0.0, -1.0),  # WP8: Tiếp cận cửa Khoa Cấp Cứu
	Vector3(-8.0, 0.0, -3.5),  # WP9: Vào trong Khoa Cấp Cứu
	Vector3(-10.0, 0.0, -4.5), # WP10: Tiếp cận Cáng đẩy & Giường 1
	Vector3(-8.5, 0.0, -6.8),  # WP11: Tiếp cận Cọc truyền dịch & Vital Monitor
	Vector3(-6.2, 0.0, -5.5),  # WP12: Giữa phòng Cấp Cứu
	Vector3(-6.0, 0.0, -2.5),  # WP13: Hướng ra cửa Cấp Cứu
	Vector3(-6.0, 0.0, -0.2),  # WP14: Trở lại Hành lang chính, rẽ Đông
	Vector3(-2.5, 0.0, 0.0),   # WP15: Hành lang trung tâm, trạm PCCC
	Vector3(0.0, 0.0, -1.0),   # WP16: Trước cửa Phòng Xét Nghiệm Máu
	Vector3(0.0, 0.0, -4.0),   # WP17: Vào trong Phòng Xét Nghiệm
	Vector3(0.0, 0.0, -1.0),   # WP18: Trở ra Hành lang chính
	Vector3(4.0, 0.0, 0.0),    # WP19: Hành lang Đông
	Vector3(8.0, 0.0, -1.0),   # WP20: Tiếp cận cửa Phòng Khám Nội
	Vector3(8.0, 0.0, -4.5),   # WP21: Vào trong Phòng Khám Nội
	Vector3(10.5, 0.0, -6.0),  # WP22: Cạnh Giường khám, Cọc truyền Clinic & Bác sĩ
	Vector3(8.0, 0.0, -2.5),   # WP23: Trở ra Hành lang
	Vector3(8.5, 0.0, 1.0),    # WP24: Rẽ Nam về khu Thang Máy
	Vector3(9.0, 0.0, 4.0),    # WP25: Trước Cửa Thang Máy & Cửa WC
	Vector3(9.5, 0.0, 6.2),    # WP26: Trạm Nhà Vệ Sinh (WC Station)
	Vector3(6.0, 0.0, 5.5),    # WP27: Cây nước uống (Water Dispenser)
	Vector3(2.5, 0.0, 4.5),    # WP28: Trở về Sảnh Chính
	Vector3(0.0, 0.0, 4.0)     # WP29: Khép vòng tại điểm xuất phát
]

var prop_catalog: Array[Dictionary] = []
var _total_saved: int = 0
var _class_box_counts: Dictionary = {}
var spawned_wheelchairs: Array[Node3D] = []

func _ready() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0

	for c in CLASS_NAMES:
		_class_box_counts[c] = 0

	call_deferred("_run_patrol_pipeline")

func _open_all_sliding_doors() -> void:
	if room_node == null: return
	var doors = room_node.find_children("*", "SlidingDoor", true, false)
	for d in doors:
		if d is SlidingDoor:
			d.force_open()
	print("[RobotPatrol] Đã mở toàn bộ ", doors.size(), " cửa trượt âm tường để robot tuần tra tự do.")

func _build_catalog() -> void:
	prop_catalog.clear()
	if room_node == null:
		push_error("Room node is null!")
		return

	var furn = room_node.find_child("Furniture", true, false)
	if furn == null:
		push_error("Furniture node not found in Room!")
		return

	# Lấy mẫu wheelchair gốc để thêm vào hành lang và phòng cấp cứu nếu chưa có
	var orig_wc = furn.find_child("LobbyWheelchair", true, false) as Node3D
	if orig_wc and spawned_wheelchairs.is_empty():
		var wc_corridor = orig_wc.duplicate() as Node3D
		wc_corridor.name = "CorridorWheelchair"
		furn.add_child(wc_corridor)
		wc_corridor.global_position = Vector3(0.5, 0.0, 3.2)
		wc_corridor.rotation_degrees = Vector3(0, 45, 0)
		spawned_wheelchairs.append(wc_corridor)

		var wc_emergency = orig_wc.duplicate() as Node3D
		wc_emergency.name = "EmergencyWheelchair"
		furn.add_child(wc_emergency)
		wc_emergency.global_position = Vector3(-8.8, 0.0, -4.5)
		wc_emergency.rotation_degrees = Vector3(0, -90, 0)
		spawned_wheelchairs.append(wc_emergency)

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
		{"class": "wheelchair", "names": ["LobbyWheelchair", "CorridorWheelchair", "EmergencyWheelchair"]},
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

	# Lập chỉ mục toàn bộ NPC Bác sĩ, Y tá và Bệnh nhân
	var all_children = furn.find_children("*", "Node3D", true, false)
	for npc in all_children:
		var n_name = npc.name
		if n_name.begins_with("WaitingPatient") or n_name.begins_with("Patient"):
			var aabb = _compute_node_aabb(npc)
			prop_catalog.append({
				"class_name": "patient",
				"class_id": UNIFIED_CLASS_MAP["patient"],
				"node_name": n_name,
				"node": npc,
				"local_aabb": aabb,
				"center_offset": aabb.get_center(),
				"look_height": 0.90
			})
		elif n_name.begins_with("MedicalStaff") or n_name.begins_with("Doctor"):
			var aabb = _compute_node_aabb(npc)
			prop_catalog.append({
				"class_name": "doctor",
				"class_id": UNIFIED_CLASS_MAP["doctor"],
				"node_name": n_name,
				"node": npc,
				"local_aabb": aabb,
				"center_offset": aabb.get_center(),
				"look_height": 1.10
			})

	print("[RobotPatrol] Đã lập chỉ mục ", prop_catalog.size(), " đối tượng y tế và nhân vật lâm sàng.")

func _compute_node_aabb(root_node: Node3D) -> AABB:
	var total_aabb := AABB()
	var found := false

	var is_npc = root_node.name.begins_with("WaitingPatient") or root_node.name.begins_with("MedicalStaff") or (root_node is HospitalNpc) or root_node.name.begins_with("Patient") or root_node.name.begins_with("Doctor")

	# Với NPC (Bác sĩ, Y tá, Bệnh nhân): MeshInstance3D trên FBX skinned mesh có get_aabb() bị sai trục / bind-pose nằm bẹp (Y ~0.35m).
	# Do đó với NPC, ưu tiên TUYỆT ĐỐI CollisionShape3D để lấy thể tích người đứng/ngồi thực tế (Y 1.35m - 1.75m).
	if is_npc:
		var shapes = root_node.find_children("*", "CollisionShape3D", true, false)
		for s in shapes:
			var cs = s as CollisionShape3D
			if cs and cs.shape:
				var sz = Vector3(0.5, 1.75, 0.5)
				if cs.shape is BoxShape3D:
					sz = (cs.shape as BoxShape3D).size
				elif cs.shape is CapsuleShape3D:
					var cap = cs.shape as CapsuleShape3D
					sz = Vector3(cap.radius * 2.0, cap.height, cap.radius * 2.0)
				var s_aabb = AABB(-sz * 0.5, sz)
				var rel_xform = root_node.global_transform.affine_inverse() * cs.global_transform
				var x_aabb = rel_xform * s_aabb
				if not found:
					total_aabb = x_aabb
					found = true
				else:
					total_aabb = total_aabb.merge(x_aabb)
		if found:
			return total_aabb

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
				elif cs.shape is CapsuleShape3D:
					var cap = cs.shape as CapsuleShape3D
					sz = Vector3(cap.radius * 2.0, cap.height, cap.radius * 2.0)
				var s_aabb = AABB(-sz * 0.5, sz)
				var rel_xform = root_node.global_transform.affine_inverse() * cs.global_transform
				var x_aabb = rel_xform * s_aabb
				if not found:
					total_aabb = x_aabb
					found = true
				else:
					total_aabb = total_aabb.merge(x_aabb)

	if not found:
		total_aabb = AABB(Vector3(-0.35, 0.0, -0.35), Vector3(0.7, 1.2, 0.7))

	return total_aabb

func _run_patrol_pipeline() -> void:
	var start_ms = Time.get_ticks_msec()
	_open_all_sliding_doors()
	_build_catalog()

	var abs_out_dir: String = ProjectSettings.globalize_path(output_root_dir)
	var abs_img_dir: String = abs_out_dir.path_join("images")
	var abs_lbl_dir: String = abs_out_dir.path_join("labels")
	DirAccess.make_dir_recursive_absolute(abs_img_dir)
	DirAccess.make_dir_recursive_absolute(abs_lbl_dir)

	print("==================================================================")
	print("🤖 BẮT ĐẦU CHUYẾN TUẦN TRA ROBOT TOÀN BỘ BỆNH VIỆN (PATROL TEST)")
	print("📂 Thư mục xuất: ", abs_out_dir)
	print("🚶 Tổng số Waypoints chính: ", PATROL_WAYPOINTS.size())
	print("==================================================================")

	var vp: Viewport = get_viewport()
	vp.size = capture_resolution

	# Nội suy các bước di chuyển mịn giữa các Waypoints
	var sampled_poses: Array[Dictionary] = _generate_dense_patrol_poses()
	print("📸 Tổng số khung hình tuần tra dự kiến: ", sampled_poses.size(), " frames.")

	for i in range(sampled_poses.size()):
		var pose = sampled_poses[i]
		var fid = "PATROL_%04d" % (_total_saved + 1)
		
		# Đặt camera robot tại pose
		robot_camera.global_position = pose["pos"]
		robot_camera.look_at(pose["pos"] + pose["forward"], Vector3.UP)

		await _capture_and_label(abs_img_dir, abs_lbl_dir, fid, pose["zone"])

	var elapsed_sec = (Time.get_ticks_msec() - start_ms) / 1000.0
	_write_dataset_manifest(abs_out_dir, elapsed_sec)

	print("==================================================================")
	print("✅ HOÀN TẤT CHUYẾN TUẦN TRA BỆNH VIỆN CỦA ROBOT THÀNH CÔNG!")
	print("📊 Tổng frames tuần tra: ", _total_saved)
	print("📊 Thống kê hộp nhãn 17 classes thu thập: ", _class_box_counts)
	print("⏱️ Thời gian thực thi: ", elapsed_sec, " giây.")
	print("==================================================================")
	get_tree().quit(0)

func _generate_dense_patrol_poses() -> Array[Dictionary]:
	var poses: Array[Dictionary] = []
	var cam_h = 1.18 # Chiều cao mắt camera robot

	for w_idx in range(PATROL_WAYPOINTS.size() - 1):
		var p1 = PATROL_WAYPOINTS[w_idx]
		var p2 = PATROL_WAYPOINTS[w_idx + 1]

		var seg_len = p1.distance_to(p2)
		var num_steps = maxi(1, int(round(seg_len / step_distance)))
		var dir_norm = (p2 - p1).normalized()

		for s in range(num_steps):
			var t = float(s) / float(num_steps)
			var cur_pos = p1.lerp(p2, t)
			cur_pos.y = cam_h

			# Tiếp tuyến hướng nhìn mượt mà (ngẩng xuống nhẹ 5 độ)
			var forward = Vector3(dir_norm.x, -tan(deg_to_rad(5.0)), dir_norm.z).normalized()

			# Xác định phân khu (Zone)
			var zone = _identify_zone(cur_pos)

			poses.append({
				"pos": cur_pos,
				"forward": forward,
				"zone": zone
			})

	return poses

func _identify_zone(pos: Vector3) -> String:
	if pos.z > 2.0:
		if pos.x < -6.0: return "Pharmacy"
		elif pos.x > 6.0: return "Elevator_Restroom"
		else: return "Lobby_Reception"
	elif pos.z < -2.0:
		if pos.x < -4.0: return "Emergency_Ward"
		elif pos.x > 4.0: return "Clinic_Room"
		else: return "Blood_Lab"
	else:
		return "Main_Corridor"

func _capture_and_label(abs_img_dir: String, abs_lbl_dir: String, frame_id: String, zone: String) -> void:
	await get_tree().physics_frame
	await get_tree().process_frame
	if DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw

	var full_img_path = abs_img_dir.path_join("%s.jpg" % frame_id)
	var img: Image = get_viewport().get_texture().get_image()
	if img:
		img.save_jpg(full_img_path, 0.94)

	var yolo_lines: PackedStringArray = []
	var detected_props: Array[String] = []
	var vp_w = float(capture_resolution.x)
	var vp_h = float(capture_resolution.y)

	var space_state = get_world_3d().direct_space_state

	for prop in prop_catalog:
		var node = prop["node"] as Node3D
		if node == null or not node.is_inside_tree() or not node.visible: continue

		var aabb: AABB = prop["local_aabb"]
		var is_npc: bool = prop["class_name"] == "patient" or prop["class_name"] == "doctor"
		var local_pts: Array[Vector3] = []

		if is_npc:
			var h_min = aabb.position.y
			var h_max = aabb.end.y
			var c = aabb.get_center()
			var rx = aabb.size.x * 0.45
			var rz = aabb.size.z * 0.45
			var y_levels = [h_min + 0.1, h_min + 0.35 * aabb.size.y, h_min + 0.65 * aabb.size.y, h_max - 0.08]
			for y_val in y_levels:
				for deg in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
					var rad = deg_to_rad(deg)
					local_pts.append(Vector3(c.x + rx * cos(rad), y_val, c.z + rz * sin(rad)))
			local_pts.append(Vector3(c.x, h_max, c.z))
			local_pts.append(Vector3(c.x, h_min + 0.05, c.z))
			local_pts.append(c)
		else:
			var min_p = aabb.position
			var max_p = aabb.end
			var c = aabb.get_center()
			local_pts = [
				Vector3(min_p.x, min_p.y, min_p.z),
				Vector3(max_p.x, min_p.y, min_p.z),
				Vector3(min_p.x, max_p.y, min_p.z),
				Vector3(max_p.x, max_p.y, min_p.z),
				Vector3(min_p.x, min_p.y, max_p.z),
				Vector3(max_p.x, min_p.y, max_p.z),
				Vector3(min_p.x, max_p.y, max_p.z),
				Vector3(max_p.x, max_p.y, max_p.z),
				Vector3(c.x, min_p.y, c.z),
				Vector3(c.x, max_p.y, c.z),
				Vector3(min_p.x, c.y, c.z),
				Vector3(max_p.x, c.y, c.z),
				Vector3(c.x, c.y, min_p.z),
				Vector3(c.x, c.y, max_p.z),
				c
			]

		var visible_screens: Array[Vector2] = []
		var front_count = 0

		for lp in local_pts:
			var wp = node.global_transform * lp
			if robot_camera.is_position_behind(wp):
				continue
			front_count += 1
			var uv = robot_camera.unproject_position(wp)
			if uv.x >= -25.0 and uv.x <= vp_w + 25.0 and uv.y >= -25.0 and uv.y <= vp_h + 25.0:
				var q = PhysicsRayQueryParameters3D.create(robot_camera.global_position, wp)
				var hit = space_state.intersect_ray(q)
				if hit.is_empty() or hit.get("collider") == node or node.is_ancestor_of(hit.get("collider")):
					visible_screens.append(uv)

		var min_vis_pts = 4 if is_npc else 3
		if visible_screens.size() >= min_vis_pts and front_count >= 3:
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
				var nw = bw / vp_w
				var nh = bh / vp_h

				var cid = prop["class_id"]
				var cname = prop["class_name"]
				yolo_lines.append("%d %.6f %.6f %.6f %.6f" % [cid, cx, cy, nw, nh])
				detected_props.append(prop["node_name"])
				_class_box_counts[cname] = _class_box_counts.get(cname, 0) + 1

	var lbl_file = abs_lbl_dir.path_join("%s.txt" % frame_id)
	var fa = FileAccess.open(lbl_file, FileAccess.WRITE)
	if fa:
		for line in yolo_lines:
			fa.store_line(line)
		fa.close()

	_total_saved += 1
	if _total_saved % 40 == 0 or _total_saved == 1:
		print("  🚶 Robot tuần tra: Frame ", _total_saved, " | Phân khu: ", zone, " -> Phát hiện: ", detected_props.size(), " đối tượng")

func _write_dataset_manifest(out_dir: String, elapsed_sec: float) -> void:
	var manifest = {
		"dataset_type": "hospital_robot_patrol_continuous_testset",
		"total_frames": _total_saved,
		"camera_setup": {
			"fov_deg": robot_camera.fov,
			"height_m": 1.18,
			"pitch_deg": -5.0,
			"resolution": [capture_resolution.x, capture_resolution.y]
		},
		"classes": {
			"0": "patient", "1": "doctor", "2": "wheelchair", "3": "queue_kiosk",
			"4": "reception_counter", "5": "sanitizer_stand", "6": "pccc_station",
			"7": "medical_waste_bin", "8": "water_dispenser", "9": "wc_station",
			"10": "elevator_doors", "11": "hospital_bed", "12": "stretcher_cart",
			"13": "iv_stand", "14": "medical_trolley", "15": "vital_monitor",
			"16": "hospital_chair"
		},
		"class_box_counts": _class_box_counts,
		"generation_time_sec": elapsed_sec,
		"timestamp": Time.get_datetime_string_from_system(true)
	}
	var f = FileAccess.open(out_dir.path_join("manifest.json"), FileAccess.WRITE)
	if f:
		f.store_string(JSON.stringify(manifest, "\t"))
		f.close()
