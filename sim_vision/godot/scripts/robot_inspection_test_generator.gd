# robot_inspection_test_generator.gd
## BỘ SINH DỮ LIỆU KIỂM THỬ TRẠM QUAN SÁT MỤC TIÊU ROBOT (TARGETED ROBOT INSPECTION BENCHMARK)
## Mục tiêu: Sinh riêng bộ ảnh kiểm thử bổ sung tập trung 100% vào các trạm quan sát mục tiêu thiết bị y tế.
## KHÔNG lặp lại các đoạn tuần tra hành lang của bộ test cũ.
## 17 Trạm quan sát chuyên biệt, tiếp cận đa cự ly và đa góc độ từ tầm mắt robot (h=1.18m, fov=70 deg).
extends Node3D

@export var robot_camera: Camera3D
@export var room_node: Node3D
@export var capture_resolution: Vector2i = Vector2i(640, 480)
@export var output_root_dir: String = "res://../../data/vision/dataset/hospital_robot_inspection_testset"

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
var spawned_wheelchairs: Array[Node3D] = []

func _ready() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0

	for c in CLASS_NAMES:
		_class_box_counts[c] = 0

	call_deferred("_run_inspection_pipeline")

func _open_all_sliding_doors() -> void:
	if room_node == null: return
	var doors = room_node.find_children("*", "SlidingDoor", true, false)
	for d in doors:
		if d is SlidingDoor:
			d.force_open()
	print("[TargetedInspection] Đã mở toàn bộ ", doors.size(), " cửa trượt âm tường để robot quan sát tự do.")

func _build_catalog() -> void:
	prop_catalog.clear()
	if room_node == null:
		push_error("Room node is null!")
		return

	var furn = room_node.find_child("Furniture", true, false)
	if furn == null:
		push_error("Furniture node not found in Room!")
		return

	var orig_wc = furn.find_child("LobbyWheelchair", true, false) as Node3D
	if orig_wc and spawned_wheelchairs.is_empty():
		var wc_corridor = orig_wc.duplicate() as Node3D
		wc_corridor.name = "CorridorWheelchair"
		furn.add_child(wc_corridor)
		wc_corridor.global_position = Vector3(0.5, 0.0, 3.2)
		wc_corridor.rotation_degrees = Vector3(0, 45, 0)
		spawned_wheelchairs.append(wc_corridor)

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
		{"class": "hospital_chair", "names": ["WaitingBenchWest", "WaitingBenchEast", "WaitingBenchWestBack", "WaitingBenchEastBack", "WcWaitingBench", "PharmacyPlasticChair", "PharmacyPlasticChair2", "WcPlasticChair"]}
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

	print("[TargetedInspection] Đã lập chỉ mục ", prop_catalog.size(), " đối tượng y tế và nhân vật lâm sàng.")

func _compute_node_aabb(root_node: Node3D) -> AABB:
	var total_aabb := AABB()
	var found := false

	var is_npc = root_node.name.begins_with("WaitingPatient") or root_node.name.begins_with("MedicalStaff") or (root_node is HospitalNpc) or root_node.name.begins_with("Patient") or root_node.name.begins_with("Doctor")

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

func _run_inspection_pipeline() -> void:
	var start_ms = Time.get_ticks_msec()
	_open_all_sliding_doors()
	_build_catalog()

	var abs_out_dir: String = ProjectSettings.globalize_path(output_root_dir)
	var abs_img_dir: String = abs_out_dir.path_join("images")
	var abs_lbl_dir: String = abs_out_dir.path_join("labels")
	
	# Xóa sạch dữ liệu cũ
	if DirAccess.dir_exists_absolute(abs_img_dir):
		var old_files = DirAccess.get_files_at(abs_img_dir)
		for f in old_files:
			DirAccess.remove_absolute(abs_img_dir.path_join(f))
	if DirAccess.dir_exists_absolute(abs_lbl_dir):
		var old_files = DirAccess.get_files_at(abs_lbl_dir)
		for f in old_files:
			DirAccess.remove_absolute(abs_lbl_dir.path_join(f))

	DirAccess.make_dir_recursive_absolute(abs_img_dir)
	DirAccess.make_dir_recursive_absolute(abs_lbl_dir)

	print("==================================================================")
	print("🎯 BẮT ĐẦU CHUYẾN QUÉT TRẠM MỤC TIÊU ĐỘC LẬP (TARGETED INSPECTION BENCHMARK)")
	print("📂 Thư mục xuất: ", abs_out_dir)
	print("==================================================================")

	var vp: Viewport = get_viewport()
	vp.size = capture_resolution

	# Chỉ sinh các tư thế trạm quan sát mục tiêu chuyên biệt (100% không lặp cảnh tuần tra cũ)
	var station_poses: Array[Dictionary] = _generate_target_inspection_poses()
	print("🚀 TỔNG CỘNG KHUNG HÌNH TRẠM MỤC TIÊU BỔ SUNG: ", station_poses.size(), " frames.")

	for i in range(station_poses.size()):
		var pose = station_poses[i]
		var fid = "INSPECT_%04d" % (_total_saved + 1)
		
		robot_camera.global_position = pose["pos"]
		robot_camera.look_at(pose["pos"] + pose["forward"], Vector3.UP)

		await _capture_and_label(abs_img_dir, abs_lbl_dir, fid, pose["station_name"])

	var elapsed_sec = (Time.get_ticks_msec() - start_ms) / 1000.0
	_write_dataset_manifest(abs_out_dir, elapsed_sec)

	print("==================================================================")
	print("✅ HOÀN TẤT BỘ TEST TRẠM MỤC TIÊU BỔ SUNG THÀNH CÔNG!")
	print("📊 Tổng frames độc lập thu thập: ", _total_saved)
	print("📊 Thống kê hộp nhãn 17 classes: ")
	for cname in CLASS_NAMES:
		var cnt = _class_box_counts.get(cname, 0)
		print("   • %-20s: %4d boxes" % [cname, cnt])
	print("⏱️ Thời gian thực thi: ", elapsed_sec, " giây.")
	print("==================================================================")
	get_tree().quit(0)

func _generate_target_inspection_poses() -> Array[Dictionary]:
	var poses: Array[Dictionary] = []
	var cam_h = 1.18

	var stations: Array[Dictionary] = [
		# 1. Cáng cấp cứu (EmergencyStretcherCart tại -10.30, 0.0, -2.55)
		{
			"name": "StretcherCart_Station",
			"target": Vector3(-10.30, 0.70, -2.55),
			"standpoints": [
				Vector3(-8.20, cam_h, -2.55), # Cự ly 2.1m trực diện
				Vector3(-8.80, cam_h, -2.55), # Cự ly 1.5m cận cảnh trực diện
				Vector3(-7.80, cam_h, -2.80), # Cự ly 2.5m xiên
				Vector3(-8.50, cam_h, -2.20), # Cự ly 1.8m
				Vector3(-8.00, cam_h, -3.20), # Cự ly 2.4m
				Vector3(-9.00, cam_h, -3.00), # Cự ly 1.4m
				Vector3(-8.20, cam_h, -2.20)  # Cự ly 2.1m
			],
			"yaws": [-15.0, -8.0, 0.0, 8.0, 15.0]
		},
		# 2. Cửa thang máy bệnh viện (HospitalElevator tại 11.88, 0.0, 4.60)
		{
			"name": "Elevator_Station",
			"target": Vector3(11.88, 1.25, 4.60),
			"standpoints": [
				Vector3(9.00, cam_h, 4.60),  # Cự ly 2.88m trực diện
				Vector3(9.80, cam_h, 4.60),  # Cự ly 2.08m trực diện
				Vector3(10.50, cam_h, 4.60), # Cự ly 1.38m cận cảnh
				Vector3(9.20, cam_h, 3.80),  # Cự ly 2.8m xiên Bắc
				Vector3(9.20, cam_h, 5.40),  # Cự ly 2.8m xiên Nam
				Vector3(10.00, cam_h, 3.90), # Cự ly 2.0m
				Vector3(10.00, cam_h, 5.30)  # Cự ly 2.0m
			],
			"yaws": [-15.0, -8.0, 0.0, 8.0, 15.0]
		},
		# 3. Cây nước uống (LobbyWaterDispenser tại 5.45, 0.0, 6.60 - mặt hướng Tây)
		{
			"name": "WaterDispenser_Station",
			"target": Vector3(5.45, 0.85, 6.60),
			"standpoints": [
				Vector3(4.20, cam_h, 6.60), # Cự ly 1.25m trực diện
				Vector3(3.60, cam_h, 6.60), # Cự ly 1.85m trực diện
				Vector3(4.00, cam_h, 5.80), # Cự ly 1.65m xiên
				Vector3(4.50, cam_h, 5.50), # Cự ly 1.45m xiên
				Vector3(3.20, cam_h, 6.00), # Cự ly 2.3m
				Vector3(4.20, cam_h, 7.30), # Cự ly 1.4m
				Vector3(3.80, cam_h, 7.00)  # Cự ly 1.7m
			],
			"yaws": [-15.0, -8.0, 0.0, 8.0, 15.0]
		},
		# 4. Trạm PCCC hành lang (CorridorPccc tại -5.00, 0.0, -1.72)
		{
			"name": "PCCC_Station",
			"target": Vector3(-5.00, 1.10, -1.72),
			"standpoints": [
				Vector3(-5.00, cam_h, -0.20), # Cự ly 1.5m trực diện
				Vector3(-4.00, cam_h, 0.20),  # Cự ly 2.1m xiên Đông
				Vector3(-6.00, cam_h, 0.20),  # Cự ly 2.1m xiên Tây
				Vector3(-5.00, cam_h, 0.80),  # Cự ly 2.5m
				Vector3(-3.80, cam_h, -0.60), # Cự ly 1.6m
				Vector3(-6.20, cam_h, -0.60)  # Cự ly 1.6m
			],
			"yaws": [-14.0, -7.0, 0.0, 7.0, 14.0]
		},
		# 5. Kiosk lấy số (LobbyQueueKiosk tại 2.20, 0.0, 7.55)
		{
			"name": "QueueKiosk_Station",
			"target": Vector3(2.20, 0.90, 7.55),
			"standpoints": [
				Vector3(2.20, cam_h, 5.80),
				Vector3(1.20, cam_h, 6.50),
				Vector3(3.20, cam_h, 6.50),
				Vector3(2.20, cam_h, 6.40),
				Vector3(1.80, cam_h, 6.00)
			],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		# 6. Trạm nhà vệ sinh (RestroomStation tại 9.20, 0.0, 8.60)
		{
			"name": "Restroom_Station",
			"target": Vector3(9.20, 1.10, 8.60),
			"standpoints": [
				Vector3(9.20, cam_h, 6.80),
				Vector3(8.00, cam_h, 7.40),
				Vector3(10.20, cam_h, 7.20),
				Vector3(9.20, cam_h, 7.50)
			],
			"yaws": [-14.0, -7.0, 0.0, 7.0, 14.0]
		},
		# 7. Quầy thuốc bệnh viện (PharmacyCounter tại -9.00, 0.0, 7.60)
		{
			"name": "Pharmacy_Station",
			"target": Vector3(-9.00, 0.70, 7.60),
			"standpoints": [
				Vector3(-9.00, cam_h, 5.60),
				Vector3(-7.80, cam_h, 6.20),
				Vector3(-10.20, cam_h, 6.20)
			],
			"yaws": [-14.0, -7.0, 0.0, 7.0, 14.0]
		},
		# 8. Quầy tiếp đón sảnh (ReceptionCounter tại 0.00, 0.0, 7.80)
		{
			"name": "Reception_Station",
			"target": Vector3(0.00, 0.70, 7.80),
			"standpoints": [
				Vector3(0.00, cam_h, 5.80),
				Vector3(-1.40, cam_h, 6.20),
				Vector3(1.40, cam_h, 6.20)
			],
			"yaws": [-14.0, -7.0, 0.0, 7.0, 14.0]
		},
		# 9. Máy theo dõi sinh hiệu Cấp cứu & Phòng khám
		{
			"name": "VitalMonitor_Emergency_Station",
			"target": Vector3(-9.55, 1.10, -6.80),
			"standpoints": [
				Vector3(-8.20, cam_h, -6.80),
				Vector3(-9.00, cam_h, -5.30)
			],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		{
			"name": "VitalMonitor_Clinic_Station",
			"target": Vector3(10.45, 1.10, -5.45),
			"standpoints": [
				Vector3(9.00, cam_h, -5.45),
				Vector3(9.80, cam_h, -4.10)
			],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		# 10. Cọc truyền dịch (IV Stand) Cấp cứu & Phòng khám
		{
			"name": "IVStand_Emergency_Station",
			"target": Vector3(-6.65, 1.30, -7.20),
			"standpoints": [
				Vector3(-6.65, cam_h, -5.60),
				Vector3(-5.40, cam_h, -6.80)
			],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		{
			"name": "IVStand_Clinic_Station",
			"target": Vector3(11.30, 1.30, -5.55),
			"standpoints": [
				Vector3(9.80, cam_h, -5.55),
				Vector3(10.60, cam_h, -4.20)
			],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		# 11. Xe lăn (Wheelchair) Sảnh, Cấp cứu, Hành lang
		{
			"name": "Wheelchair_Lobby_Station",
			"target": Vector3(-4.80, 0.60, 7.60),
			"standpoints": [Vector3(-3.80, cam_h, 6.80)],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		{
			"name": "Wheelchair_Emergency_Station",
			"target": Vector3(-11.20, 0.60, -5.50),
			"standpoints": [Vector3(-9.80, cam_h, -5.50)],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		{
			"name": "Wheelchair_Corridor_Station",
			"target": Vector3(0.50, 0.60, 3.20),
			"standpoints": [Vector3(0.50, cam_h, 1.80)],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		# 12. Xe đẩy y tế (Medical Trolley) Cấp cứu 1 & 2, Lab, Clinic
		{
			"name": "Trolley_Emergency1_Station",
			"target": Vector3(-9.45, 0.70, -7.80),
			"standpoints": [Vector3(-8.20, cam_h, -7.20)],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		{
			"name": "Trolley_Emergency2_Station",
			"target": Vector3(-4.75, 0.70, -7.80),
			"standpoints": [Vector3(-5.80, cam_h, -7.20)],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		{
			"name": "Trolley_Lab_Station",
			"target": Vector3(0.10, 0.70, -8.35),
			"standpoints": [Vector3(0.10, cam_h, -6.90)],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		{
			"name": "Trolley_Clinic_Station",
			"target": Vector3(11.30, 0.70, -2.75),
			"standpoints": [Vector3(9.80, cam_h, -2.75)],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		# 13. Giường bệnh (Hospital Bed) Cấp cứu 1 & 2, Phòng khám
		{
			"name": "Bed_Emergency1_Station",
			"target": Vector3(-10.50, 0.60, -7.40),
			"standpoints": [Vector3(-8.80, cam_h, -6.50)],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		{
			"name": "Bed_Emergency2_Station",
			"target": Vector3(-5.80, 0.60, -7.40),
			"standpoints": [Vector3(-6.80, cam_h, -6.50)],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		{
			"name": "Bed_Clinic_Station",
			"target": Vector3(11.31, 0.60, -4.17),
			"standpoints": [Vector3(9.80, cam_h, -4.17)],
			"yaws": [-12.0, -6.0, 0.0, 6.0, 12.0]
		},
		# 14. Trạm khử khuẩn (Sanitizer Stand) Sảnh & Hành lang
		{
			"name": "Sanitizer_Lobby_Station",
			"target": Vector3(-2.55, 0.90, 7.55),
			"standpoints": [
				Vector3(-2.55, cam_h, 6.00),
				Vector3(-1.80, cam_h, 6.50),
				Vector3(-3.20, cam_h, 6.50)
			],
			"yaws": [-12.0, 0.0, 12.0]
		},
		{
			"name": "Sanitizer_Corridor_Station",
			"target": Vector3(3.20, 0.90, -1.55),
			"standpoints": [
				Vector3(3.20, cam_h, 0.00),
				Vector3(2.40, cam_h, -0.60),
				Vector3(4.00, cam_h, -0.60)
			],
			"yaws": [-12.0, 0.0, 12.0]
		},
		# 15. Thùng rác y tế (Medical Waste Bin)
		{
			"name": "WasteBin_CorridorEast_Station",
			"target": Vector3(5.20, 0.40, -1.62),
			"standpoints": [Vector3(5.20, cam_h, 0.00)],
			"yaws": [-12.0, 0.0, 12.0]
		},
		{
			"name": "WasteBin_CorridorWest_Station",
			"target": Vector3(-5.20, 0.40, -1.62),
			"standpoints": [Vector3(-5.20, cam_h, 0.00)],
			"yaws": [-12.0, 0.0, 12.0]
		},
		{
			"name": "WasteBin_Emergency_Station",
			"target": Vector3(-6.10, 0.40, -2.42),
			"standpoints": [Vector3(-6.10, cam_h, -3.60)],
			"yaws": [-12.0, 0.0, 12.0]
		},
		{
			"name": "WasteBin_Lab_Station",
			"target": Vector3(1.80, 0.40, -2.42),
			"standpoints": [Vector3(1.80, cam_h, -3.60)],
			"yaws": [-12.0, 0.0, 12.0]
		},
		{
			"name": "WasteBin_Clinic_Station",
			"target": Vector3(6.20, 0.40, -2.42),
			"standpoints": [Vector3(6.20, cam_h, -3.60)],
			"yaws": [-12.0, 0.0, 12.0]
		},
		# 16. Nhân viên y tế (Doctor / Nurse)
		{
			"name": "Doctor_Station",
			"target": Vector3(-3.20, 1.10, -1.35),
			"standpoints": [
				Vector3(-2.00, cam_h, -0.60),
				Vector3(-3.20, cam_h, 0.00)
			],
			"yaws": [-12.0, 0.0, 12.0]
		},
		{
			"name": "Nurse_Station",
			"target": Vector3(-1.90, 1.10, 7.45),
			"standpoints": [
				Vector3(-1.90, cam_h, 6.00),
				Vector3(-0.80, cam_h, 6.60)
			],
			"yaws": [-12.0, 0.0, 12.0]
		},
		# 17. Ghế chờ và Người bệnh (Waiting Patient & Bench quan sát trực diện từ sảnh)
		{
			"name": "WaitingPatient_West_Station",
			"target": Vector3(-3.80, 0.65, 4.20),
			"standpoints": [
				Vector3(-2.20, cam_h, 4.20),
				Vector3(-3.80, cam_h, 2.80)
			],
			"yaws": [-12.0, 0.0, 12.0]
		},
		{
			"name": "WaitingPatient_East_Station",
			"target": Vector3(3.80, 0.65, 4.20),
			"standpoints": [
				Vector3(2.20, cam_h, 4.20),
				Vector3(3.80, cam_h, 2.80)
			],
			"yaws": [-12.0, 0.0, 12.0]
		}
	]

	for st in stations:
		var st_name: String = st["name"]
		var tgt: Vector3 = st["target"]
		var standpoints: Array = st["standpoints"]
		var yaws: Array = st["yaws"]

		for sp in standpoints:
			var cam_pos = sp as Vector3
			var base_dir = (tgt - cam_pos).normalized()

			for yaw_deg in yaws:
				var yaw_rad = deg_to_rad(yaw_deg)
				var dir_rot = base_dir.rotated(Vector3.UP, yaw_rad)
				var forward = Vector3(dir_rot.x, -tan(deg_to_rad(4.0)), dir_rot.z).normalized()

				poses.append({
					"pos": cam_pos,
					"forward": forward,
					"station_name": st_name
				})

	return poses

func _capture_and_label(abs_img_dir: String, abs_lbl_dir: String, frame_id: String, station_name: String) -> void:
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
				var unoccluded = hit.is_empty() or hit.get("collider") == node or node.is_ancestor_of(hit.get("collider"))
				if not unoccluded and hit.has("position"):
					var hit_dist = robot_camera.global_position.distance_to(hit.get("position"))
					var target_dist = robot_camera.global_position.distance_to(wp)
					if hit_dist >= target_dist - 0.08:
						unoccluded = true
				if unoccluded:
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
		print("  🎯 [Trạm mục tiêu] Frame %d | Trạm: %s -> %d boxes" % [_total_saved, station_name, yolo_lines.size()])

func _write_dataset_manifest(out_dir: String, elapsed_sec: float) -> void:
	var manifest = {
		"dataset_type": "hospital_robot_targeted_inspection_testset",
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
