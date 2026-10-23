# extended_dataset_generator.gd
## Bộ sinh dữ liệu mở rộng 10,800 khung hình chuẩn hóa (Nhóm N1 - N6, 36 ca kiểm thử lâm sàng N1.1 - N6.5)
## Xuất Siêu Metadata 5W1H & Physics, nhãn YOLO chuẩn hóa và hình ảnh JPEG chất lượng cao cho Notebook 02.
extends Node3D

@export var robot_camera: Camera3D
@export var world_env: WorldEnvironment
@export var main_dir_light: DirectionalLight3D
@export var actors_pool: Node3D
@export var obstacle_props: Node3D
@export var hard_negatives_pool: Node3D

@export var capture_resolution: Vector2i = Vector2i(640, 480)
@export var random_seed: int = 20261023
@export var output_root_dir: String = "res://../data/vision_eval_extended_10800"
@export var auto_start_on_ready: bool = false

var _rng := RandomNumberGenerator.new()
var _jsonl_file: FileAccess
var _jsonl_rich_file: FileAccess
var _total_saved_frames: int = 0
var _start_time_msec: int = 0

const corridor_anchor := Vector3(0.0, 0.0, 0.0)
const clinic_door_anchor := Vector3(-2.8, 0.0, 1.2)
const reception_anchor := Vector3(2.4, 0.0, -1.5)

# 15 cự ly quan sát liên tục
const DISTANCES: Array[float] = [
	0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5
]
# 5 góc mở camera (Bearing Offsets)
const CAM_YAW_OFFSETS: Array[float] = [-16.0, -8.0, 0.0, 8.0, 16.0]
# 2 độ cao cảm biến robot
const CAM_HEIGHTS: Array[float] = [0.85, 1.25]
# 2 hướng xoay NPC (hoặc 2 mức ánh sáng)
const NPC_YAWS: Array[float] = [0.0, 45.0]


func _ready() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0 # Tối đa tốc độ render offscreen
	var cmd_args: PackedStringArray = OS.get_cmdline_user_args()
	if auto_start_on_ready or ("--generate-extended-10800" in cmd_args) or ("--generate-dataset" in cmd_args):
		call_deferred("run_extended_10800_pipeline")


## Hàm điều phối chính: Chạy tuần tự 36 ca lâm sàng (N1.1 -> N6.5), đúng 300 frames/ca = 10,800 frames
func run_extended_10800_pipeline() -> void:
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
	if _jsonl_file == null or _jsonl_rich_file == null:
		push_error("[ExtendedGenerator] Không thể mở file ghi tại: " + jsonl_path)
		get_tree().quit(1)
		return

	print("==================================================================")
	print("[ExtendedGenerator] BẮT ĐẦU SINH BỘ DỮ LIỆU CỰC ĐẠI (36 CASES x 300 = 10,800 FRAMES)")
	print("[ExtendedGenerator] Thư mục đích: ", abs_out_dir)
	print("==================================================================")

	_hide_all_entities()
	_apply_lighting_profile("fluorescent_standard_450lux")

	# Chạy tuần tự 6 Nhóm Lâm Sàng (Tổng cộng 36 ca kiểm thử)
	await _generate_group_n1(abs_img_dir, abs_lbl_dir) # 7 cases x 300 = 2,100 frames
	await _generate_group_n2(abs_img_dir, abs_lbl_dir) # 5 cases x 300 = 1,500 frames
	await _generate_group_n3(abs_img_dir, abs_lbl_dir) # 7 cases x 300 = 2,100 frames
	await _generate_group_n4(abs_img_dir, abs_lbl_dir) # 8 cases x 300 = 2,400 frames
	await _generate_group_n5(abs_img_dir, abs_lbl_dir) # 4 cases x 300 = 1,200 frames
	await _generate_group_n6(abs_img_dir, abs_lbl_dir) # 5 cases x 300 = 1,500 frames

	_jsonl_file.flush()
	_jsonl_file.close()
	_jsonl_rich_file.flush()
	_jsonl_rich_file.close()

	var elapsed_sec: float = (Time.get_ticks_msec() - _start_time_msec) / 1000.0
	_write_dataset_manifest(abs_out_dir, elapsed_sec)

	print("==================================================================")
	print("[ExtendedGenerator] HOÀN TẤT THÀNH CÔNG! Đã xuất ", _total_saved_frames, " frames trong ", snappedf(elapsed_sec, 0.1), "s.")
	print("==================================================================")
	get_tree().quit(0)


# ==============================================================================
# NHÓM N1: NHÂN KHẨU HỌC & ĐỘ TƯƠNG PHẢN MÀU ÁO (7 Cases: N1.1 - N1.7 = 2,100 frames)
# ==============================================================================

func _generate_group_n1(abs_img_dir: String, abs_lbl_dir: String) -> void:
	var cases: Array[Dictionary] = [
		{
			"case_id": "N1.1", "name_vi": "Bác Sĩ Blouse Trắng Ngụy Trang Tường",
			"name_en": "Medical Doctor White Lab Coat Camouflage", "actor": "doctor_male",
			"subcat": "standing", "hazard": "Tier_1_Direct_Collision_Risk",
			"delta_e": 8.4, "who": "Bác sĩ nam áo blouse trắng dài", "why": "Thử nghiệm ngụy trang quang học tương phản thấp với tường trắng"
		},
		{
			"case_id": "N1.2", "name_vi": "Điều Dưỡng Áo Scrub Xanh Trùng Màu Rèm",
			"name_en": "Nurse Scrub Chroma Blending with Curtain", "actor": "nurse_female",
			"subcat": "standing", "hazard": "Tier_2_Tracking_Jitter_Risk",
			"delta_e": 10.1, "who": "Điều dưỡng nữ áo scrub xanh ngọc", "why": "Thử nghiệm hòa lẫn sắc độ chroma blending với rèm y tế xanh"
		},
		{
			"case_id": "N1.3", "name_vi": "Cụ Bà Tóc Bạc Áo Hoa Văn Elder F09",
			"name_en": "Elderly Female Gray Hair Floral Jacket", "actor": "elder_f09",
			"subcat": "standing", "hazard": "Tier_3_Demographic_Misclassification",
			"delta_e": 18.5, "who": "Cụ bà tóc bạc áo hoa tím sẫm cao 1.46m", "why": "Thử nghiệm biến dạng giải phẫu và mất đặc trưng tóc tối màu"
		},
		{
			"case_id": "N1.4", "name_vi": "Cụ Ông Lưng Còng Elder M03",
			"name_en": "Elderly Male Kyphosis Hunched Posture", "actor": "elder_m03",
			"subcat": "elderly_cane", "hazard": "Tier_3_Bearing_Angle_Error",
			"delta_e": 14.2, "who": "Cụ ông lưng còng áo len xám", "why": "Thử nghiệm cột sống nghiêng 35 độ làm lệch tâm trọng trường"
		},
		{
			"case_id": "N1.5", "name_vi": "Cụ Ông Áo Kẻ Sọc Ho Mệt Mỏi Elder M14",
			"name_en": "Elderly Male Striped Shirt Texture Noise", "actor": "elder_m14",
			"subcat": "standing", "hazard": "Tier_3_Confidence_Fluctuation",
			"delta_e": 16.0, "who": "Cụ ông áo kẻ sọc mảnh tương phản", "why": "Thử nghiệm nhiễu hoa văn sọc mảnh trên mạng tích chập"
		},
		{
			"case_id": "N1.6", "name_vi": "Bệnh Nhân Nữ Áo Khoác Dài Che Chân",
			"name_en": "Female Patient Long Trench Coat", "actor": "patient_f08",
			"subcat": "standing", "hazard": "Tier_2_Bounding_Box_Shrinkage",
			"delta_e": 15.1, "who": "Bệnh nhân nữ mặc áo khoác dài qua đầu gối", "why": "Thử nghiệm mất đặc trưng phân tách 2 cẳng chân"
		},
		{
			"case_id": "N1.7", "name_vi": "Bệnh Nhân Nam Trung Niên Thể Hình Đậm",
			"name_en": "Male Patient Heavy Build Wide Aspect Ratio", "actor": "patient_m05",
			"subcat": "standing", "hazard": "Tier_3_Safe_Standard",
			"delta_e": 22.0, "who": "Bệnh nhân nam to béo vai rộng 0.62m", "why": "Thử nghiệm tỷ lệ aspect ratio bề ngang lớn"
		}
	]

	for c in cases:
		await _generate_standard_static_case(abs_img_dir, abs_lbl_dir, c, "N1", "Demographics_And_Attire_Contrast")


# ==============================================================================
# NHÓM N2: TƯ THẾ LÂM SÀNG & NGƯỜI NGỒI GHẾ (5 Cases: N2.1 - N2.5 = 1,500 frames)
# ==============================================================================

func _generate_group_n2(abs_img_dir: String, abs_lbl_dir: String) -> void:
	var cases: Array[Dictionary] = [
		{
			"case_id": "N2.1", "name_vi": "Bệnh Nhân Ngồi Ghế Băng Dài",
			"name_en": "Patient Seated on Waiting Bench", "actor": "seated_bench_patient",
			"subcat": "wheelchair", "hazard": "Tier_2_Distant_Recall_Degradation",
			"delta_e": 16.0, "who": "Bệnh nhân nam ngồi tựa lưng trên ghế băng", "why": "Thử nghiệm nén chiều cao cơ thể H/W khi ngồi ghế"
		},
		{
			"case_id": "N2.2", "name_vi": "Bệnh Nhân Ngồi Gập Người Ho Khạc 45 Độ",
			"name_en": "Patient Seated Coughing Torso Bent 45deg", "actor": "seated_cough_patient",
			"subcat": "wheelchair", "hazard": "Tier_1_Direct_Collision_Risk",
			"delta_e": 17.5, "who": "Bệnh nhân ngồi gập người ho tay che mặt", "why": "Thử nghiệm biến dạng cực hạn H/W và mất 90% mốc mặt"
		},
		{
			"case_id": "N2.3", "name_vi": "Bệnh Nhân Ngồi Buông Thõng Mệt Mỏi",
			"name_en": "Patient Seated Exhausted Limbs Limp", "actor": "seated_exhausted_patient",
			"subcat": "wheelchair", "hazard": "Tier_3_Box_Compression",
			"delta_e": 15.0, "who": "Cụ bà ngồi ngả đầu mỏi mệt hai tay buông thõng", "why": "Thử nghiệm cánh tay lọt thỏm giữa đùi làm biến dạng gradient"
		},
		{
			"case_id": "N2.4", "name_vi": "Bác Sĩ Cúi Đầu Đọc Bệnh Án Clipboard",
			"name_en": "Doctor Reading Medical Clipboard", "actor": "doctor_clipboard",
			"subcat": "standing", "hazard": "Tier_3_Chest_Fragmentation",
			"delta_e": 12.0, "who": "Bác sĩ cúi đầu 30 độ nâng kẹp bệnh án ngang ngực", "why": "Thử nghiệm che khuất 35% vùng ngực và cằm bởi kẹp hồ sơ"
		},
		{
			"case_id": "N2.5", "name_vi": "Cụ Bà Ngồi Bắt Chéo Chân Ghế Băng",
			"name_en": "Elderly Female Seated Crossed Legs", "actor": "elder_f09",
			"subcat": "wheelchair", "hazard": "Tier_3_Safe_Standard",
			"delta_e": 18.0, "who": "Cụ bà ngồi bắt chéo chân chữ ngũ trên ghế", "why": "Thử nghiệm dáng chân tam giác bất đối xứng"
		}
	]

	for c in cases:
		await _generate_standard_static_case(abs_img_dir, abs_lbl_dir, c, "N2", "Clinical_Poses_And_Seated_Morphology")


# ==============================================================================
# NHÓM N3: CHE KHUẤT BỞI ĐẠO CỤ Y TẾ MỚI (7 Cases: N3.1 - N3.7 = 2,100 frames)
# ==============================================================================

func _generate_group_n3(abs_img_dir: String, abs_lbl_dir: String) -> void:
	var cases: Array[Dictionary] = [
		{
			"case_id": "N3.1", "name_vi": "Bệnh Nhân Truyền Dịch Bên Cột Inox",
			"name_en": "Patient with Slender IV Stand Pole Occlusion", "actor": "patient_m02",
			"prop": "iv_stand", "prop_offset": Vector3(-0.20, 0.0, 0.15),
			"subcat": "standing", "hazard": "Tier_2_Double_Box_Trap",
			"delta_e": 14.5, "who": "Bệnh nhân nam đứng sau cột truyền dịch inox", "why": "Thử nghiệm thanh kim loại mảnh cắt dọc bản đồ đặc trưng"
		},
		{
			"case_id": "N3.2", "name_vi": "Điều Dưỡng Thao Tác Xe Monitor Sinh Hiệu",
			"name_en": "Nurse Behind Vital Signs Monitor Cart", "actor": "nurse_female",
			"prop": "vital_monitor", "prop_offset": Vector3(0.0, 0.0, 0.35),
			"subcat": "standing", "hazard": "Tier_2_Lower_Body_Truncation",
			"delta_e": 11.2, "who": "Điều dưỡng nữ đứng sau xe đẩy monitor 1.2m", "why": "Thử nghiệm che khuất 65% thân dưới bởi màn hình và khay đỡ"
		},
		{
			"case_id": "N3.3", "name_vi": "Bệnh Nhân Đứng Bấm Kiosk Lấy Số",
			"name_en": "Patient Standing at Queue Kiosk", "actor": "patient_m05",
			"prop": "queue_kiosk", "prop_offset": Vector3(0.0, 0.0, 0.40),
			"subcat": "standing", "hazard": "Tier_1_Direct_Collision_Risk",
			"delta_e": 18.0, "who": "Bệnh nhân cúi bấm màn hình kiosk cao 1.55m", "why": "Thử nghiệm che khuất 70% mặt trước chỉ hở đầu và vai"
		},
		{
			"case_id": "N3.4", "name_vi": "Bệnh Nhân Lấy Nước Cây Nóng Lạnh",
			"name_en": "Patient at Water Dispenser Lateral Block", "actor": "elder_m14",
			"prop": "water_dispenser", "prop_offset": Vector3(0.30, 0.0, 0.15),
			"subcat": "standing", "hazard": "Tier_3_Center_Shift",
			"delta_e": 15.5, "who": "Cụ ông đứng cúi hứng nước bên cây nước 1.3m", "why": "Thử nghiệm che khuất một bên sườn làm lệch trọng tâm hộp bao"
		},
		{
			"case_id": "N3.5", "name_vi": "Bệnh Nhân Rửa Tay Cột Khử Khuẩn",
			"name_en": "Patient Hands Extended at Sanitizer Stand", "actor": "elder_f09",
			"prop": "sanitizer_stand", "prop_offset": Vector3(0.0, 0.0, 0.25),
			"subcat": "standing", "hazard": "Tier_3_Box_Horizontal_Inflation",
			"delta_e": 17.0, "who": "Cụ bà chụm hai tay dưới vòi xịt tự động", "why": "Thử nghiệm hai tay duỗi ra trước làm phình hộp bao"
		},
		{
			"case_id": "N3.6", "name_vi": "Hộ Lý Đẩy Xe Cáng Cấp Cứu",
			"name_en": "Staff Pushing Mobile Stretcher Cart", "actor": "doctor_male",
			"prop": "stretcher_cart", "prop_offset": Vector3(0.0, 0.0, 1.10),
			"subcat": "standing", "hazard": "Tier_2_Oversized_Bounding_Box",
			"delta_e": 10.5, "who": "Nhân viên y tế đứng ở đuôi xe cáng dài 2.1m", "why": "Thử nghiệm xe cáng chiếm tiền cảnh làm phóng đại kích thước box"
		},
		{
			"case_id": "N3.7", "name_vi": "Bệnh Nhân Đứng Sát Cạnh Standee Y Tế",
			"name_en": "Patient Adjacent to Info Standee Edge", "actor": "patient_f08",
			"prop": "info_standee", "prop_offset": Vector3(-0.35, 0.0, 0.05),
			"subcat": "standing", "hazard": "Tier_3_Edge_Adhesion",
			"delta_e": 16.5, "who": "Bệnh nhân nữ nép sát cạnh pano standee 1.8m", "why": "Thử nghiệm đường biên thẳng của standee dính vào thân người"
		}
	]

	for c in cases:
		await _generate_prop_occluded_case(abs_img_dir, abs_lbl_dir, c, "N3", "Prop_Interactions_And_Occlusion")


# ==============================================================================
# NHÓM N4: BẪY ĐẠO CỤ RỖNG (FALSE POSITIVE STRESS) (8 Cases: N4.1 - N4.8 = 2,400 frames)
# ==============================================================================

func _generate_group_n4(abs_img_dir: String, abs_lbl_dir: String) -> void:
	var cases: Array[Dictionary] = [
		{
			"case_id": "N4.1", "name_vi": "Bẫy Cột Truyền Dịch Inox Rỗng",
			"name_en": "Empty Slender IV Stand Trap", "prop": "empty_iv_stand",
			"hazard": "Tier_2_Phantom_Braking", "why": "Trụ inox cao 1.8m có móc treo đỉnh mô phỏng người đứng từ xa"
		},
		{
			"case_id": "N4.2", "name_vi": "Bẫy Cột Khử Khuẩn Tự Động Rỗng",
			"name_en": "Empty Sanitizer Stand Trap", "prop": "empty_sanitizer_stand",
			"hazard": "Tier_3_Evasive_Maneuver", "why": "Cột trắng 1.25m có đầu hộp mô phỏng trẻ em hoặc người ngồi"
		},
		{
			"case_id": "N4.3", "name_vi": "Bẫy Kiosk Bấm Số Khám Bệnh Rỗng",
			"name_en": "Empty Queue Kiosk Trap", "prop": "empty_queue_kiosk",
			"hazard": "Tier_2_Phantom_Braking", "why": "Khối chữ nhật đứng sảnh đón có gradient cạnh biên mạnh khớp mẫu người"
		},
		{
			"case_id": "N4.4", "name_vi": "Bẫy Xe Đẩy Máy Monitor Rỗng",
			"name_en": "Empty Vital Monitor Cart Trap", "prop": "empty_vital_monitor",
			"hazard": "Tier_3_Speed_Deceleration", "why": "Khung xe và màn hình mô phỏng người ngồi xe lăn"
		},
		{
			"case_id": "N4.5", "name_vi": "Bẫy Xe Cáng Cấp Cứu Rỗng",
			"name_en": "Empty Stretcher Cart Sheet Folds Trap", "prop": "empty_stretcher_cart",
			"hazard": "Tier_1_Emergency_Brake", "why": "Nếp gấp ga trải giường trắng mô phỏng người nằm cấp cứu"
		},
		{
			"case_id": "N4.6", "name_vi": "Bẫy Cây Nước Uống Rỗng",
			"name_en": "Empty Water Dispenser Trap", "prop": "empty_water_dispenser",
			"hazard": "Tier_3_False_Alarm_Low", "why": "Bình nước xanh tròn trên thân máy trụ mô phỏng phần đầu và thân"
		},
		{
			"case_id": "N4.7", "name_vi": "Bẫy Standee In Hình Y Tế Rỗng (Poster Trap)",
			"name_en": "2D Info Standee Poster Trap", "prop": "empty_info_standee",
			"hazard": "Tier_1_Infinite_Obstacle_Lock", "why": "Hình 2D in bác sĩ kích thước thật đánh lừa toàn diện mạng tích chập 2D"
		},
		{
			"case_id": "N4.8", "name_vi": "Bẫy Màn Hình Gọi Số Treo Tường",
			"name_en": "Empty Queue LED Display Trap", "prop": "empty_queue_display",
			"hazard": "Tier_3_Safe_Standard", "why": "Đèn LED phát quang chữ số thử thách loại trừ nhiễu ánh sáng cục bộ"
		}
	]

	for c in cases:
		await _generate_hard_negative_case(abs_img_dir, abs_lbl_dir, c, "N4", "Hard_Negatives_And_Traps")


# ==============================================================================
# NHÓM N5: ĐỘNG HỌC DI CHUYỂN & NÉ NGANG (4 Cases: N5.1 - N5.4 = 1,200 frames)
# ==============================================================================

func _generate_group_n5(abs_img_dir: String, abs_lbl_dir: String) -> void:
	var cases: Array[Dictionary] = [
		{
			"case_id": "N5.1", "name_vi": "Bác Sĩ Đi Tuần Tra Dọc Hành Lang Looming",
			"name_en": "Doctor Patrol Gait Looming Reverse Vector", "actor": "doctor_male",
			"mode": "looming", "hazard": "Tier_2_Looming_Box_Dilation",
			"why": "Vận tốc tiếp cận 1.08 m/s, hộp bao phóng to nhanh theo hàm 1/z"
		},
		{
			"case_id": "N5.2", "name_vi": "Né Ngang Nhường Đường Pardon Sidestep 0.55m",
			"name_en": "Pardon Sidestep Abrupt Lateral Shift", "actor": "doctor_male",
			"mode": "pardon", "hazard": "Tier_3_Steering_Angle_Chattering",
			"why": "Bước né ngang 0.55m sinh vận tốc góc 28.1 deg/s thử thách Kalman Filter"
		},
		{
			"case_id": "N5.3", "name_vi": "Cụ Già Chuyển Hướng 90 Độ Qua Ngưỡng Cửa",
			"name_en": "Elderly 90deg Turn at Clinic Doorway", "actor": "elder_m03",
			"mode": "doorway", "hazard": "Tier_2_Aspect_Transition_Drift",
			"why": "Xoay người 90 độ và bị mép cửa che khuất từng phần"
		},
		{
			"case_id": "N5.4", "name_vi": "Điều Dưỡng Đi Nhanh Lao Ra Từ Góc Khuất",
			"name_en": "Nurse Sudden Emergence at Corridor Corner", "actor": "nurse_female",
			"mode": "corner", "hazard": "Tier_1_Direct_Collision_Risk",
			"why": "Xuất hiện đột ngột cự ly 2.2m vận tốc 0.82 m/s thử độ trễ t_detect"
		}
	]

	for c in cases:
		await _generate_kinematics_case(abs_img_dir, abs_lbl_dir, c, "N5", "Kinematics_And_Dynamic_Avoidance")


# ==============================================================================
# NHÓM N6: MẬT ĐỘ XÃ HỘI & TƯƠNG TÁC LÂM SÀNG (5 Cases: N6.1 - N6.5 = 1,500 frames)
# ==============================================================================

func _generate_group_n6(abs_img_dir: String, abs_lbl_dir: String) -> void:
	var cases: Array[Dictionary] = [
		{
			"case_id": "N6.1", "name_vi": "Hai Cụ Già Ngồi Liền Kề Ghế Băng Dài",
			"name_en": "Two Elderly Seated Adjacent on Bench", "mode": "seated_pair",
			"actors": ["elder_m03", "elder_f09"], "subcats": ["wheelchair", "wheelchair"],
			"hazard": "Tier_1_Greedy_NMS_Suppression_Trap", "why": "Khoảng cách tâm 0.32m chồng lấn IoU > 40%, bẫy triệt tiêu của Greedy NMS"
		},
		{
			"case_id": "N6.2", "name_vi": "Cặp Đôi Bác Sĩ Đứng - Bệnh Nhân Ngồi",
			"name_en": "Doctor Standing Adjacent to Seated Patient", "mode": "doctor_patient",
			"actors": ["doctor_male", "elder_m03"], "subcats": ["standing", "wheelchair"],
			"hazard": "Tier_2_Multi_Scale_Vertical_Disparity", "why": "Bác sĩ cao 1.78m lồng vào bệnh nhân cao 1.15m, phân tách 2 scale FPN"
		},
		{
			"case_id": "N6.3", "name_vi": "Hàng Chờ Trước Kiosk 3 Bệnh Nhân",
			"name_en": "Three-Patient Queue Sequence at Kiosk", "mode": "queue_kiosk",
			"actors": ["patient_f08", "patient_m02", "doctor_male"], "subcats": ["standing", "standing", "standing"],
			"hazard": "Tier_2_Axial_Occlusion_Chain", "why": "Chuỗi che khuất trục dọc, người sau bị người trước che 55% - 70%"
		},
		{
			"case_id": "N6.4", "name_vi": "Bác Sĩ & Điều Dưỡng Thảo Luận Đối Mặt",
			"name_en": "Doctor Back 180deg Facing Nurse Front 0deg", "mode": "facing_pair",
			"actors": ["doctor_male", "nurse_female"], "subcats": ["standing", "standing"],
			"hazard": "Tier_3_Symmetric_Feature_Extraction", "why": "Trích xuất đồng thời người quay lưng hoàn toàn và người đối diện"
		},
		{
			"case_id": "N6.5", "name_vi": "Bệnh Nhân Được Người Nhà Dìu Đi Chậm",
			"name_en": "Coupled Twin Guided Walking Elderly and Kin", "mode": "guided_walk",
			"actors": ["elder_m03", "patient_f08"], "subcats": ["elderly_cane", "standing"],
			"hazard": "Tier_2_Coupled_Twin_Box_Fusion", "why": "Hai cơ thể dính chặt nhau thử thách phân rã twin bounding box"
		}
	]

	for c in cases:
		await _generate_crowd_case(abs_img_dir, abs_lbl_dir, c, "N6", "Crowd_Density_And_Social_Disentanglement")


# ==============================================================================
# HÀM XỬ LÝ SINH CHI TIẾT THEO CẤU HÌNH (300 FRAMES MỖI CA)
# ==============================================================================

## Sinh 300 frames cho ca tĩnh chuẩn (N1, N2)
func _generate_standard_static_case(
	abs_img_dir: String, abs_lbl_dir: String,
	c: Dictionary, group_id: String, group_name: String
) -> void:
	var actor: Node3D = _get_actor(c["actor"])
	var look_h: float = 0.70 if c["subcat"] == "wheelchair" else 0.90
	var case_idx: int = 0

	for dist in DISTANCES:               # 15 cự ly
		for cam_off in CAM_YAW_OFFSETS:    # 5 góc camera
			for cam_h in CAM_HEIGHTS:        # 2 chiều cao camera
				for yaw in NPC_YAWS:           # 2 góc xoay NPC -> 15x5x2x2 = 300 frames
					case_idx += 1
					var frame_id: String = "%s_%04d" % [c["case_id"].replace(".", "_"), case_idx]
					_hide_all_entities()

					if actor:
						actor.visible = true
						actor.global_position = corridor_anchor
						actor.rotation_degrees = Vector3(0.0, yaw, 0.0)
						_position_camera_6dof(corridor_anchor, dist, cam_off, cam_h, look_h)

					await _record_frame_and_ultra_metadata(
						abs_img_dir, abs_lbl_dir, frame_id, c["case_id"], c["name_vi"], c["name_en"],
						group_id, group_name, c["hazard"], false,
						[actor] if actor else [], [c["subcat"]],
						c.get("delta_e", 15.0), c.get("who", ""), c.get("why", ""),
						dist, cam_off, cam_h, yaw, 0
					)


## Sinh 300 frames cho ca che khuất đạo cụ (N3)
func _generate_prop_occluded_case(
	abs_img_dir: String, abs_lbl_dir: String,
	c: Dictionary, group_id: String, group_name: String
) -> void:
	var actor: Node3D = _get_actor(c["actor"])
	var prop: Node3D = _get_prop(obstacle_props, c["prop"])
	var look_h: float = 0.70 if c["subcat"] == "wheelchair" else 0.90
	var case_idx: int = 0

	for dist in DISTANCES:
		for cam_off in CAM_YAW_OFFSETS:
			for cam_h in CAM_HEIGHTS:
				for yaw in NPC_YAWS:
					case_idx += 1
					var frame_id: String = "%s_%04d" % [c["case_id"].replace(".", "_"), case_idx]
					_hide_all_entities()

					if actor and prop:
						actor.visible = true; prop.visible = true
						actor.global_position = corridor_anchor
						actor.rotation_degrees = Vector3(0.0, yaw, 0.0)
						prop.global_position = corridor_anchor + c["prop_offset"]
						_position_camera_6dof(corridor_anchor, dist, cam_off, cam_h, look_h)

					await _record_frame_and_ultra_metadata(
						abs_img_dir, abs_lbl_dir, frame_id, c["case_id"], c["name_vi"], c["name_en"],
						group_id, group_name, c["hazard"], false,
						[actor] if actor else [], [c["subcat"]],
						c.get("delta_e", 15.0), c.get("who", ""), c.get("why", ""),
						dist, cam_off, cam_h, yaw, 0
					)


## Sinh 300 frames cho ca bẫy đạo cụ rỗng (N4 - Ground Truth RỖNG)
func _generate_hard_negative_case(
	abs_img_dir: String, abs_lbl_dir: String,
	c: Dictionary, group_id: String, group_name: String
) -> void:
	var prop: Node3D = _get_prop(hard_negatives_pool, c["prop"])
	var case_idx: int = 0

	for dist in DISTANCES:
		for cam_off in CAM_YAW_OFFSETS:
			for cam_h in CAM_HEIGHTS:
				for light_mode in [0, 1]: # 0 = chuẩn 450lux, 1 = hành lang tối 220lux
					case_idx += 1
					var frame_id: String = "%s_%04d" % [c["case_id"].replace(".", "_"), case_idx]
					_hide_all_entities()

					if light_mode == 0:
						_apply_lighting_profile("fluorescent_standard_450lux")
					else:
						_apply_lighting_profile("corridor_dim_220lux")

					if prop:
						prop.visible = true
						prop.global_position = corridor_anchor
						prop.rotation_degrees = Vector3(0.0, 0.0 if light_mode == 0 else 45.0, 0.0)
						_position_camera_6dof(corridor_anchor, dist, cam_off, cam_h, 0.75)

					await _record_frame_and_ultra_metadata(
						abs_img_dir, abs_lbl_dir, frame_id, c["case_id"], c["name_vi"], c["name_en"],
						group_id, group_name, c["hazard"], true, # is_hard_negative = true
						[], [], # Ground Truth RỖNG
						0.0, "Đạo cụ y tế rỗng không có người", c.get("why", ""),
						dist, cam_off, cam_h, 0.0, 0
					)

	_apply_lighting_profile("fluorescent_standard_450lux")


## Sinh 300 frames cho ca động học (N5: 30 bước thời gian x 5 góc x 2 chiều cao camera)
func _generate_kinematics_case(
	abs_img_dir: String, abs_lbl_dir: String,
	c: Dictionary, group_id: String, group_name: String
) -> void:
	var actor: Node3D = _get_actor(c["actor"])
	var case_idx: int = 0

	for step in range(30):              # 30 bước thời gian (t0 -> t29, step 0.033s)
		var t_norm: float = float(step) / 29.0
		var pos: Vector3 = corridor_anchor
		var rot_y: float = 0.0

		if c["mode"] == "looming":
			# Tiến thẳng đối đầu từ xa 6.0m về 1.2m
			pos = corridor_anchor + Vector3(0.0, 0.0, lerpf(6.0, 1.2, t_norm))
			rot_y = 180.0
		elif c["mode"] == "pardon":
			# Khi tới gần (cự ly 2.0m), bước né ngang 0.55m
			var lateral_shift: float = 0.55 * clampf((t_norm - 0.3) / 0.4, 0.0, 1.0)
			pos = corridor_anchor + Vector3(lateral_shift, 0.0, lerpf(3.0, 1.3, t_norm))
			rot_y = 180.0
		elif c["mode"] == "doorway":
			# Chuyển hướng 90 độ qua cửa
			pos = clinic_door_anchor + Vector3(lerpf(-0.5, 0.8, t_norm), 0.0, 0.0)
			rot_y = lerpf(0.0, 90.0, t_norm)
		else: # corner
			# Lao ra từ góc vuông hành lang
			pos = reception_anchor + Vector3(lerpf(-1.5, 0.5, t_norm), 0.0, 0.0)
			rot_y = -90.0

		for cam_off in CAM_YAW_OFFSETS:    # 5 góc camera
			for cam_h in CAM_HEIGHTS:        # 2 chiều cao camera -> 30x5x2 = 300 frames
				case_idx += 1
				var frame_id: String = "%s_%04d" % [c["case_id"].replace(".", "_"), case_idx]
				_hide_all_entities()

				if actor:
					actor.visible = true
					actor.global_position = pos
					actor.rotation_degrees = Vector3(0.0, rot_y, 0.0)
					_position_camera_6dof(pos, 2.5, cam_off, cam_h, 0.90)

				await _record_frame_and_ultra_metadata(
					abs_img_dir, abs_lbl_dir, frame_id, c["case_id"], c["name_vi"], c["name_en"],
					group_id, group_name, c["hazard"], false,
					[actor] if actor else [], ["standing"],
					15.0, "Đối tượng chuyển động thời gian thực", c.get("why", ""),
					2.5, cam_off, cam_h, rot_y, step
				)


## Sinh 300 frames cho ca mật độ đa người (N6)
func _generate_crowd_case(
	abs_img_dir: String, abs_lbl_dir: String,
	c: Dictionary, group_id: String, group_name: String
) -> void:
	var case_idx: int = 0
	var actors: Array = []
	for aname in c["actors"]:
		actors.append(_get_actor(aname))

	for dist in DISTANCES:               # 15 cự ly
		for cam_off in CAM_YAW_OFFSETS:    # 5 góc camera
			for cam_h in CAM_HEIGHTS:        # 2 chiều cao camera
				for crowd_shift in [0.0, 0.15]: # 2 khoảng cách biến thiên -> 15x5x2x2 = 300 frames
					case_idx += 1
					var frame_id: String = "%s_%04d" % [c["case_id"].replace(".", "_"), case_idx]
					_hide_all_entities()

					var active_actors: Array = []
					var active_subcats: Array = []

					if c["mode"] == "seated_pair":
						var p1: Node3D = actors[0]; var p2: Node3D = actors[1]
						if p1 and p2:
							p1.visible = true; p2.visible = true
							p1.global_position = corridor_anchor + Vector3(-0.25, 0.06, 0.02)
							p2.global_position = corridor_anchor + Vector3(0.25 + crowd_shift, 0.06, 0.02)
							active_actors = [p1, p2]; active_subcats = c["subcats"]
					elif c["mode"] == "doctor_patient":
						var doc: Node3D = actors[0]; var pat: Node3D = actors[1]
						if doc and pat:
							doc.visible = true; pat.visible = true
							doc.global_position = corridor_anchor + Vector3(-0.40, 0.0, -0.10)
							pat.global_position = corridor_anchor + Vector3(0.15 + crowd_shift, 0.06, 0.02)
							active_actors = [doc, pat]; active_subcats = c["subcats"]
					elif c["mode"] == "queue_kiosk":
						var p1: Node3D = actors[0]; var p2: Node3D = actors[1]; var p3: Node3D = actors[2]
						var k: Node3D = _get_prop(obstacle_props, "queue_kiosk")
						if k: k.visible = true; k.global_position = corridor_anchor + Vector3(0.0, 0.0, 0.5)
						if p1 and p2 and p3:
							p1.visible = true; p2.visible = true; p3.visible = true
							p1.global_position = corridor_anchor + Vector3(0.0, 0.0, 0.0)
							p2.global_position = corridor_anchor + Vector3(crowd_shift, 0.0, -0.75)
							p3.global_position = corridor_anchor + Vector3(0.1, 0.0, -1.50)
							active_actors = [p1, p2, p3]; active_subcats = c["subcats"]
					elif c["mode"] == "facing_pair":
						var doc: Node3D = actors[0]; var nurse: Node3D = actors[1]
						if doc and nurse:
							doc.visible = true; nurse.visible = true
							doc.global_position = corridor_anchor + Vector3(-0.25, 0.0, 0.0)
							doc.rotation_degrees = Vector3(0.0, 180.0, 0.0) # quay lưng
							nurse.global_position = corridor_anchor + Vector3(0.25 + crowd_shift, 0.0, 0.0)
							nurse.rotation_degrees = Vector3(0.0, 0.0, 0.0) # quay mặt
							active_actors = [doc, nurse]; active_subcats = c["subcats"]
					else: # guided_walk
						var elder: Node3D = actors[0]; var fam: Node3D = actors[1]
						if elder and fam:
							elder.visible = true; fam.visible = true
							elder.global_position = corridor_anchor + Vector3(-0.20, 0.0, 0.0)
							fam.global_position = corridor_anchor + Vector3(0.20 + crowd_shift * 0.5, 0.0, -0.05)
							active_actors = [elder, fam]; active_subcats = c["subcats"]

					_position_camera_6dof(corridor_anchor, dist, cam_off, cam_h, 0.85)

					await _record_frame_and_ultra_metadata(
						abs_img_dir, abs_lbl_dir, frame_id, c["case_id"], c["name_vi"], c["name_en"],
						group_id, group_name, c["hazard"], false,
						active_actors, active_subcats,
						16.0, "Cụm nhiều người tương tác lâm sàng", c.get("why", ""),
						dist, cam_off, cam_h, 0.0, 0
					)


# ==============================================================================
# HÀM CHỤP FRAME, TÍNH BOUNDING BOX & XUẤT SIÊU METADATA 7 PHÂN HỆ
# ==============================================================================

func _record_frame_and_ultra_metadata(
	abs_img_dir: String,
	abs_lbl_dir: String,
	frame_id: String,
	case_id: String,
	name_vi: String,
	name_en: String,
	group_id: String,
	group_name: String,
	hazard_tier: String,
	is_hard_neg: bool,
	active_actors: Array,
	subcategories: Array,
	delta_e: float,
	who_desc: String,
	why_desc: String,
	distance_m: float,
	cam_offset_deg: float,
	cam_h: float,
	npc_yaw_deg: float,
	timestep: int
) -> void:
	if robot_camera:
		robot_camera.force_update_transform()
	for act in active_actors:
		if act:
			_force_update_subtree_transforms(act)

	await RenderingServer.frame_post_draw

	# 1. Lưu ảnh JPEG chất lượng cao
	var rel_img_path: String = "images/%s.jpg" % frame_id
	var full_img_path: String = abs_img_dir.path_join("%s.jpg" % frame_id)
	var img: Image = get_viewport().get_texture().get_image()
	img.save_jpg(full_img_path, 0.90)

	# 2. Tính toán Ground Truth Bounding Box
	var vp_size := Vector2(capture_resolution.x, capture_resolution.y)
	var gt_list: Array[Dictionary] = []
	var yolo_lines: PackedStringArray = []

	if not is_hard_neg:
		for i in range(active_actors.size()):
			var actor: Node3D = active_actors[i]
			if actor == null:
				continue
			var subcat: String = subcategories[i] if i < subcategories.size() else "standing"
			var gt_box: Dictionary = _compute_actor_ground_truth(actor, subcat, vp_size)
			if gt_box.get("valid", false):
				gt_box.erase("valid")
				gt_list.append(gt_box)

				# Chuyển sang định dạng YOLO chuẩn: 0 cx cy w h
				var xyxy: Array = gt_box["bbox_xyxy_px"]
				var cx: float = (xyxy[0] + xyxy[2]) * 0.5 / vp_size.x
				var cy: float = (xyxy[1] + xyxy[3]) * 0.5 / vp_size.y
				var bw: float = (xyxy[2] - xyxy[0]) / vp_size.x
				var bh: float = (xyxy[3] - xyxy[1]) / vp_size.y
				yolo_lines.append("0 %.6f %.6f %.6f %.6f" % [cx, cy, bw, bh])

	# 3. Xuất file nhãn YOLO
	var full_lbl_path: String = abs_lbl_dir.path_join("%s.txt" % frame_id)
	var lbl_file: FileAccess = FileAccess.open(full_lbl_path, FileAccess.WRITE)
	if lbl_file:
		if not yolo_lines.is_empty():
			lbl_file.store_string("\n".join(yolo_lines) + "\n")
		lbl_file.close()

	# 4. Tạo bản ghi Siêu Metadata 7 phân hệ
	var fov_h: float = robot_camera.fov if robot_camera else 75.0
	var cam_pos: Vector3 = robot_camera.global_position if robot_camera else Vector3.ZERO
	var cam_rot: Vector3 = robot_camera.rotation_degrees if robot_camera else Vector3.ZERO

	var rich_record: Dictionary = {
		"frame_id": frame_id,
		"case_id": case_id,
		"case_name_vi": name_vi,
		"case_name_en": name_en,
		"group_id": group_id,
		"group_name": group_name,
		"clinical_hazard_tier": hazard_tier,
		"is_hard_negative": is_hard_neg,

		"image_metadata": {
			"file_path": rel_img_path,
			"width": capture_resolution.x,
			"height": capture_resolution.y,
			"format": "JPEG",
			"color_space": "sRGB",
			"render_engine": "Godot_4_ForwardPlus_Vulkan"
		},

		"robot_camera_6dof": {
			"position_world": [snappedf(cam_pos.x, 0.001), snappedf(cam_pos.y, 0.001), snappedf(cam_pos.z, 0.001)],
			"rotation_euler_deg": [snappedf(cam_rot.x, 0.01), snappedf(cam_rot.y, 0.01), snappedf(cam_rot.z, 0.01)],
			"sensor_height_mode": "torso_0.85m" if cam_h < 1.0 else "eye_level_1.25m",
			"intrinsics": {
				"fov_horizontal_deg": snappedf(fov_h, 0.1),
				"fov_vertical_deg": snappedf(fov_h * 480.0 / 640.0, 0.1),
				"focal_length_px": snappedf((640.0 * 0.5) / tan(deg_to_rad(fov_h * 0.5)), 0.01),
				"principal_point_px": [320.0, 240.0]
			},
			"bearing_offset_deg": snappedf(cam_offset_deg, 0.1),
			"distance_to_target_m": snappedf(distance_m, 0.02)
		},

		"photometric_and_lighting": {
			"lighting_profile": "fluorescent_standard_450lux",
			"illuminance_lux_estimated": 450.0,
			"color_temperature_k": 6500,
			"delta_e_cie_lab_estimated": snappedf(delta_e, 0.1),
			"ambient_shadow_ratio": 0.15
		},

		"kinematics_and_temporal": {
			"is_dynamic": (group_id == "N5"),
			"temporal_timestep_idx": timestep,
			"timestamp_sec": snappedf(float(timestep) * 0.033, 0.001),
			"npc_yaw_deg": snappedf(npc_yaw_deg, 0.1),
			"bearing_angle_gt_deg": snappedf(cam_offset_deg, 0.1)
		},

		"entity_clinical_context": {
			"actor_count": active_actors.size(),
			"subcategories": subcategories
		},

		"ground_truth_annotations": gt_list,

		"context_5w1h": {
			"who": who_desc,
			"what": name_vi,
			"where": "Hành lang phòng khám bệnh viện đa khoa",
			"when": "Ca trực chuẩn ánh sáng huỳnh quang 450 lux",
			"why": why_desc,
			"how": "Cự ly %.1fm, góc ngắm %.1f deg, camera h=%.2fm" % [distance_m, cam_offset_deg, cam_h]
		}
	}

	var serialized: String = JSON.stringify(rich_record)
	_jsonl_file.store_line(serialized)
	_jsonl_rich_file.store_line(serialized)
	_total_saved_frames += 1

	if _total_saved_frames % 500 == 0:
		print("[ExtendedGenerator] Đã xuất: ", _total_saved_frames, " / 10,800 frames (", snappedf(float(_total_saved_frames) / 108.0, 0.1), "%)...")


# ==============================================================================
# HÀM TÍNH TOÁN GROUND TRUTH BOUNDING BOX OBB
# ==============================================================================

func _compute_actor_ground_truth(actor_root: Node3D, subcategory: String, vp_size: Vector2) -> Dictionary:
	var world_pts := PackedVector3Array()
	var h: float = 1.75
	var r: float = 0.28

	if subcategory == "wheelchair":
		h = 1.35
		r = 0.40
		for y_norm in [0.02, 0.15, 0.35, 0.55, 0.75, 0.90, 0.98]:
			var y_val: float = h * y_norm
			for theta in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
				var rad = deg_to_rad(theta)
				var cur_r = r if y_val < 0.65 else r * 0.75
				var z_offset = 0.12 if y_val < 0.65 else 0.0
				var lp = Vector3(cur_r * cos(rad), y_val, cur_r * sin(rad) + z_offset)
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

	var center_world: Vector3 = actor_root.global_position + Vector3(0.0, h * 0.5, 0.0)
	var cam_dist: float = robot_camera.global_position.distance_to(center_world) if robot_camera else 2.0
	var fov_rad: float = deg_to_rad(robot_camera.fov if robot_camera else 75.0)
	var theoretical_h: float = (h / (2.0 * cam_dist * tan(fov_rad * 0.5))) * vp_size.y
	var vis_ratio: float = clampf(bh / maxf(theoretical_h, 1.0), 0.05, 1.0)

	var cx: float = (min_x + max_x) * 0.5
	var head_y: float = min_y
	var foot_y: float = max_y

	return {
		"valid": true,
		"track_id": 1,
		"category_id": 0,
		"category_name": "person",
		"subcategory": subcategory,
		"bbox_xyxy_px": [snappedf(min_x, 0.1), snappedf(min_y, 0.1), snappedf(max_x, 0.1), snappedf(max_y, 0.1)],
		"aspect_ratio": snappedf(bh / maxf(bw, 1.0), 0.01),
		"area_px": snappedf(bw * bh, 0.1),
		"euclidean_distance_m": snappedf(cam_dist, 0.02),
		"visibility_ratio": snappedf(vis_ratio, 0.01),
		"head_center_2d_px": [snappedf(cx, 0.1), snappedf(head_y, 0.1)],
		"foot_ground_2d_px": [snappedf(cx, 0.1), snappedf(foot_y, 0.1)]
	}


# ==============================================================================
# HÀM PHỤ TRỢ POSE VÀ ÁNH SÁNG
# ==============================================================================

func _position_camera_6dof(
	anchor: Vector3,
	dist: float,
	yaw_offset_deg: float,
	cam_height: float,
	target_look_height: float
) -> void:
	if robot_camera == null:
		return
	var target: Vector3 = anchor + Vector3(0.0, target_look_height, 0.0)
	var rad = deg_to_rad(yaw_offset_deg)
	var cam_x: float = anchor.x + dist * sin(rad)
	var cam_z: float = anchor.z + dist * cos(rad)
	robot_camera.global_position = Vector3(cam_x, cam_height, cam_z)
	robot_camera.look_at(target, Vector3.UP)


func _hide_all_entities() -> void:
	if actors_pool:
		for child in actors_pool.get_children():
			if child is Node3D:
				child.visible = false
	if obstacle_props:
		for child in obstacle_props.get_children():
			if child is Node3D:
				child.visible = false
	if hard_negatives_pool:
		for child in hard_negatives_pool.get_children():
			if child is Node3D:
				child.visible = false


func _get_actor(actor_name: String) -> Node3D:
	if actors_pool and actors_pool.has_node(actor_name):
		return actors_pool.get_node(actor_name) as Node3D
	return null


func _get_prop(pool: Node3D, prop_name: String) -> Node3D:
	if pool and pool.has_node(prop_name):
		return pool.get_node(prop_name) as Node3D
	return null


func _apply_lighting_profile(profile_name: String) -> void:
	if main_dir_light == null:
		return
	if profile_name == "fluorescent_standard_450lux":
		main_dir_light.light_color = Color(0.95, 0.96, 1.0)
		main_dir_light.light_energy = 1.0
	elif profile_name == "corridor_dim_220lux":
		main_dir_light.light_color = Color(0.85, 0.85, 0.90)
		main_dir_light.light_energy = 0.55


func _force_update_subtree_transforms(node: Node3D) -> void:
	if node == null:
		return
	node.force_update_transform()
	for child in node.get_children():
		if child is Node3D:
			_force_update_subtree_transforms(child)


func _write_dataset_manifest(abs_out_dir: String, elapsed_sec: float) -> void:
	var manifest: Dictionary = {
		"dataset_title": "Hospital Clinical Demographics & Kinematics Big Data Vision Benchmark",
		"total_frames": _total_saved_frames,
		"expected_frames": 10800,
		"case_count": 36,
		"frames_per_case": 300,
		"group_distribution": {
			"N1_Demographics_Attire": 2100,
			"N2_Clinical_Poses": 1500,
			"N3_Prop_Occlusion": 2100,
			"N4_Hard_Negatives": 2400,
			"N5_Kinematics_Pardon": 1200,
			"N6_Crowd_Social": 1500
		},
		"resolution": [capture_resolution.x, capture_resolution.y],
		"format": "JPEG",
		"generation_time_sec": snappedf(elapsed_sec, 0.1),
		"seed": random_seed
	}
	var f: FileAccess = FileAccess.open(abs_out_dir.path_join("metadata.json"), FileAccess.WRITE)
	if f:
		f.store_string(JSON.stringify(manifest, "\t"))
		f.close()
