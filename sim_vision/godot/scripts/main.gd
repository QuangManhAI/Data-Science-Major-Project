class_name Main
extends Node3D
## Scene chính (G8): vision_client ➔ controller.step ➔ apply_safety ➔ robot.set_command.
## Hội thoại (G16): ở LISTEN / THINK / RESPOND robot đứng yên (v = 0, omega = 0), không gọi controller.
## Phím R đặt lại vị trí ngẫu nhiên, C đổi góc nhìn camera, M bật/tắt tiếng ồn bệnh viện, Esc thoát.

## Phát sau mỗi chu kỳ nhận diện (sau apply_safety); G9 ghi log từ đây.
signal cycle_completed(info: Dictionary)

enum CameraMode { FOLLOW, TOP_DOWN, LOBBY }

const START_CLEARANCE: float = 0.6      # G8.1: get_free_position(0.6)
const MIN_START_SEPARATION: float = 3.0 # G8.1: robot cách người ≥ 3 m
const DOCK_START_SEPARATION: float = 3.5  # đầu ca: bệnh nhân cách trạm robot ≥ 3.5 m (như env lúc train)
const MAX_PLACEMENT_TRIES: int = 100
const RLEnvBridgeScript = preload("res://scripts/rl_env_bridge.gd")
const RLControllerScript = preload("res://scripts/rl_controller.gd")
const RLPolicyClientScript = preload("res://scripts/rl_policy_client.gd")
const InspectionPanelScript = preload("res://scripts/inspection_panel.gd")
const PATIENT_ENGAGE_DIST_M: float = 1.85  # Khoảng cách tiếp cận thực tế đến vị trí bệnh nhân để bắt đầu hội thoại
const TASK_TIMEOUT_S: float = 45.0         # = stagnation_timeout_s lúc train: không tiến triển quá lâu ➔ hủy
const TASK_PROGRESS_M: float = 1.0         # = STAGNATION_PROGRESS_M: lại gần đích thêm 1 m ➔ đặt lại đồng hồ
const ARRIVED_WAIT_S: float = 10.0         # đứng ARRIVED mà NPC không bắt đầu nói ➔ đi tiếp

@warning_ignore("shadowed_global_identifier")
@export var seed: int = -1              # −1 = ngẫu nhiên
@export var controller_path: NodePath = ^"Controller"
@export var use_rl_controller: bool = false
var solo_patient_mode: bool = false
var escort_only_mode: bool = false
const ESCORT_FALLBACK_TARGET: String = "tiep_don"

var state: int = Config.RobotState.SEARCH
var camera_mode: int = CameraMode.FOLLOW
var camera_orbit_angle: float = 0.0
const CAMERA_ORBIT_SPEED: float = 2.0

var _rng := RandomNumberGenerator.new()
var _controller: Node
var _rl_bridge: Node = null
var _is_in_elevator_transit: bool = false
var _policy: Node = null                # RLPolicyClient; null khi không ở chế độ RL
var _policy_accum: float = 0.0
var _task_elapsed_s: float = 0.0
var _task_best_dist: float = -1.0
var _arrived_wait_s: float = 0.0
var _escort_debug_timer: float = 0.0
var _connected: bool = false
var _last_seen_ms: int = -1             # −1 = chưa thấy người lần nào
var _last_result_ms: int = -1
var _start_ms: int = 0
var _latest_yolo_dets: Array = []

@onready var room: Room = $Room
@onready var human: Human = $Human
var active_escorted_patient: CharacterBody3D = null
var current_target_patient: CharacterBody3D = null
var _pending_turn_reset: bool = false
## Lượt bệnh nhân cảm ơn khi tới nơi: không tạo nhiệm vụ / không cộng thưởng; hết lượt (về ARRIVED) ➔ _after_thanks.
var _thanks_pending: bool = false
var _eval_min_clear_m: float = INF
var _eval_bump_frames: int = 0
## Âm thanh nền bệnh viện (chỉ cho người xem; STT tự trộn nhiễu ở server). Tạo bằng gen_hospital_ambience.py.
const AMBIENCE_PATH: String = "res://assets/sounds/hospital_ambience.wav"
const MONITOR_BEEP_PATH: String = "res://assets/sounds/monitor_beep.wav"
const AMBIENCE_DB: float = -8.0        # file đã chuẩn hóa đỉnh 0.89; giọng NPC / robot vẫn to hơn nền ~8 dB
const MONITOR_BEEP_DB: float = -6.0
var _ambience_on: bool = true
var _ambience_players: Array[Node] = []
## Trạm thang máy camera FOLLOW đang nhìn vào trong lúc đi thang (null = bám robot như thường).
var _elev_cam_station: ElevatorStation = null
const ELEV_ROBOT_SPEED: float = 0.7      # m/s khi vào / ra thang máy
const ELEV_PATIENT_SPEED: float = 0.75
const ELEV_PATIENT_DELAY_S: float = 0.6  # bệnh nhân bước theo sau robot
var _after_thanks: Callable = Callable()

# --- Tương tác khi dẫn đường (docs/UPGRADE_HRI_PLAN.md mục A–C) ---
const PATIENT_LAG_PROB: float = 0.5               # xác suất bệnh nhân tụt lại một lần trong ca (mô phỏng); --lag-prob=x
const LAG_START_RANGE_S: Vector2 = Vector2(6.0, 20.0)
const LAG_DURATION_RANGE_S: Vector2 = Vector2(6.0, 30.0)
const REST_HOLD_S: Vector2 = Vector2(3.0, 5.0)     # bệnh nhân xin nghỉ: đứng nghỉ chừng này (mô phỏng)
const REMIND_HEED_PROB: float = 0.6               # nghe nhắc thì bước tiếp ngay
const ANSWER_WEIGHTS: Dictionary = {"tiep_tuc": 0.7, "dung_lai": 0.2, "cam_on": 0.1}   # trả lời "có dẫn tiếp không"
const GOAL_ANNOUNCE_MAX_M: float = 7.0
const GOAL_ANNOUNCE_MIN_M: float = 2.0
const GOAL_ANNOUNCE_MAX_ANGLE: float = deg_to_rad(100.0)
const HOLD_TURN_GAIN: float = 1.8
var patient_lag_prob: float = PATIENT_LAG_PROB
## Dán ảnh mặt người thật lên đầu NPC (npc_face.gd) khi có texture trong assets/textures/npc_faces/; --no-face-paste tắt.
var face_paste: bool = true
const FACE_LOOK_ALIASES: Dictionary = {"medical_m01": "medical_doctor", "medical_f01": "medical_nurse"}
var announcer: RobotAnnouncer = null
var escort_monitor: EscortMonitor = EscortMonitor.new()
var _ask_pending: bool = false
var _escort_t: float = 0.0
var _lag_at_s: float = -1.0
var _lag_dur_s: float = 0.0
var _goal_announced_key: String = ""
var _after_line: Dictionary = {}   # key câu thoại ➔ Callable gọi khi robot nói xong câu đó

# --- Nhận diện đúng bệnh nhân bằng camera (docs/UPGRADE_HRI_PLAN.md D3–D4) ---
const ESCORT_GRACE_S: float = 4.0                 # đầu ca: bệnh nhân đang vòng ra phía trước robot, chưa xét tụt lại
const ENROLL_WAIT_S: float = 4.0                  # chưa ghi nhớ được mặt ➔ đứng chờ tối đa chừng này, nhờ nhìn về robot
const SPEAKER_TURN_GAIN: float = 2.0
const ID_SEEN_S: float = 0.5                      # khớp trong 0.5 s gần nhất ➔ đang thấy
const ID_ASSUME_FAR_S: float = 4.0                # không thấy quá 4 s ➔ coi như đã xa (lúc rẽ bệnh nhân hay ra khỏi khung 1–3 s)
const SEARCH_TURN_OMEGA: float = 0.6
const ROBOT_CAM_HFOV: float = 2.0 * atan(tan(deg_to_rad(35.0)) * 4.0 / 3.0)   # FOV dọc 70°, 4:3 ➔ ngang ~86°
var identity: PatientIdentity = PatientIdentity.new()
var _reid_supported: bool = false                 # yolo_server có trả embedding (có model, không --no-reid)
var _id_det: Dictionary = {}                      # box của bệnh nhân mình trong frame mới nhất ({} = không thấy)
## Người vừa dẫn tới nơi: giữ mẫu ReID của họ thêm SERVED_FORGET_S (chỉ trong RAM, rồi xoá) để không coi họ là người
## cần giúp mới (robot đứng cạnh họ sau khi tới nơi, box của họ to giữa khung ➔ policy hỏi lại "Cần tôi giúp gì").
const SERVED_FORGET_S: float = 30.0
var _served_id: PatientIdentity = PatientIdentity.new()
var _served_until_s: float = -INF
var _served_det: Dictionary = {}
var _id_seen_s: float = -INF
var _id_dist_m: float = 1.0
var _id_est_pos: Vector3 = Vector3.ZERO           # vị trí ước lượng từ camera lần cuối thấy
var _id_stats: Dictionary = {}                    # số liệu D5 (log)
const REAR_CAPTURE_HZ: float = 10.0
var rear_vision: VisionClient = null              # camera sau (chỉ khi dẫn đường)
var _latest_rear_dets: Array = []
var _id_rear_det: Dictionary = {}
var _asked_look: bool = false
var _enroll_misses: int = 0                       # frame lúc nói chuyện có box patient mà không ghi nhớ được
var _enroll_reasons: Dictionary = {}
const FACE_CROP_EVERY_MS: int = 66                # ô "MẶT": cắt mặt đang thấy ~15 lần / giây
var _last_face_crop_ms: int = -100000
@onready var robot: Robot = $Robot
@onready var vision: VisionClient = $VisionClient
@onready var overlay: ViewOverlay = $UI/ViewOverlay
@onready var inspection_panel: Node = get_node_or_null(^"UI/InspectionPanel")
@onready var help_label: Label = get_node_or_null(^"UI/Help")
@onready var viewer_camera: Camera3D = get_node_or_null(^"ViewerCamera")
@onready var ceiling_mesh: Node3D = get_node_or_null(^"Room/Shell/Ceiling")
@onready var logger: RunLogger = get_node_or_null(^"RunLogger")  # G9; không có thì không ghi log
@onready var dialogue: DialogueManager = get_node_or_null(^"DialogueManager")  # G16; không có thì không hội thoại

var _interior_walls: Array[MeshInstance3D] = []
var _occluding_walls: Array[MeshInstance3D] = []
var _transparent_wall_mat: StandardMaterial3D
var _human_home_pos: Vector3 = Vector3.ZERO
var _human_home_yaw: float = 0.0


func _ready() -> void:
	var all_intents_flag: bool = false
	for arg: String in OS.get_cmdline_user_args() + OS.get_cmdline_args():
		if arg == "--controller=rl" or arg == "--rl":
			use_rl_controller = true
		elif arg in ["--solo-patient", "--solo_patient", "--solo"]:
			solo_patient_mode = true
		elif arg in ["--escort-only", "--escort_only", "--escort"]:
			escort_only_mode = true
		elif arg in ["--all-intents", "--all_intents"]:
			all_intents_flag = true
		elif arg in ["--no-face-paste", "--no_face_paste"]:
			face_paste = false
		elif arg.begins_with("--lag-prob="):
			patient_lag_prob = clampf(arg.get_slice("=", 1).to_float(), 0.0, 1.0)
	if use_rl_controller and not all_intents_flag:
		escort_only_mode = true
	if use_rl_controller:
		var rl_ctrl: Node = RLControllerScript.new()
		rl_ctrl.name = "RLController"
		add_child(rl_ctrl)
		_controller = rl_ctrl
		_rl_bridge = RLEnvBridgeScript.new()
		_rl_bridge.name = "RLEnvBridge"
		add_child(_rl_bridge)
		_rl_bridge.setup(robot, room)
		_policy = RLPolicyClientScript.new()
		_policy.name = "RLPolicyClient"
		add_child(_policy)
		_policy.action_received.connect(_on_policy_action)
		_policy.engage_received.connect(_on_policy_engage)
		_policy.sectors_received.connect(overlay.set_visual_sectors)
		if inspection_panel != null:
			_policy.sectors_received.connect(inspection_panel.update_visual_sectors)
		if dialogue != null:
			dialogue.turn_completed.connect(_on_dialogue_turn_completed)
			dialogue.clarify_gave_up.connect(_on_clarify_gave_up)
	else:
		_controller = get_node(controller_path)
	if inspection_panel != null:
		inspection_panel.set_pure_vision_mode(true)
	if seed >= 0:
		_rng.seed = seed
	else:
		_rng.randomize()
	vision.setup(robot.get_camera_viewport())
	overlay.set_viewport(robot.get_camera_viewport())
	_setup_rear_vision()
	vision.detections_received.connect(_on_detections)
	vision.connection_changed.connect(_on_connection_changed)
	_start_ms = Time.get_ticks_msec()
	if logger != null:
		logger.open()
	if human != null:
		_human_home_pos = human.global_position
		_human_home_yaw = human.rotation.y
		if _human_home_pos == Vector3.ZERO:
			_human_home_pos = Vector3(4.5, 0.0, -1.35)
			human.global_position = _human_home_pos
		if human.waypoints.is_empty():
			human.waypoints = [Vector3(4.5, 0.0, -1.35), Vector3(7.5, 0.0, -1.35), Vector3(4.5, 0.0, -1.35), Vector3(2.5, 0.0, -1.35)]
			human.walks = true
			human.pardon_enabled = true
	_setup_hospital_npcs()
	if escort_only_mode:
		_apply_escort_only_mode()
	_setup_announcer()
	if face_paste:
		_apply_face_paste()
	reset_positions()
	if viewer_camera != null:
		RenderLayers.setup_viewer_camera(viewer_camera)
	_update_viewer_camera()
	_setup_hospital_ambience()


func _unhandled_input(event: InputEvent) -> void:
	var key := event as InputEventKey
	if key == null or not key.pressed or key.echo:
		return
	if key.keycode == KEY_R:
		reset_positions()
	elif key.keycode == KEY_C:
		camera_mode = (camera_mode + 1) % 3
		_update_viewer_camera()
	elif key.keycode == KEY_A and not key.echo:
		camera_orbit_angle += 0.25
		_update_viewer_camera()
	elif key.keycode == KEY_D and not key.echo:
		camera_orbit_angle -= 0.25
		_update_viewer_camera()
	elif key.keycode == KEY_G and not key.echo:
		trigger_test_escort()
	elif key.keycode == KEY_M:
		set_ambience_enabled(not _ambience_on)
	elif key.keycode == KEY_ESCAPE:
		get_tree().quit()


func trigger_test_escort(dest_key: String = "") -> void:
	if _rl_bridge == null:
		return
	var dest_pool: Array[String] = ["110", "112", "nha_thuoc", "201", "204", "205", "210", "220", "301", "302", "303", "315"]
	var target: String = dest_key if dest_key != "" else dest_pool[_rng.randi_range(0, dest_pool.size() - 1)]
	var fl: int = Room.get_room_floor(target)
	var pat: CharacterBody3D = active_escorted_patient if active_escorted_patient != null else (current_target_patient if current_target_patient != null else human)
	active_escorted_patient = pat
	current_target_patient = pat
	if active_escorted_patient != null and active_escorted_patient.has_method("set_state"):
		active_escorted_patient.set_state(HospitalNpc.State.ESCORTED)
	_rl_bridge.begin_post_dialogue_task("guide", target, fl, active_escorted_patient)
	_begin_escort_hri()
	_task_elapsed_s = 0.0
	_task_best_dist = -1.0
	state = Config.RobotState.APPROACH
	overlay.set_status("Dẫn đường test (Phím G): tới %s (Tầng %d)" % [_rl_bridge.goal_key, fl])


func _process(delta: float) -> void:
	if Input.is_key_pressed(KEY_A):
		camera_orbit_angle += CAMERA_ORBIT_SPEED * delta
	elif Input.is_key_pressed(KEY_D):
		camera_orbit_angle -= CAMERA_ORBIT_SPEED * delta
	_update_viewer_camera()
	_update_inspection_panel()
	_update_vision_modes()
	# Quá LOST_TIMEOUT_S không có kết quả mới (nhưng vẫn kết nối) ➔ coi như không thấy ai.
	var now: int = Time.get_ticks_msec()
	if Main.is_dialogue_state(state):
		return
	if _rl_bridge != null and _rl_bridge.task_phase in [RLEnvBridgeScript.TaskPhase.GUIDE_ESCORT, RLEnvBridgeScript.TaskPhase.CALL_STAFF]:
		return
	if _policy_active():
		return
	if _connected and _last_result_ms >= 0 and now - _last_result_ms > int(Config.LOST_TIMEOUT_S * 1000.0):
		_apply(_controller.step(state, [], _time_since_seen_s(now)))


func _physics_process(delta: float) -> void:
	if _rl_bridge == null:
		return
	if _is_in_elevator_transit:
		return
	if Main.is_dialogue_state(state) and (not policy_v3 or _escort_holding()):
		_face_speaker()                   # luật (baseline V2); V3 tự học việc này, trừ lúc giữ dẫn (policy không chạy)

	# Số liệu đánh giá (scripts/eval_godot_escort.py): khoảng trống nhỏ nhất (LiDAR, m) và số frame robot va vào
	# tường / đồ đạc kể từ dòng log trước — đo mỗi physics frame để không sót cú quẹt ngắn.
	var lid: PackedFloat32Array = robot.get_lidar_readings()
	for r: float in lid:
		_eval_min_clear_m = minf(_eval_min_clear_m, r * Robot.LIDAR_MAX_RANGE)
	for ci: int in robot.get_slide_collision_count():
		if absf(robot.get_slide_collision(ci).get_normal().y) < 0.5:
			_eval_bump_frames += 1
			break

	# Log vị trí & hành vi robot + bệnh nhân mỗi 1.0 giây
	_escort_debug_timer += delta
	if _escort_debug_timer >= 1.0:
		_escort_debug_timer = 0.0
		var r_pos: Vector3 = robot.global_position if robot != null else Vector3.ZERO
		var r_cmd: Vector2 = robot.get_command() if robot != null else Vector2.ZERO
		var r_vel: Vector3 = robot.get_real_velocity() if robot != null else Vector3.ZERO
		var esc_p: CharacterBody3D = active_escorted_patient if active_escorted_patient != null else (current_target_patient if current_target_patient != null else human)
		var p_pos: Vector3 = esc_p.global_position if esc_p != null else Vector3.ZERO
		var d_tether: float = Human.flat_distance(r_pos, p_pos)
		var ph_name: String = "SEARCH"
		if _rl_bridge != null:
			match _rl_bridge.task_phase:
				RLEnvBridgeScript.TaskPhase.SEARCH_PATIENT: ph_name = "SEARCH"
				RLEnvBridgeScript.TaskPhase.APPROACH_PATIENT: ph_name = "APPROACH"
				RLEnvBridgeScript.TaskPhase.GUIDE_ESCORT: ph_name = "GUIDE_ESCORT"
				RLEnvBridgeScript.TaskPhase.CALL_STAFF: ph_name = "CALL_STAFF"
		var g_key: String = _rl_bridge.goal_key if _rl_bridge != null else "none"
		var g_pos: Vector3 = _rl_bridge.goal_position if _rl_bridge != null else Vector3.ZERO
		var d_goal: float = Human.flat_distance(r_pos, g_pos)
		var d_p_goal: float = Human.flat_distance(p_pos, g_pos)
		var trail_count: int = _rl_bridge.escort_trail.size() if _rl_bridge != null else 0
		print("[ESCORT_DEBUG] t=%5.1fs | State=%-8s | Phase=%-12s | Robot=(%5.2f, %5.2f) cmd=(v=%.2f, w=%+5.2f) vel=%4.2f | Patient=(%5.2f, %5.2f) tether=%4.2fm | Goal=%s (dist_r=%4.2fm, dist_p=%4.2fm) | Trail=%d" % [
			(Time.get_ticks_msec() - _start_ms) / 1000.0,
			Config.state_name(state), ph_name,
			r_pos.x, r_pos.z, r_cmd.x, r_cmd.y, r_vel.length(),
			p_pos.x, p_pos.z, d_tether,
			g_key, d_goal, d_p_goal,
			trail_count
		] + " | clear_min=%.2fm bump=%d floor=%d" % [_eval_min_clear_m, _eval_bump_frames, Room.get_floor(r_pos)])
		_eval_min_clear_m = INF
		_eval_bump_frames = 0

	# Đồng hồ kẹt áp dụng cả pha tìm bệnh nhân (đích = bệnh nhân), trừ lúc hội thoại / đã tới nơi chờ NPC nói
	if not Main.is_dialogue_state(state) and state != Config.RobotState.ARRIVED:
		_task_elapsed_s += delta
		var current_pat: CharacterBody3D = active_escorted_patient if active_escorted_patient != null else (current_target_patient if current_target_patient != null else human)
		var pat_target_pos: Vector3 = current_pat.global_position if active_escorted_patient != null else Main.get_patient_interaction_pos(current_pat)
		var task_target: Vector3 = pat_target_pos if _is_search_phase() else _rl_bridge.goal_position
		var d_task: float = Human.flat_distance(robot.global_position, _leg_target(task_target))
		if _task_best_dist < 0.0 or d_task < _task_best_dist - TASK_PROGRESS_M:
			_task_best_dist = d_task
			_task_elapsed_s = 0.0

		var cur_floor_now: int = Room.get_floor(robot.global_position)
		# KHÔNG tự reset episode khi robot đang tìm kiếm / quay về ở tầng trên (cur_floor > 1)
		# Robot có thể tự tìm đường quay về thang máy hoặc tìm bệnh nhân tầng trên mà không bị cưỡng chế reset
		if cur_floor_now > 1 and _is_search_phase():
			_task_elapsed_s = 0.0
		elif _policy_active():
			# Trong chế độ RL inference: không ép reset vị trí làm gián đoạn thử nghiệm
			if _task_elapsed_s > TASK_TIMEOUT_S:
				_task_elapsed_s = 0.0
				print("[ESCORT_DEBUG] >>> RL Inference: timeout suppressed to keep policy exploring without auto-reset.")
		elif _task_elapsed_s > TASK_TIMEOUT_S:
			print("[ESCORT_DEBUG] >>> Task TIMED OUT! elapsed=%.1fs > %.1fs, Goal=%s" % [_task_elapsed_s, TASK_TIMEOUT_S, _rl_bridge.goal_key])
			overlay.set_status("Hủy nhiệm vụ %s: kẹt quá %.0fs" % [_rl_bridge.goal_key, TASK_TIMEOUT_S])
			# Như kết thúc episode lúc train: đặt lại cả robot; chỉ dời bệnh nhân thì robot kẹt góc quay vòng mãi
			reset_positions(false)
			return
	if _rl_bridge.task_phase == RLEnvBridgeScript.TaskPhase.GUIDE_ESCORT:
		var has_escort: bool = active_escorted_patient != null
		if has_escort:
			RLEnvBridgeScript.step_escorted_patient(active_escorted_patient, robot.global_position, delta, _rl_bridge.escort_trail, robot.rotation.y)
		var d_goal: float = Human.flat_distance(robot.global_position, _rl_bridge.goal_position)
		var esc_p: CharacterBody3D = active_escorted_patient if has_escort else human
		var d_tether: float = Human.flat_distance(robot.global_position, esc_p.global_position)
		var d_patient_goal: float = Human.flat_distance(esc_p.global_position, _rl_bridge.goal_position) if has_escort else 0.0
		if has_escort and not _is_in_elevator_transit:
			_update_escort_hri(esc_p, d_tether, delta)

		# Tự động mở cửa thang máy khi robot tiến lại gần (d <= 3.2m)
		if _rl_bridge.goal_key in ["thang_may", "thang_may_t2", "thang_may_t3"]:
			var cur_fl_app: int = Room.get_floor(robot.global_position)
			var cur_st_app: ElevatorStation = room.get_elevator_station(cur_fl_app) if room != null else null
			if cur_st_app != null and not cur_st_app.is_open and Human.flat_distance(robot.global_position, cur_st_app.global_position) <= 3.2:
				print("[ESCORT_DEBUG] >>> Approaching elevator (d <= 3.2m), opening station doors...")
				cur_st_app.open_doors()

		if d_goal <= RLEnvBridgeScript.GOAL_ARRIVE_ROBOT_M and (not has_escort or d_patient_goal <= RLEnvBridgeScript.GOAL_ARRIVE_PATIENT_M):
			# Kiểm tra chuyển tầng qua thang máy
			if _rl_bridge.goal_key in ["thang_may", "thang_may_t2", "thang_may_t3"] and _rl_bridge.final_target_key != "" and _rl_bridge.final_target_key != _rl_bridge.goal_key:
				var cur_fl: int = Room.get_floor(robot.global_position)
				var dest_fl: int = Room.get_room_floor(_rl_bridge.final_target_key)
				print("[ESCORT_DEBUG] >>> Elevator transit triggered: Floor %d -> Floor %d (FinalTarget: %s, with_patient: %s)" % [cur_fl, dest_fl, _rl_bridge.final_target_key, has_escort])
				_execute_elevator_transit(cur_fl, dest_fl, has_escort)
				return
			print("[ESCORT_DEBUG] >>> Arrived at escort goal '%s'! d_goal=%.2fm, d_patient_goal=%.2fm" % [_rl_bridge.goal_key, d_goal, d_patient_goal])
			robot.set_command(0.0, 0.0)
			_rl_bridge.cumulative_team_reward += RLEnvBridgeScript.REWARD_GUIDE_ARRIVED
			_rl_bridge.patients_helped += 1
			_rl_bridge.task_phase = RLEnvBridgeScript.TaskPhase.SEARCH_PATIENT
			overlay.set_status("Đã dẫn tới: %s (R_team=%.0f)" % [_rl_bridge.goal_key, _rl_bridge.cumulative_team_reward])
			if has_escort and esc_p.has_method("end_escort"):
				esc_p.end_escort()
			if has_escort:
				# Lúc chọn bệnh nhân, pardon bị tắt (đứng chờ robot tới) mà trước đây không bật lại ➔ người vừa
				# giao đứng chắn cửa không bao giờ nhường đường, robot (không lùi) kẹt mãi (log run_1335, run_1427).
				esc_p.set("pardon_enabled", true)
				esc_p.set_meta("helped", true)
				_log_id_stats("tới nơi")
				_remember_served()
			active_escorted_patient = null
			var cur_floor: int = Room.get_floor(robot.global_position)
			# Robot chỉ vào phòng và báo đã tới; nói xong bệnh nhân mới cảm ơn.
			if has_escort and _announce_arrival(esc_p, cur_floor):
				return
			_thanks_or_finish(esc_p if has_escort else null, cur_floor)
			return
		if not _policy_active():
			robot.set_command(0.0, 0.0)
			overlay.set_status("LỖI: Mất kết nối tới RL Policy Server (ws://127.0.0.1:8768)!")
			push_error("[Main] ERROR: RL Policy Server is disconnected or inactive! Robot halted (rule-based fallback disabled).")
			return
	elif _rl_bridge.task_phase == RLEnvBridgeScript.TaskPhase.CALL_STAFF:
		# Nhân viên ở tầng khác (tầng 2 không có nhân viên) ➔ robot đi thang máy một mình, bệnh nhân chờ tại chỗ
		var robot_floor: int = Room.get_floor(robot.global_position)
		var staff_floor: int = Room.get_floor(_rl_bridge.goal_position)
		if staff_floor != robot_floor:
			var cur_st_app: ElevatorStation = room.get_elevator_station(robot_floor) if room != null else null
			if cur_st_app != null and not cur_st_app.is_open and Human.flat_distance(robot.global_position, cur_st_app.global_position) <= 3.2:
				print("[ESCORT_DEBUG] >>> Approaching elevator for solo transit (d <= 3.2m), opening station doors...")
				cur_st_app.open_doors()
		if staff_floor != robot_floor \
				and Human.flat_distance(robot.global_position, _leg_target(_rl_bridge.goal_position)) <= RLEnvBridgeScript.GOAL_ARRIVE_ROBOT_M:
			print("[ESCORT_DEBUG] >>> Elevator transit triggered: Floor %d -> Floor %d (Solo Return)" % [robot_floor, staff_floor])
			_execute_elevator_transit(robot_floor, staff_floor, false)
			return
		var d_goal: float = Human.flat_distance(robot.global_position, _rl_bridge.goal_position)
		if d_goal <= RLEnvBridgeScript.GOAL_ARRIVE_ROBOT_M:
			robot.set_command(0.0, 0.0)
			_rl_bridge.cumulative_team_reward += RLEnvBridgeScript.REWARD_CALL_STAFF
			_rl_bridge.patients_helped += 1
			_rl_bridge.task_phase = RLEnvBridgeScript.TaskPhase.SEARCH_PATIENT
			overlay.set_status("Đã báo nhân viên y tế (R_team=%.0f)" % _rl_bridge.cumulative_team_reward)
			reset_positions(true)
			return
		if not _policy_active():
			robot.set_command(0.0, 0.0)
			overlay.set_status("LỖI: Mất kết nối tới RL Policy Server (ws://127.0.0.1:8768)!")
			push_error("[Main] ERROR: RL Policy Server is disconnected or inactive! Robot halted (rule-based fallback disabled).")
			return
	if _policy_active() and not _escort_holding():
		_step_policy(delta)


## Đích của chặng hiện tại: đích ở tầng khác ➔ thang máy tầng robot đang đứng (như env lúc train).
func _leg_target(target: Vector3) -> Vector3:
	var robot_floor: int = Room.get_floor(robot.global_position)
	if room == null or Room.get_floor(target) == robot_floor:
		return target
	var station_key: String = "thang_may" if robot_floor == 1 else ("thang_may_t2" if robot_floor == 2 else "thang_may_t3")
	return room.get_room_waypoint(station_key)


func _policy_active() -> bool:
	return _policy != null and _policy.is_server_connected()


func is_task_busy() -> bool:
	if active_escorted_patient != null:
		return true
	if _rl_bridge != null and _rl_bridge.task_phase in [
		RLEnvBridgeScript.TaskPhase.GUIDE_ESCORT,
		RLEnvBridgeScript.TaskPhase.CALL_STAFF
	]:
		return true
	return false



func get_active_speaking_patient() -> CharacterBody3D:
	if dialogue != null and dialogue.active_npc != null:
		var p: Node = dialogue.active_npc.get_parent()
		if p is CharacterBody3D:
			return p as CharacterBody3D
	return current_target_patient if current_target_patient != null else human


static func get_patient_interaction_pos(pat: CharacterBody3D) -> Vector3:
	if pat == null:
		return Vector3.ZERO
	return pat.global_position


func get_available_patients() -> Array[CharacterBody3D]:
	var list: Array[CharacterBody3D] = []
	if solo_patient_mode:
		if human != null and not human.get_meta("helped", false):
			list.append(human)
		elif human != null:
			human.set_meta("helped", false)
			list.append(human)
		return list
	var furn: Node = room.get_node_or_null(^"Furniture") if room != null else null
	if furn != null:
		for child: Node in furn.get_children():
			if child is HospitalNpc and (child as HospitalNpc).staff_role == "patient" and child.visible and not child.get_meta("helped", false):
				list.append(child as CharacterBody3D)
	if human != null and human.visible and not human.get_meta("helped", false):
		list.append(human)
	if list.is_empty():
		# Tất cả bệnh nhân đã được phục vụ -> làm mới để bệnh viện vận hành liên tục
		if furn != null:
			for child: Node in furn.get_children():
				if child is HospitalNpc and (child as HospitalNpc).staff_role == "patient" and child.visible:
					child.set_meta("helped", false)
					list.append(child as CharacterBody3D)
		if human != null and human.visible:
			human.set_meta("helped", false)
			list.append(human)
	return list


func pick_next_target_patient(exclude: CharacterBody3D = null) -> CharacterBody3D:
	var list := get_available_patients()
	if list.is_empty():
		return human
	if list.size() > 1 and exclude != null:
		var filtered: Array[CharacterBody3D] = list.filter(func(p: CharacterBody3D) -> bool: return p != exclude)
		if not filtered.is_empty():
			list = filtered
	if robot != null and list.size() > 1:
		var cur_fl: int = Room.get_floor(robot.global_position)
		var same_fl: Array[CharacterBody3D] = list.filter(func(p: CharacterBody3D) -> bool: return Room.get_floor(p.global_position) == cur_fl)
		if not same_fl.is_empty():
			list = same_fl
	var chosen: CharacterBody3D = list[_rng.randi_range(0, list.size() - 1)]
	if chosen != null:
		# Bệnh nhân đang chờ giúp đỡ đứng yên, không né chạy ("pardon") khi robot tiếp cận
		chosen.set("pardon_enabled", false)
		if chosen.get("walks") == true:
			chosen.set("walks", false)
	if chosen is HospitalNpc and (chosen as HospitalNpc).has_npc():
		(chosen as HospitalNpc).get_npc().reset()
	elif chosen != null:
		var n: Node = chosen.get_node_or_null(^"Npc")
		if n != null and n.has_method("reset"):
			n.reset()
	return chosen

func _is_search_phase() -> bool:
	return _rl_bridge.task_phase in [RLEnvBridgeScript.TaskPhase.SEARCH_PATIENT, RLEnvBridgeScript.TaskPhase.APPROACH_PATIENT]


func has_line_of_sight(from_node: Node3D, to_node: Node3D) -> bool:
	if from_node == null or to_node == null:
		return false
	var space_state := get_world_3d().direct_space_state
	if space_state == null:
		return true
	var from_pos := from_node.global_position + Vector3(0.0, 0.8, 0.0)
	var to_pos := to_node.global_position + Vector3(0.0, 0.8, 0.0)
	var query := PhysicsRayQueryParameters3D.create(from_pos, to_pos)
	var exclude: Array[RID] = []
	if from_node is CollisionObject3D:
		exclude.append((from_node as CollisionObject3D).get_rid())
	if to_node is CollisionObject3D:
		exclude.append((to_node as CollisionObject3D).get_rid())
	query.exclude = exclude
	var hit := space_state.intersect_ray(query)
	return hit.is_empty()


## Người được che khỏi depth (đích của robot, không phải vật cản): người đang dẫn, hoặc người đang tiếp cận ở pha tìm.
static func depth_masked_person(escorted: Node, target: Node, search_phase: bool, approaching: bool) -> Node:
	if escorted != null:
		return escorted
	return target if (search_phase and approaching) else null


## PPO lái robot ở mọi pha ngoài hội thoại; gửi trạng thái mỗi STEP_DT_S như dt lúc train.
## Pha tìm bệnh nhân: tới ≤ PATIENT_ENGAGE_DIST_M ➔ ARRIVED, dừng chờ NPC bắt đầu hội thoại.
func _step_policy(delta: float) -> void:
	var in_dialogue: bool = Main.is_dialogue_state(state)
	if in_dialogue and not policy_v3:
		return

	# Parse multi-class YOLO detections for patient channel
	var yolo_feat: Array = []
	var best_patient_det: Dictionary = {}
	var policy_dets: Array = _policy_dets()
	for d_var: Variant in policy_dets:
		var d: Dictionary = d_var as Dictionary
		var cid: int = int(d.get("class_id", -1))
		var cname: String = str(d.get("class_name", ""))
		if cid == 0 or cname == "patient":
			if best_patient_det.is_empty() or float(d.get("conf", 0.0)) > float(best_patient_det.get("conf", 0.0)):
				best_patient_det = d
	var yolo_cx: float = 0.0
	var yolo_h: float = 0.0
	if not best_patient_det.is_empty():
		yolo_cx = ((float(best_patient_det.get("x1", 0.0)) + float(best_patient_det.get("x2", 0.0))) / 2.0) - 0.5
		yolo_h = float(best_patient_det.get("y2", 0.0)) - float(best_patient_det.get("y1", 0.0))
		var conf: float = float(best_patient_det.get("conf", 0.9))
		yolo_feat = [1.0, yolo_cx, yolo_h, conf]

	if _is_search_phase() and not in_dialogue:
		var closest_info: Dictionary = get_closest_available_patient()
		var candidate_pat: CharacterBody3D = closest_info.get("patient") as CharacterBody3D
		var d_human: float = float(closest_info.get("dist", 999.0))
		var has_los: bool = (candidate_pat != null and has_line_of_sight(robot, candidate_pat))

		# Bind candidate patient if within visual/physical approach radius AND line of sight is clear
		if candidate_pat != null and has_los and d_human <= Config.TALK_TRIGGER_DIST + 1.5:
			if current_target_patient != candidate_pat:
				current_target_patient = candidate_pat
			current_target_patient.set("pardon_enabled", false)

		var yolo_patient_close: bool = (not best_patient_det.is_empty() and yolo_h >= 0.38 and absf(yolo_cx) < 0.35 and d_human <= PATIENT_ENGAGE_DIST_M and has_los)

		if state == Config.RobotState.ARRIVED:
			_arrived_wait_s += delta
			if _arrived_wait_s > ARRIVED_WAIT_S:
				overlay.set_status("Bệnh nhân không phản hồi, chuyển sang người khác")
				reset_positions(true)
				return
			if d_human <= PATIENT_ENGAGE_DIST_M + 0.5:
				if not policy_v3:
					robot.set_command(0.0, 0.0)
					return
				# V3: đứng chờ người bệnh nói nhưng policy vẫn tự quay / ngẩng đầu về phía họ (xem _on_policy_action)
			else:
				state = Config.RobotState.SEARCH  # bệnh nhân đi xa trước khi nói ➔ tìm lại
		elif not policy_v3 and has_los and (d_human <= PATIENT_ENGAGE_DIST_M or yolo_patient_close):
			# V3 không dùng luật dừng theo khoảng cách này: policy tự tới, tự dừng, tự hỏi (như lúc train). Luật này
			# từng bắt robot dừng cạnh người đi ngang qua trong khi nó đang tới bệnh nhân khác (log run_1500).
			if candidate_pat != null:
				current_target_patient = candidate_pat
				current_target_patient.set("pardon_enabled", false)
				if current_target_patient.get("walks") == true:
					current_target_patient.set("walks", false)
			print("[ESCORT_DEBUG] >>> Reached patient engage dist (d=%.2fm <= %.2fm, yolo_close=%s, has_los=true), waiting for speech..." % [d_human, PATIENT_ENGAGE_DIST_M, str(yolo_patient_close)])
			_arrived_wait_s = 0.0
			robot.set_command(0.0, 0.0)
			state = Config.RobotState.ARRIVED
			overlay.show_detections([], _display_state_name(), _connected)
			return

	_policy_accum += delta
	if _policy_accum < RLPolicyClientScript.STEP_DT_S:
		return
	_policy_accum = fmod(_policy_accum, RLPolicyClientScript.STEP_DT_S)
	var search: bool = _is_search_phase()
	var target_p: CharacterBody3D = active_escorted_patient if active_escorted_patient != null else (current_target_patient if current_target_patient != null else human)
	var pat_pos: Vector3 = target_p.global_position if active_escorted_patient != null else (robot.global_position if not search else target_p.global_position)
	# Che bệnh nhân đang tiếp cận / đang dẫn khỏi depth (khớp mask trong HospitalPureVisionV2Env)
	# Chỉ che khi đang tiếp cận (pha tìm) hoặc đang dẫn. Trước đây sau khi giao xong (pha quay về, state = APPROACH)
	# người vừa giao vẫn bị che ➔ robot không thấy người đứng chắn lối ra và cứ đẩy vào họ (log run_1335/1427/1500).
	robot.depth_ignore = Main.depth_masked_person(active_escorted_patient, current_target_patient, search,
			state == Config.RobotState.APPROACH) as CollisionObject3D
	_policy.send_state(
		robot.global_position, robot.rotation.y, robot.get_real_velocity().length(), robot.omega,
		robot.get_lidar_readings(), pat_pos,
		RLEnvBridgeScript.TaskPhase.SEARCH_PATIENT if search else _rl_bridge.task_phase,
		pat_pos if search else _leg_target(_rl_bridge.goal_position),
		_rl_bridge.patients_helped,
		yolo_feat,
		policy_dets,
		robot.get_depth_readings(),
		_v3_features()
	)


func _on_policy_action(v: float, omega: float, phase: int) -> void:
	if policy_v3 and (Main.is_dialogue_state(state) or (state == Config.RobotState.ARRIVED and _is_search_phase())) \
			and _policy_active() and not _is_in_elevator_transit:
		robot.set_command(0.0, omega)    # V3 tự quay về người đang nói / đang chờ nói (không đi khi đang nói chuyện)
		return
	if not _policy_active() or Main.is_dialogue_state(state) or _is_in_elevator_transit or _escort_holding():
		return
	if _is_search_phase():
		if state == Config.RobotState.ARRIVED:
			return
		state = Config.RobotState.APPROACH if phase == RLEnvBridgeScript.TaskPhase.APPROACH_PATIENT else Config.RobotState.SEARCH
	else:
		state = Config.RobotState.APPROACH
	robot.set_command(v, omega)


## Hỏi lại hết lượt mà vẫn chưa hiểu: escort-only ➔ đi gọi nhân viên (giống policy phía server sau MAX_CLARIFY).
func _on_clarify_gave_up(row: Dictionary) -> void:
	if not escort_only_mode or is_task_busy():
		return
	var r: Dictionary = row.duplicate()
	r["action"] = "call_staff"
	r["target"] = ""
	print("[ESCORT_DEBUG] >>> Hỏi lại vẫn chưa rõ ➔ gọi nhân viên")
	_on_dialogue_turn_completed(r)


func _on_dialogue_turn_completed(row: Dictionary) -> void:
	if _rl_bridge == null:
		return
	var act: String = str(row.get("action", ""))
	var target_key: String = str(row.get("target", ""))
	var fl: int = int(row.get("floor", 1))

	var speaking_pat: CharacterBody3D = get_active_speaking_patient()
	if _thanks_pending:
		_thanks_pending = false
		print("[ESCORT_DEBUG] >>> Thanks turn done: heard='%s', intent=%s, error='%s'" % [
			row.get("hyp_text", ""), row.get("intent_pred", ""), row.get("error", "")])
		return
	if _ask_pending:
		_ask_pending = false
		print("[ESCORT_DEBUG] >>> Continue? answer: heard='%s', intent=%s, error='%s'" % [
			row.get("hyp_text", ""), row.get("intent_pred", ""), row.get("error", "")])
		_on_continue_answer(str(row.get("intent_pred", "")) if str(row.get("error", "")) == "" else "")
		return

	# Chưa nghe rõ ➔ DialogueManager cho người bệnh nói lại; chưa được giao việc gì (trước đây escort-only
	# biến lượt này thành "guide" tới một phòng ngẫu nhiên, nên robot nói "chưa nghe rõ" mà vẫn dẫn đi).
	if act == "clarify":
		print("[ESCORT_DEBUG] >>> Chưa nghe rõ (heard='%s', intent=%s, conf=%.2f) ➔ nhờ nói lại" % [
			row.get("hyp_text", ""), row.get("intent_pred", ""), float(row.get("intent_conf", 0.0))])
		return
	if escort_only_mode and not act in ["guide", "call_staff", "emergency"]:
		# Hỏi thông tin / ngoài phạm vi: không đoán phòng, đưa về quầy tiếp đón.
		print("[ESCORT_DEBUG] >>> action='%s' không có đích ➔ dẫn tới quầy tiếp đón" % act)
		act = "guide"
		target_key = ESCORT_FALLBACK_TARGET
		fl = Room.get_room_floor(target_key)

	if inspection_panel != null:
		inspection_panel.update_dialogue(
			dialogue != null and dialogue.stt != null and dialogue.stt.is_server_connected(),
			dialogue != null and dialogue.tts != null and dialogue.tts.is_server_connected(),
			str(row.get("hyp_text", "")),
			str(row.get("intent_pred", "")),
			target_key,
			fl
		)
	if act == "guide" or act == "call_staff" or act == "emergency":
		active_escorted_patient = speaking_pat
		current_target_patient = speaking_pat
		if active_escorted_patient != null and active_escorted_patient.has_method("set_state"):
			active_escorted_patient.set_state(HospitalNpc.State.ESCORTED)
		_rl_bridge.begin_post_dialogue_task(act, target_key, fl, active_escorted_patient)
		if act == "guide":
			_begin_escort_hri()
		_task_elapsed_s = 0.0  # đích đổi từ bệnh nhân sang phòng / nhân viên
		_task_best_dist = -1.0
		if not Main.is_dialogue_state(state):
			state = Config.RobotState.APPROACH
		print("[ESCORT_DEBUG] >>> Post-dialogue task initiated: act='%s', target='%s', floor=%d -> Goal: key='%s', pos=(%.2f, %.2f), final='%s'" % [
			act, target_key, fl, _rl_bridge.goal_key, _rl_bridge.goal_position.x, _rl_bridge.goal_position.z, _rl_bridge.final_target_key
		])
		overlay.set_status("Bắt đầu dẫn đường bệnh nhân tới: %s (Tầng %d)" % [_rl_bridge.goal_key, fl])
		overlay.show_detections([], _display_state_name(), _connected)
	else:
		if act in ["answer", "repeat", "refuse", "adjust"]:
			_rl_bridge.cumulative_team_reward += RLEnvBridgeScript.REWARD_ANSWER
			_rl_bridge.patients_helped += 1
			overlay.set_status("Đã hỗ trợ bệnh nhân (R_team=%.0f)" % _rl_bridge.cumulative_team_reward)
		if _policy_active():
			if Main.is_dialogue_state(state):
				_pending_turn_reset = true
			else:
				reset_positions(true)


## Thực hiện chu trình đi thang máy 2 chiều đầy đủ hiệu ứng âm thanh, đèn LED và animation di chuyển
## Đi thang máy; `with_patient = false` (gọi nhân viên) ➔ robot đi một mình và giữ nguyên đích.
func _execute_elevator_transit(from_floor: int, to_floor: int, with_patient: bool = true) -> void:
	if _is_in_elevator_transit:
		return
	_is_in_elevator_transit = true
	robot.set_command(0.0, 0.0)
	if with_patient and announcer != null:
		announcer.say("elevator_enter")

	var station_from: ElevatorStation = room.get_elevator_station(from_floor) if room != null else null
	var station_to: ElevatorStation = room.get_elevator_station(to_floor) if room != null else null

	if station_from == null or station_to == null:
		var offset_x: float = float(to_floor - from_floor) * 50.0
		robot.global_position.x += offset_x
		if with_patient:
			var esc_p: CharacterBody3D = active_escorted_patient if active_escorted_patient != null else human
			esc_p.global_position.x += offset_x
			_rl_bridge.goal_key = _rl_bridge.final_target_key
			_rl_bridge.goal_position = room.get_room_waypoint(_rl_bridge.final_target_key) if room != null else Vector3.ZERO
		_is_in_elevator_transit = false
		_task_elapsed_s = 0.0  # chặng mới sau thang máy
		_task_best_dist = -1.0
		return

	var arrow: String = "▲" if to_floor > from_floor else "▼"
	station_from.set_floor_indicator(from_floor, arrow)
	station_to.set_floor_indicator(from_floor, arrow)
	overlay.set_status("Đang gọi thang máy: Tầng %d -> Tầng %d" % [from_floor, to_floor])
	_elev_cam_station = station_from
	var esc_p: CharacterBody3D = (active_escorted_patient if active_escorted_patient != null else human) if with_patient else null

	# 1. Cửa tầng xuất phát mở (thường đã mở khi robot tới gần ➔ không kêu chuông lần nữa)
	station_from.open_doors()

	# 2. Robot chạy vào buồng qua giữa cửa rồi quay mặt ra cửa; bệnh nhân đi bộ theo sau, đứng cạnh robot.
	#    Toạ độ cục bộ trạm: cửa ở z≈0, buồng z∈[-2.1, 0], hành lang z>0 (+Z hướng ra ngoài).
	var tf_from: Transform3D = station_from.global_transform
	var inv_from: Transform3D = tf_from.affine_inverse()
	var face_out_from: float = station_from.global_rotation.y + PI   # robot nhìn theo -Z cục bộ của nó
	var r_loc: Vector3 = inv_from * robot.global_position
	var p_loc: Vector3 = inv_from * esc_p.global_position if esc_p != null else Vector3(-1.0, 0.0, 1.0)
	# Mỗi người giữ phía (trái / phải) đang đứng ➔ không đi cắt ngang nhau trong buồng
	var r_side: float = 1.0 if r_loc.x >= p_loc.x else -1.0
	var t_robot_in: float = _tween_walk(robot, _elev_entry_path(tf_from, r_loc, Vector3(0.35 * r_side, 0.0, -1.25)),
			ELEV_ROBOT_SPEED, true, face_out_from, 0.0)
	var t_pat_in: float = 0.0
	if esc_p != null:
		t_pat_in = _tween_walk(esc_p, _elev_entry_path(tf_from, p_loc, Vector3(-0.4 * r_side, 0.0, -1.15)),
				ELEV_PATIENT_SPEED, false, station_from.global_rotation.y, ELEV_PATIENT_DELAY_S)

	var seq := create_tween()
	seq.tween_interval(maxf(t_robot_in, t_pat_in) + 0.2)

	# 3. Đóng cửa tầng xuất phát, động cơ chạy, đèn LED đổi tầng
	seq.tween_callback(func():
		station_from.close_doors()
	)
	seq.tween_interval(0.8)
	seq.tween_callback(func():
		station_from.start_hum()
		station_to.start_hum()
		overlay.set_status("Thang máy đang di chuyển (%s Tầng %d)..." % [("Lên" if to_floor > from_floor else "Xuống"), to_floor])
	)
	seq.tween_interval(0.8)
	seq.tween_callback(func():
		station_from.set_floor_indicator(to_floor, arrow)
		station_to.set_floor_indicator(to_floor, arrow)
	)
	seq.tween_interval(0.8)

	# 4. Đến nơi (cửa đang đóng nên không thấy dịch chuyển): đặt robot / bệnh nhân vào buồng tầng đích, giữ tư thế
	var tf_to: Transform3D = station_to.global_transform
	# Cửa đóng khi đổi tầng ➔ luôn xếp robot bên +X, bệnh nhân bên −X của buồng; ra ngoài bệnh nhân bước thẳng tới
	# chỗ đứng chếch trước-phải robot (giống lúc dẫn đường: nằm trong camera) mà không cắt ngang trước mặt robot.
	var robot_out: Array[Vector3] = [tf_to * Vector3(0.3, 0.0, 0.5), tf_to * Vector3(0.3, 0.0, 1.5)]
	var pat_out: Array[Vector3] = [tf_to * Vector3(-0.3, 0.0, 0.4), tf_to * Vector3(-0.15, 0.0, 2.6)]
	seq.tween_callback(func():
		station_from.stop_hum()
		station_to.stop_hum()
		var face_out_to: float = station_to.global_rotation.y + PI
		robot.global_position = tf_to * Vector3(0.35, 0.0, -1.25)
		robot.rotation.y = face_out_to
		if esc_p != null:
			esc_p.global_position = tf_to * Vector3(-0.4, 0.0, -1.15)
			esc_p.rotation.y = station_to.global_rotation.y
		station_from.set_floor_indicator(to_floor, "")
		station_to.set_floor_indicator(to_floor, "")
		_elev_cam_station = station_to
		_update_viewer_camera()
	)
	seq.tween_interval(0.5)

	# 5. Mở cửa tầng đích (open_doors tự phát chuông ding-dong một lần)
	seq.tween_callback(func():
		station_to.open_doors()
	)
	seq.tween_interval(0.7)

	# 6. Robot ra trước, bệnh nhân bước ra theo sau lưng (cách ~1 m, không đứng chồng lên robot)
	var t_out: float = (tf_to * Vector3(0.35, 0.0, -1.25)).distance_to(robot_out[0]) / ELEV_ROBOT_SPEED \
			+ robot_out[0].distance_to(robot_out[1]) / ELEV_ROBOT_SPEED + 0.4
	seq.tween_callback(func():
		_tween_walk(robot, robot_out, ELEV_ROBOT_SPEED, true, station_to.global_rotation.y + PI, 0.0)
		if esc_p != null:
			_tween_walk(esc_p, pat_out, ELEV_PATIENT_SPEED, false, station_to.global_rotation.y, ELEV_PATIENT_DELAY_S)
	)
	seq.tween_interval(t_out + ELEV_PATIENT_DELAY_S)

	# 7. Đóng cửa tầng đích, chuyển goal sang phòng khám cuối cùng (hoặc sảnh tìm bệnh nhân)
	seq.tween_callback(func():
		station_to.close_doors()
		_elev_cam_station = null
		_rl_bridge.escort_trail.clear()   # vết đường của tầng cũ không còn dùng được
		RLEnvBridgeScript.reset_escort_follow()
		if with_patient:
			_rl_bridge.goal_key = _rl_bridge.final_target_key
			_rl_bridge.goal_position = room.get_room_waypoint(_rl_bridge.final_target_key)
			overlay.set_status("Đã đến Tầng %d, tiếp tục tới: %s" % [to_floor, _rl_bridge.goal_key])
			if announcer != null:
				announcer.say("elevator_exit", {"floor": to_floor})
			escort_monitor.far_s = 0.0
		else:
			_rl_bridge.task_phase = RLEnvBridgeScript.TaskPhase.SEARCH_PATIENT
			_rl_bridge.goal_key = "patient"
			state = Config.RobotState.SEARCH
			current_target_patient = pick_next_target_patient()
			overlay.set_status("Đã trở về Sảnh Tầng 1, tiếp tục tìm bệnh nhân...")
		_is_in_elevator_transit = false
		_task_elapsed_s = 0.0  # chặng mới sau thang máy
		_task_best_dist = -1.0
	)


## Đường vào buồng (toạ độ thế giới): còn ở hành lang ➔ đi qua giữa cửa trước; đã ở cửa / trong buồng ➔ thẳng tới chỗ đứng.
static func _elev_entry_path(tf: Transform3D, cur_local: Vector3, slot_local: Vector3) -> Array[Vector3]:
	var pts: Array[Vector3] = []
	if cur_local.z > 0.35:
		pts.append(tf * Vector3(clampf(cur_local.x, -0.45, 0.45) * 0.5 + slot_local.x * 0.5, 0.0, 0.6))
	pts.append(tf * slot_local)
	return pts


## Cho `node` đi qua các điểm `pts` (mặt phẳng XZ) với tốc độ `speed`, quay theo hướng đi (góc ngắn nhất),
## cuối cùng quay về `final_yaw`. `forward_neg_z` = mặt trước là -Z (robot); NPC nhìn theo +Z.
## Trả tổng thời gian (giây, gồm `delay`). Di chuyển theo vị trí ➔ NPC tự chạy animation đi bộ.
func _tween_walk(node: Node3D, pts: Array[Vector3], speed: float, forward_neg_z: bool, final_yaw: float, delay: float) -> float:
	var tw := create_tween()
	var total: float = delay
	if delay > 0.0:
		tw.tween_interval(delay)
	var prev: Vector3 = node.global_position
	for p: Vector3 in pts:
		var target := Vector3(p.x, prev.y, p.z)
		var d := target - prev
		var dist: float = Vector2(d.x, d.z).length()
		if dist < 0.02:
			continue
		var heading: float = atan2(-d.x, -d.z) if forward_neg_z else atan2(d.x, d.z)
		var t_turn: float = 0.25
		var t_move: float = dist / speed
		_tween_turn(tw, node, heading, t_turn)
		tw.tween_property(node, ^"global_position", target, t_move)
		total += t_turn + t_move
		prev = target
	_tween_turn(tw, node, final_yaw, 0.4)
	return total + 0.4


## Quay `node` tới `yaw` theo góc ngắn nhất. Góc xuất phát đọc lúc bắt đầu quay (không lúc dựng tween), vì Godot
## trả rotation.y đã quy về [−π, π] (robot.gd ghi rotation.y mỗi frame) ➔ nội suy số thực thẳng có thể quay cả vòng.
func _tween_turn(tw: Tween, node: Node3D, yaw: float, duration: float) -> void:
	var from := [0.0]
	tw.tween_callback(func() -> void: from[0] = node.rotation.y)
	tw.tween_method(func(t: float) -> void: node.rotation.y = lerp_angle(from[0], yaw, t), 0.0, 1.0, duration)


func _update_viewer_camera() -> void:
	if viewer_camera == null or robot == null:
		return
	if ceiling_mesh is VisualInstance3D:
		RenderLayers.set_viewer_visible(ceiling_mesh as VisualInstance3D, camera_mode != CameraMode.TOP_DOWN)
	if room != null:
		room.set_ceiling_visible(camera_mode != CameraMode.TOP_DOWN)
	var x_center: float = 100.0 if robot.global_position.x >= 75.0 else (50.0 if robot.global_position.x >= 25.0 else 0.0)
	match camera_mode:
		CameraMode.FOLLOW when _elev_cam_station != null:
			# Đang đi thang máy: góc cố định ngoài hành lang nhìn thẳng vào buồng (camera bám robot bị tường che)
			var stf: Transform3D = _elev_cam_station.global_transform
			viewer_camera.look_at_from_position(stf * Vector3(0.9, 1.75, 2.7), stf * Vector3(0.0, 0.9, -0.8), Vector3.UP)
		CameraMode.FOLLOW:
			var cam_target := robot.global_position + Vector3(0.0, 0.8, 0.0)
			var orbit_dist: float = 2.4
			var orbit_height: float = 1.2
			var ox: float = sin(camera_orbit_angle) * orbit_dist
			var oz: float = cos(camera_orbit_angle) * orbit_dist
			var cam_pos := cam_target + Vector3(ox, orbit_height, oz)
			cam_pos.x = clampf(cam_pos.x, x_center - 11.5, x_center + 11.5)
			cam_pos.y = clampf(cam_pos.y, 0.8, 2.8)
			cam_pos.z = clampf(cam_pos.z, -8.5, 8.5)
			if absf(cam_pos.z - cam_target.z) < 0.2 and absf(cam_pos.x - cam_target.x) < 0.2:
				cam_pos.z = cam_target.z + 0.5
			viewer_camera.look_at_from_position(
				cam_pos,
				cam_target,
				Vector3.UP
			)
		CameraMode.TOP_DOWN:
			# Nhìn từ trên cao xuống toàn bộ bệnh viện (tự động ẩn trần như RimWorld)
			viewer_camera.look_at_from_position(
				Vector3(x_center, 13.0, 1.2),
				Vector3(x_center, 0.0, 0.0),
				Vector3.UP
			)
		CameraMode.LOBBY:
			# Góc nhìn bao quát sảnh chính đón tiếp ở độ cao 2.3 m bên trong phòng
			viewer_camera.look_at_from_position(
				Vector3(x_center - 4.0, 2.7, 7.8),
				Vector3(x_center, 1.2, 5.0),
				Vector3.UP
			)
	_update_wall_transparency()


func _get_transparent_wall_mat() -> StandardMaterial3D:
	if _transparent_wall_mat != null:
		return _transparent_wall_mat
	var base_mat: StandardMaterial3D = null
	if not _interior_walls.is_empty() and _interior_walls[0].mesh != null:
		base_mat = _interior_walls[0].mesh.material as StandardMaterial3D
	if base_mat != null:
		_transparent_wall_mat = base_mat.duplicate() as StandardMaterial3D
	else:
		_transparent_wall_mat = StandardMaterial3D.new()
	_transparent_wall_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	_transparent_wall_mat.albedo_color = Color(0.96, 0.97, 0.95, 0.22)
	return _transparent_wall_mat


func _cache_interior_walls() -> void:
	_interior_walls.clear()
	var iw: Node = get_node_or_null(^"Room/InteriorWalls")
	if iw != null:
		for child: Node in iw.get_children():
			var mesh_inst := child as MeshInstance3D
			if mesh_inst != null and mesh_inst.mesh is BoxMesh:
				_interior_walls.append(mesh_inst)
	var iw2: Node = get_node_or_null(^"Room/Floor2/InteriorWalls")
	if iw2 != null:
		for child: Node in iw2.get_children():
			var mesh_inst := child as MeshInstance3D
			if mesh_inst != null and mesh_inst.mesh is BoxMesh:
				_interior_walls.append(mesh_inst)
	var iw3: Node = get_node_or_null(^"Room/Floor3/InteriorWalls")
	if iw3 != null:
		for child: Node in iw3.get_children():
			var mesh_inst := child as MeshInstance3D
			if mesh_inst != null and mesh_inst.mesh is BoxMesh:
				_interior_walls.append(mesh_inst)


func _update_wall_transparency() -> void:
	if viewer_camera == null or robot == null:
		return
	if camera_mode == CameraMode.TOP_DOWN:
		_clear_wall_transparency()
		return
	if _interior_walls.is_empty():
		_cache_interior_walls()

	var cam_pos: Vector3 = viewer_camera.global_position
	var r_pos: Vector3 = robot.global_position
	var targets: Array[Vector3] = [
		r_pos + Vector3(0.0, 0.8, 0.0),
		r_pos + Vector3(0.0, 1.1, 0.0),
		r_pos + Vector3(0.0, 0.3, 0.0),
	]

	var new_occluding: Array[MeshInstance3D] = []
	for wall: MeshInstance3D in _interior_walls:
		var box: BoxMesh = wall.mesh as BoxMesh
		if box == null:
			continue
		var aabb := AABB(wall.global_position - box.size / 2.0, box.size)
		for t: Vector3 in targets:
			if aabb.intersects_segment(cam_pos, t) != null:
				new_occluding.append(wall)
				break

	for wall: MeshInstance3D in _occluding_walls:
		if not new_occluding.has(wall) and is_instance_valid(wall):
			_hide_wall_ghost(wall)

	for wall: MeshInstance3D in new_occluding:
		if not _occluding_walls.has(wall):
			_show_wall_ghost(wall)

	_occluding_walls = new_occluding


func _update_inspection_panel() -> void:
	if inspection_panel == null or not inspection_panel.is_open or robot == null:
		return
	if not use_rl_controller and not inspection_panel.is_pure_vision:
		inspection_panel.update_lidar(robot.get_lidar_readings())
	var current_pat: CharacterBody3D = active_escorted_patient if active_escorted_patient != null else (current_target_patient if current_target_patient != null else human)
	var search: bool = _is_search_phase()
	var pat_pos: Vector3 = current_pat.global_position if active_escorted_patient != null else Main.get_patient_interaction_pos(current_pat)
	var g_pos: Vector3 = pat_pos if (search or _rl_bridge == null) else _rl_bridge.goal_position
	var r_pos: Vector3 = robot.global_position
	var d_goal: float = Human.flat_distance(r_pos, g_pos)
	var r_yaw: float = robot.rotation.y
	var fwd := Vector3(-sin(r_yaw), 0.0, -cos(r_yaw))
	var to_goal := Vector3(g_pos.x - r_pos.x, 0.0, g_pos.z - r_pos.z)
	var angle_rad: float = 0.0
	if to_goal.length() > 1e-4:
		angle_rad = deg_to_rad(RunLogger.signed_angle_deg(fwd, to_goal))
	var esc_p: CharacterBody3D = active_escorted_patient if active_escorted_patient != null else human
	var d_tether: float = Human.flat_distance(r_pos, esc_p.global_position)
	var cur_vel: Vector3 = robot.get_real_velocity()
	var phase_str: String = Config.state_name(state)
	if _rl_bridge != null:
		match _rl_bridge.task_phase:
			RLEnvBridgeScript.TaskPhase.SEARCH_PATIENT:
				phase_str = "SEARCH_PATIENT"
			RLEnvBridgeScript.TaskPhase.APPROACH_PATIENT:
				phase_str = "APPROACH_PATIENT"
			RLEnvBridgeScript.TaskPhase.GUIDE_ESCORT:
				phase_str = "GUIDE_ESCORT"
			RLEnvBridgeScript.TaskPhase.CALL_STAFF:
				phase_str = "CALL_STAFF"
	inspection_panel.update_rl_navigation(
		phase_str, d_goal, sin(angle_rad), cos(angle_rad), d_tether, cur_vel.length(), robot.omega
	)



func _clear_wall_transparency() -> void:
	for wall: MeshInstance3D in _occluding_walls:
		if is_instance_valid(wall):
			_hide_wall_ghost(wall)
	_occluding_walls.clear()


## Tường chắn người xem: tường thật chuyển sang layer chỉ camera robot render (robot vẫn thấy tường đặc, không nhìn
## xuyên được), thêm một bản "ma" trong suốt chỉ ViewerCamera render. Va chạm / tia depth dùng physics, không đổi.
func _show_wall_ghost(wall: MeshInstance3D) -> void:
	RenderLayers.set_viewer_visible(wall, false)
	if wall.get_node_or_null(^"ViewerGhost") != null:
		return
	var ghost := MeshInstance3D.new()
	ghost.name = "ViewerGhost"
	ghost.mesh = wall.mesh
	ghost.material_override = _get_transparent_wall_mat()
	ghost.layers = RenderLayers.VIEWER_ONLY
	ghost.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	wall.add_child(ghost)


func _hide_wall_ghost(wall: MeshInstance3D) -> void:
	RenderLayers.set_viewer_visible(wall, true)
	var ghost: Node = wall.get_node_or_null(^"ViewerGhost")
	if ghost != null:
		wall.remove_child(ghost)
		ghost.queue_free()


## Đặt robot và người ở vị trí ngẫu nhiên mới, robot hướng ngẫu nhiên, về SEARCH.
func reset_positions(is_task_transition: bool = false) -> void:
	_clear_wall_transparency()
	if _rl_bridge != null:
		# Không đặt lại pha thì bệnh nhân vẫn ở GUIDE_ESCORT và cứ đi theo robot sau khi reset
		_rl_bridge.task_phase = RLEnvBridgeScript.TaskPhase.SEARCH_PATIENT
		_rl_bridge.escort_trail.clear()
	_task_elapsed_s = 0.0
	_task_best_dist = -1.0
	_arrived_wait_s = 0.0
	_reset_escort_hri()
	identity.reset()
	_voice_mem_t = -1e9
	_forget_served()
	_id_det = {}
	overlay.set_remembered_face(null)
	var prev_pat: CharacterBody3D = active_escorted_patient
	if active_escorted_patient != null and active_escorted_patient.has_method("end_escort"):
		active_escorted_patient.end_escort()
	active_escorted_patient = null
	_pending_turn_reset = false
	_thanks_pending = false
	_after_thanks = Callable()
	_elev_cam_station = null
	human.end_escort()
	current_target_patient = pick_next_target_patient(prev_pat)
	if is_task_transition:
		if solo_patient_mode and room != null:
			var cur_fl: int = Room.get_floor(robot.global_position)
			var next_pos: Vector3 = room.get_patient_spawn_position(cur_fl, _rng)
			for _i: int in 30:
				if Human.flat_distance(next_pos, robot.global_position) >= 3.0:
					break
				next_pos = room.get_patient_spawn_position(cur_fl, _rng)
			human.global_position = next_pos
			human.rotation.y = _rng.randf_range(-PI, PI)
	elif solo_patient_mode:
		var p: Dictionary = Main.pick_solo_start_positions(room, _rng)
		robot.global_position = p["robot"]
		robot.rotation.y = p["robot_yaw"]
		human.global_position = p["human"]
		human.rotation.y = p["human_yaw"]
	else:
		var dock: Dictionary = room.get_robot_dock()
		robot.global_position = dock["position"]
		robot.rotation.y = float(dock["yaw"])
		var p: Dictionary = Main.pick_start_positions(room, _rng)
		human.global_position = p["human"]
		human.rotation.y = p["human_yaw"]
		# Ngẫu nhiên hóa vị trí ban đầu của các bệnh nhân trong sảnh
		var furn: Node = room.get_node_or_null(^"Furniture") if room != null else null
		if furn != null:
			for child: Node in furn.get_children():
				if child is HospitalNpc:
					var h_npc := child as HospitalNpc
					if h_npc.staff_role == "patient" and not h_npc.is_seated and not h_npc.get_meta("helped", false):
						var np_pos: Vector3 = room.get_patient_spawn_position(1, _rng) if room != null else Vector3.ZERO
						if np_pos != Vector3.ZERO:
							h_npc.global_position = np_pos
							h_npc.rotation.y = _rng.randf_range(-PI, PI)
	robot.set_command(0.0, 0.0)
	state = Config.RobotState.SEARCH
	if dialogue != null:
		dialogue.reset()
	var npc: Npc = human.get_node_or_null(^"Npc")
	if npc != null:
		npc.reset()
	_last_seen_ms = -1
	_last_result_ms = Time.get_ticks_msec()
	overlay.show_detections([], _display_state_name(), _connected)
	_update_viewer_camera()


func get_closest_available_patient() -> Dictionary:
	var patients: Array[CharacterBody3D] = get_available_patients()
	var best_p: CharacterBody3D = null
	var min_d: float = INF
	if robot == null:
		return {"patient": current_target_patient if current_target_patient != null else human, "dist": 999.0}
	var r_pos: Vector3 = robot.global_position
	var cur_fl: int = Room.get_floor(r_pos)
	for p: CharacterBody3D in patients:
		if Room.get_floor(p.global_position) != cur_fl:
			continue
		var d: float = Human.flat_distance(r_pos, p.global_position)
		if d < min_d:
			min_d = d
			best_p = p
	if best_p == null:
		best_p = current_target_patient if current_target_patient != null else human
		min_d = Human.flat_distance(r_pos, best_p.global_position)
	return {"patient": best_p, "dist": min_d}


func human_distance() -> float:
	var spk: CharacterBody3D = get_active_speaking_patient() if Main.is_dialogue_state(state) else null
	if spk != null:
		return Human.flat_distance(robot.global_position, spk.global_position)
	if active_escorted_patient != null:
		return Human.flat_distance(robot.global_position, active_escorted_patient.global_position)
	var closest: Dictionary = get_closest_available_patient()
	return float(closest.get("dist", 999.0))


## Trạng thái robot mà NPC `pat` nhìn thấy (quyết định lúc nào họ bắt chuyện, xem Npc.next_state). SEARCH = "robot không
## rảnh với mình": đang bận ca khác, đang nói chuyện, hoặc đang nhắm người khác.
func _robot_state_for(pat: CharacterBody3D) -> int:
	if is_task_busy() or Main.is_dialogue_state(state):
		return Config.RobotState.SEARCH
	if current_target_patient != null and current_target_patient != pat:
		return Config.RobotState.SEARCH
	# V3: robot tự dừng gần người bệnh mà chưa nhận ra box của họ (state = SEARCH) ➔ họ vẫn tự bắt chuyện, như
	# "người tự tới nói" lúc train (hospital_env_engage_v3.py WALKUP_*).
	if policy_v3 and state == Config.RobotState.SEARCH:
		return Config.RobotState.APPROACH
	return state


func _setup_hospital_npcs() -> void:
	current_target_patient = pick_next_target_patient()
	if human != null:
		human.setup_robot(robot)
		var h_npc_node: Node = human.get_node_or_null(^"Npc")
		if h_npc_node != null and h_npc_node.has_method("setup"):
			h_npc_node.setup(robot, func() -> int: return _robot_state_for(human))
	var furn: Node = room.get_node_or_null(^"Furniture") if room != null else null
	if furn != null:
		for child: Node in furn.get_children():
			if child is HospitalNpc:
				var h_npc := child as HospitalNpc
				if solo_patient_mode:
					h_npc.visible = false
					h_npc.process_mode = Node.PROCESS_MODE_DISABLED
					for col: Node in h_npc.find_children("*", "CollisionShape3D", true, false):
						(col as CollisionShape3D).disabled = true
				else:
					var pat_body := h_npc as CharacterBody3D
					h_npc.setup_robot(robot, func() -> int: return _robot_state_for(pat_body))
					if dialogue != null and h_npc.has_npc():
						dialogue.register_npc(h_npc.get_npc())
	if escort_only_mode:
		_apply_escort_only_mode()


func _apply_escort_only_mode() -> void:
	if human != null:
		var n: Node = human.get_node_or_null(^"Npc")
		if n != null and "escort_only" in n:
			n.set("escort_only", true)
	var furn: Node = room.get_node_or_null(^"Furniture") if room != null else null
	if furn != null:
		for child: Node in furn.get_children():
			if child is HospitalNpc and (child as HospitalNpc).has_npc():
				var n: Node = (child as HospitalNpc).get_npc()
				if n != null and "escort_only" in n:
					n.set("escort_only", true)


func _nearest_person_distance() -> float:
	var d: float = human_distance()
	if solo_patient_mode:
		return d
	var furn: Node = room.get_node_or_null(^"Furniture") if room != null else null
	if furn != null:
		for child: Node in furn.get_children():
			if child is HospitalNpc and child.visible:
				d = minf(d, Human.flat_distance(robot.global_position, (child as Node3D).global_position))
	return d


## Góc thật tới người so với hướng trước của robot (−Z), dương khi người bên trái.
func human_angle_deg() -> float:
	var spk: CharacterBody3D = get_active_speaking_patient() if Main.is_dialogue_state(state) else null
	var target_pat: CharacterBody3D = spk if spk != null else (active_escorted_patient if active_escorted_patient != null else (current_target_patient if current_target_patient != null else human))
	var target_pos: Vector3 = target_pat.global_position if active_escorted_patient != null else Main.get_patient_interaction_pos(target_pat)
	return RunLogger.signed_angle_deg(-robot.global_transform.basis.z, target_pos - robot.global_position)


## Một dòng log G9: kết quả nhận diện (target do controller chọn) + ground truth + lệnh sau an toàn.
func build_log_row(frame_id: int, dets: Array, cmd: Dictionary, infer_ms: float, now_ms: int) -> Dictionary:
	var target: Dictionary = cmd["target"]
	var row: Dictionary = {
		"time_s": (now_ms - _start_ms) / 1000.0, "frame_id": frame_id, "state": Config.state_name(cmd["state"]),
		"n_det": dets.size(), "true_distance_m": human_distance(), "true_angle_deg": human_angle_deg(),
		"v": cmd["v"], "omega": cmd["omega"], "infer_ms": infer_ms, "connected": _connected,
	}
	if not target.is_empty():
		row["conf"] = target["conf"]
		row["cx"] = (float(target["x1"]) + float(target["x2"])) / 2.0
		row["h"] = float(target["y2"]) - float(target["y1"])
	return row


func _on_detections(frame_id: int, dets: Array, infer_ms: float) -> void:
	_latest_yolo_dets = dets
	_update_identity(dets)
	var now: int = Time.get_ticks_msec()
	_last_result_ms = now
	if not dets.is_empty():
		_last_seen_ms = now
	var cmd: Dictionary
	if Main.is_dialogue_state(state):
		# Đứng yên nhưng vẫn quay về phía người đang nói (_face_speaker trong _physics_process).
		cmd = {"state": state, "v": 0.0, "omega": robot.omega, "target": {}}
		robot.set_command(0.0, robot.omega)
	elif _rl_bridge != null and _rl_bridge.task_phase in [RLEnvBridgeScript.TaskPhase.GUIDE_ESCORT, RLEnvBridgeScript.TaskPhase.CALL_STAFF]:
		var cur_cmd: Vector2 = robot.get_command() if robot.has_method("get_command") else Vector2(robot.v, robot.omega)
		cmd = {"state": state, "v": cur_cmd.x, "omega": cur_cmd.y, "target": {}}
	elif _policy_active():
		var pol_cmd: Vector2 = robot.get_command()
		cmd = {"state": state, "v": pol_cmd.x, "omega": pol_cmd.y, "target": {}}
	else:
		cmd = _apply(_controller.step(state, dets, _time_since_seen_s(now)))
	overlay.show_detections(dets, _display_state_name(), _connected)
	overlay.show_faces(vision.last_faces, rear_vision.last_faces if rear_vision != null and rear_vision.enabled else [])
	_update_face_panel(now)
	if inspection_panel != null and inspection_panel.is_open:
		inspection_panel.update_vision(dets)
	if logger != null:
		logger.log_row(build_log_row(frame_id, dets, cmd, infer_ms, now))
	cycle_completed.emit({"frame_id": frame_id, "dets": dets, "cmd": cmd, "infer_ms": infer_ms,
			"connected": _connected})


func _on_connection_changed(connected: bool) -> void:
	_connected = connected
	if not connected:
		robot.set_command(0.0, 0.0)  # giữ nguyên trạng thái
	else:
		_last_result_ms = Time.get_ticks_msec()
	overlay.show_detections([], _display_state_name(), connected)


func _display_state_name() -> String:
	if _rl_bridge != null and _rl_bridge.task_phase == RLEnvBridgeScript.TaskPhase.GUIDE_ESCORT:
		return "ESCORT: %s" % _rl_bridge.goal_key
	elif _rl_bridge != null and _rl_bridge.task_phase == RLEnvBridgeScript.TaskPhase.CALL_STAFF:
		return "CALL_STAFF"
	return Config.state_name(state)


## LISTEN / THINK / RESPOND (Mục 2.3 Sim_Voice).
static func is_dialogue_state(s: int) -> bool:
	return s == Config.RobotState.LISTEN or s == Config.RobotState.THINK or s == Config.RobotState.RESPOND


## Gọi bởi DialogueManager. Trạng thái hội thoại ➔ dừng robot. Về ARRIVED ➔ controller chạy lại ở frame sau.
func set_dialogue_state(s: int) -> void:
	if s == Config.RobotState.LISTEN and not Main.is_dialogue_state(state) and active_escorted_patient == null:
		identity.reset()                 # người mới nói chuyện ➔ ghi nhớ lại từ đầu
		overlay.set_remembered_face(null)
		_enroll_misses = 0
		_enroll_reasons = {}
	state = s
	_arrived_wait_s = 0.0
	if Main.is_dialogue_state(s):
		robot.set_command(0.0, 0.0)
	elif s == Config.RobotState.ARRIVED:
		if _after_thanks.is_valid():
			var after: Callable = _after_thanks
			_after_thanks = Callable()
			_thanks_pending = false
			after.call()
			if is_task_busy():               # tầng trên ➔ robot tự về thang máy
				state = Config.RobotState.APPROACH
				_task_elapsed_s = 0.0
				_task_best_dist = -1.0
		elif is_task_busy():
			state = Config.RobotState.APPROACH
			_task_elapsed_s = 0.0
			_task_best_dist = -1.0
		elif _pending_turn_reset:
			_pending_turn_reset = false
			if _policy_active():
				reset_positions(true)
	_last_result_ms = Time.get_ticks_msec()
	overlay.show_detections([], _display_state_name(), _connected)


## Áp an toàn rồi gửi lệnh cho robot; trả lệnh cuối cùng.
func _apply(cmd: Dictionary) -> Dictionary:
	var dist_human: float = _nearest_person_distance()
	var dist_obstacle: float = robot.front_obstacle_distance()
	var safe: Dictionary = Main.apply_safety(cmd, dist_human, dist_obstacle)
	state = safe["state"]
	if state == Config.RobotState.APPROACH:
		var target: Dictionary = cmd.get("target", {})
		var h_box: float = (float(target["y2"]) - float(target["y1"])) if (not target.is_empty() and target.has("y2")) else 0.0
		if dist_human <= Config.SAFE_DIST_HUMAN + 0.35 or (safe["v"] == 0.0 and (dist_human <= Config.TALK_TRIGGER_DIST or h_box >= 0.45)):
			state = Config.RobotState.ARRIVED
			safe["state"] = Config.RobotState.ARRIVED
	robot.set_command(safe["v"], safe["omega"])
	return safe


func _time_since_seen_s(now: int) -> float:
	return INF if _last_seen_ms < 0 else (now - _last_seen_ms) / 1000.0


## Ghi đè an toàn: quá gần người hoặc vật cản phía trước ➔ v = 0; giữ nguyên omega và state.
static func apply_safety(cmd: Dictionary, dist_human: float, dist_obstacle: float) -> Dictionary:
	var out: Dictionary = cmd.duplicate()
	if dist_human < Config.SAFE_DIST_HUMAN or dist_obstacle < Config.SAFE_DIST_OBSTACLE:
		out["v"] = 0.0
	return out


## Vị trí đầu ca: robot ở trạm cạnh cửa chính (quay vào hành lang), bệnh nhân ở hành lang / sảnh
## (room.get_patient_spawn_position) cách robot ≥ DOCK_START_SEPARATION. Cùng rng.seed ➔ cùng kết quả.
static func pick_start_positions(r: Room, rng: RandomNumberGenerator) -> Dictionary:
	var dock: Dictionary = r.get_robot_dock()
	var robot_pos: Vector3 = dock["position"]
	var human_pos := Vector3.ZERO
	var ok: bool = false
	for i: int in MAX_PLACEMENT_TRIES:
		human_pos = r.get_patient_spawn_position(1, rng)
		if Human.flat_distance(robot_pos, human_pos) >= DOCK_START_SEPARATION:
			ok = true
			break
	if not ok:
		push_error("Main.pick_start_positions: no patient spot ≥ %.1f m from the dock" % DOCK_START_SEPARATION)
	return {"robot": robot_pos, "human": human_pos,
			"robot_yaw": float(dock["yaw"]), "human_yaw": rng.randf_range(-PI, PI)}


## Vị trí bắt đầu solo: Bệnh nhân ở hành lang/sảnh (rõ ràng, không bị kẹt sau bàn ghế), robot ở gần (2.0–3.5m) hướng mặt vào bệnh nhân với tầm nhìn trực tiếp.
static func pick_solo_start_positions(r: Room, rng: RandomNumberGenerator) -> Dictionary:
	var human_pos: Vector3 = Vector3.ZERO
	if r != null and r.has_method("get_patient_spawn_position"):
		human_pos = r.get_patient_spawn_position(1, rng)
	elif r != null:
		human_pos = r.get_free_position(START_CLEARANCE, rng)
	var robot_pos := human_pos
	var ok: bool = false
	var footprints: Array[Rect2] = r.get_obstacle_footprints() if r != null else []
	for i: int in MAX_PLACEMENT_TRIES:
		var dist: float = rng.randf_range(2.0, 3.5)
		var angle: float = rng.randf_range(-PI, PI)
		var cand := Vector3(human_pos.x + dist * cos(angle), 0.0, human_pos.z + dist * sin(angle))
		if absf(cand.x) <= 10.8 and absf(cand.z) <= 7.2 and Room.is_clear(cand, footprints, START_CLEARANCE):
			# Kiểm tra đường nhìn thẳng không bị cản bởi đồ đạc / tường
			var clear_line: bool = true
			for fp: Rect2 in footprints:
				var exp_fp := fp.grow(0.15)
				var aabb := AABB(Vector3(exp_fp.position.x, -1.0, exp_fp.position.y), Vector3(exp_fp.size.x, 2.0, exp_fp.size.y))
				if aabb.intersects_segment(human_pos, cand) != null:
					clear_line = false
					break
			if clear_line:
				robot_pos = cand
				ok = true
				break
	if not ok:
		robot_pos = r.get_free_position(START_CLEARANCE, rng) if r != null else Vector3.ZERO
	var to_h: Vector3 = human_pos - robot_pos
	var face_yaw: float = atan2(-to_h.x, -to_h.z) + rng.randf_range(-0.25, 0.25)
	return {
		"robot": robot_pos,
		"human": human_pos,
		"robot_yaw": face_yaw,
		"human_yaw": rng.randf_range(-PI, PI)
	}


## Bệnh nhân vừa được dẫn tới nơi chủ động nói clip cảm ơn ➔ lượt hội thoại thật (STT / NLU / TTS). Không được ➔ false.
func _start_thanks_turn(patient: CharacterBody3D) -> bool:
	if dialogue == null or patient == null:
		return false
	# HospitalNpc tạo Npc qua get_npc(); Human chính có sẵn node con "Npc".
	var p_npc: Npc = (patient.get_npc() if patient.has_method("get_npc") else patient.get_node_or_null(^"Npc")) as Npc
	if p_npc == null:
		return false
	dialogue.register_npc(p_npc)
	_thanks_pending = true
	var clip: String = p_npc.say_intent("cam_on")
	if clip != "" and Main.is_dialogue_state(state):
		print("[ESCORT_DEBUG] >>> Patient thanks robot with clip '%s'" % clip)
		return true
	_thanks_pending = false
	if clip != "":
		p_npc.reset()
	return false


## Kết thúc nhiệm vụ dẫn đường: ở tầng trên ➔ tự về thang máy xuống sảnh; tầng 1 ➔ chọn bệnh nhân kế tiếp.
func _finish_escort(cur_floor: int) -> void:
	if cur_floor > 1:
		print("[ESCORT_DEBUG] >>> Escort completed on Floor %d. Routing robot to return to Floor 1 via elevator..." % cur_floor)
		_rl_bridge.goal_key = "tiep_don"
		_rl_bridge.goal_position = room.get_room_waypoint("tiep_don") if room != null else Vector3.ZERO
		_rl_bridge.final_target_key = "tiep_don"
		_rl_bridge.task_phase = RLEnvBridgeScript.TaskPhase.CALL_STAFF
		_task_elapsed_s = 0.0
		_task_best_dist = -1.0
		overlay.set_status("Đã dẫn bệnh nhân tới %s. Đang tự di chuyển về Thang máy Tầng %d để xuống Sảnh..." % [_rl_bridge.goal_key, cur_floor])
		return
	reset_positions(true)


## Nạp wav vòng lặp (đọc thẳng file: không cần Godot import trước). Không có file ➔ null.
static func load_loop_wav(path: String) -> AudioStreamWAV:
	var abs_path: String = ProjectSettings.globalize_path(path)
	if not FileAccess.file_exists(abs_path):
		return null
	var stream: AudioStreamWAV = AudioStreamWAV.load_from_file(abs_path)
	if stream == null:
		return null
	stream.loop_mode = AudioStreamWAV.LOOP_FORWARD
	stream.loop_begin = 0
	stream.loop_end = int(stream.get_length() * stream.mix_rate)
	return stream


## Tiếng ồn nền (máy lạnh, người nói xa, chuông loa, xe đẩy) + tiếng bíp 3D ở từng máy theo dõi (vital_monitor),
## mỗi máy lệch nhịp / cao độ một chút để không kêu đồng loạt.
func _setup_hospital_ambience() -> void:
	var bed: AudioStreamWAV = Main.load_loop_wav(AMBIENCE_PATH)
	if bed != null:
		var p := AudioStreamPlayer.new()
		p.name = "HospitalAmbience"
		p.stream = bed
		p.volume_db = AMBIENCE_DB
		add_child(p)
		p.play(randf() * bed.get_length())
		_ambience_players.append(p)
	var beep: AudioStreamWAV = Main.load_loop_wav(MONITOR_BEEP_PATH)
	if beep == null or room == null:
		return
	for n: Node in room.find_children("*", "Node3D", true, false):
		if not n.scene_file_path.ends_with("vital_monitor.tscn"):
			continue
		var b := AudioStreamPlayer3D.new()
		b.name = "MonitorBeep"
		b.stream = beep
		b.volume_db = MONITOR_BEEP_DB
		b.unit_size = 1.5
		b.max_distance = 9.0
		b.pitch_scale = randf_range(0.92, 1.08)
		b.position = Vector3(0.0, 1.2, 0.0)
		n.add_child(b)
		b.play(randf() * beep.get_length())
		_ambience_players.append(b)
	print("[Ambience] nền=%s, máy theo dõi có tiếng bíp=%d" % [bed != null, _ambience_players.size() - (1 if bed != null else 0)])


func set_ambience_enabled(on: bool) -> void:
	_ambience_on = on
	for p: Node in _ambience_players:
		if is_instance_valid(p):
			p.set("stream_paused", not on)
	overlay.set_status("Tiếng ồn bệnh viện: %s (phím M)" % ("bật" if on else "tắt"))


# ===================== Tương tác khi dẫn đường (docs/UPGRADE_HRI_PLAN.md mục A–C) =====================

func _setup_announcer() -> void:
	if announcer != null:
		return
	announcer = RobotAnnouncer.new()
	announcer.name = "RobotAnnouncer"
	announcer.tts = get_node_or_null(^"TtsClient") as TtsClient
	announcer.voice = robot.get_node_or_null(^"Head/Voice") as RobotVoice
	announcer.subtitle = get_node_or_null(^"UI/Subtitle") as Subtitle
	announcer.is_busy = func() -> bool: return Main.is_dialogue_state(state)
	add_child(announcer)
	announcer.line_started.connect(_on_line_started)
	announcer.line_finished.connect(_on_line_finished)


## DialogueManager hỏi: có nhận lượt nói của bệnh nhân đang được dẫn không (trả lời "có dẫn tiếp không").
func accepts_patient_turn() -> bool:
	return _ask_pending


func _escort_holding() -> bool:
	return active_escorted_patient != null and _rl_bridge != null \
			and _rl_bridge.task_phase == RLEnvBridgeScript.TaskPhase.GUIDE_ESCORT \
			and (escort_monitor.hold_robot() or _waiting_enroll())


## Đầu ca dẫn mà chưa ghi nhớ được mặt (server có ReID) ➔ đứng lại, quay về phía bệnh nhân, nhờ họ nhìn về robot.
func _waiting_enroll() -> bool:
	return _reid_supported and not identity.is_enrolled() and _escort_t <= ENROLL_WAIT_S


## Bắt đầu ca dẫn: đặt lại giám sát, lên lịch bệnh nhân tụt lại (mô phỏng), robot mời đi theo.
func _begin_escort_hri() -> void:
	_reset_escort_hri()
	_anon_seen_s = _now_s()            # vừa nói chuyện xong, người bệnh đang đứng cạnh
	_anon_dist_m = 1.5
	_id_stats = _new_id_stats()
	_id_seen_s = _now_s()            # vừa nói chuyện xong, đang thấy
	_id_dist_m = 1.5
	if active_escorted_patient != null:
		_id_est_pos = active_escorted_patient.global_position
	print("[ESCORT_ID] bắt đầu dẫn: %d mẫu ghi nhớ (hụt %d frame: %s), reid=%s" % [identity.templates.size(), _enroll_misses, _enroll_reasons, _reid_supported])
	var p: CharacterBody3D = active_escorted_patient
	if p == null:
		return
	if _rng.randf() < patient_lag_prob:
		_lag_at_s = _rng.randf_range(LAG_START_RANGE_S.x, LAG_START_RANGE_S.y)
		_lag_dur_s = _rng.randf_range(LAG_DURATION_RANGE_S.x, LAG_DURATION_RANGE_S.y)
	if announcer != null:
		announcer.say("start")


func _reset_escort_hri() -> void:
	escort_monitor.reset()
	_asked_look = false
	_ask_pending = false
	_escort_t = 0.0
	_lag_at_s = -1.0
	_goal_announced_key = ""
	_after_line.clear()
	if active_escorted_patient != null:
		active_escorted_patient.set_meta("escort_hold_s", 0.0)
	if announcer != null:
		announcer.cancel()
	if robot != null and robot.arms != null:
		robot.arms.lower()


func _update_escort_hri(p: CharacterBody3D, d_tether: float, delta: float) -> void:
	_escort_t += delta
	_record_coverage(p, delta)
	if _lag_at_s >= 0.0 and _escort_t >= _lag_at_s:
		_lag_at_s = -1.0
		p.set_meta("escort_hold_s", _lag_dur_s)
		print("[ESCORT_HRI] t=%.1fs bệnh nhân dừng lại %.1fs (mô phỏng tụt lại)" % [_escort_t, _lag_dur_s])
	if _waiting_enroll():
		if not _asked_look:
			_asked_look = true
			_say("look_at_me")
			print("[ESCORT_ID] chưa ghi nhớ được mặt ➔ nhờ bệnh nhân nhìn về robot")
		_turn_towards(p.global_position)
		return
	if Main.is_dialogue_state(state) or _escort_t < ESCORT_GRACE_S:
		return
	var vis: bool = _vision_tether()
	var seen: bool
	var dist: float
	if vis:
		# Robot thật: chỉ biết bệnh nhân qua camera (box khớp người đã ghi nhớ).
		var unseen: float = _now_s() - _id_seen_s
		seen = unseen <= ID_SEEN_S
		dist = _id_dist_m if unseen <= ID_ASSUME_FAR_S else maxf(_id_dist_m, EscortMonitor.FAR_M + 0.5)
	else:
		# Không có ReID (yolo_server --no-reid / thiếu model) ➔ dùng vị trí thật trong sim.
		seen = has_line_of_sight(robot, p)
		dist = d_tether
	for ev: String in escort_monitor.step(dist, seen, delta):
		_on_escort_event(ev, p, dist)
	if escort_monitor.hold_robot():
		# Đứng chờ, quay mặt về phía bệnh nhân (camera nhìn thấy họ, cử chỉ hướng về họ).
		var target: Vector3 = _id_est_pos if vis else p.global_position
		var to: Vector3 = target - robot.global_position
		var err: float = wrapf(atan2(-to.x, -to.z) - robot.rotation.y, -PI, PI)
		var w: float = clampf(HOLD_TURN_GAIN * err, -Config.OMEGA_MAX, Config.OMEGA_MAX) if absf(err) > 0.12 else 0.0
		if vis and w == 0.0 and not seen:
			w = SEARCH_TURN_OMEGA        # đã quay về chỗ thấy lần cuối mà không có ai ➔ quay tìm
		robot.set_command(0.0, w)
		state = Config.RobotState.APPROACH
	elif escort_monitor.state == EscortMonitor.State.OK:
		_maybe_announce_goal()


func _on_escort_event(ev: String, p: CharacterBody3D, d_tether: float) -> void:
	print("[ESCORT_HRI] t=%.1fs %s (tether=%.2fm, nhắc=%d, hỏi=%d)" % [_escort_t, ev, d_tether, escort_monitor.reminds, escort_monitor.asks])
	match ev:
		"remind", "remind_again":
			_say(ev)
			robot.arms.wave(2.0)
			var hold: float = float(p.get_meta("escort_hold_s", 0.0))
			if hold > 1.0 and _rng.randf() < REMIND_HEED_PROB:
				p.set_meta("escort_hold_s", 1.0)       # nghe nhắc ➔ bước tiếp
		"resumed":
			_say("continue_ok")
			robot.arms.point_at(_leg_target(_rl_bridge.goal_position) + Vector3(0, 1.0, 0), 2.5)
		"ask_continue":
			_say_then("ask_continue", {}, _start_answer_turn.bind(p))
		"no_answer":
			# Hỏi mà không nghe được câu trả lời: thả lượt nói treo (nếu có) rồi xử như câu không rõ.
			_ask_pending = false
			var p_npc: Npc = _npc_of(p)
			if p_npc != null and p_npc.state != Npc.IDLE:
				p_npc.reset()
			_on_continue_answer("")
		"give_up":
			_end_escort_declined(p)


## Robot nói xong câu hỏi ➔ bệnh nhân trả lời bằng clip thật (tiep_tuc / dung_lai / cam_on) ➔ STT ➔ PhoBERT.
func _start_answer_turn(p: CharacterBody3D) -> void:
	if p == null or p != active_escorted_patient or dialogue == null:
		return
	var intent: String = _pick_weighted(ANSWER_WEIGHTS)
	var p_npc: Npc = (p.get_npc() if p.has_method("get_npc") else p.get_node_or_null(^"Npc")) as Npc
	if p_npc != null:
		dialogue.register_npc(p_npc)
		_ask_pending = true
		var clip: String = p_npc.say_intent(intent)
		if clip != "" and Main.is_dialogue_state(state):
			print("[ESCORT_HRI] bệnh nhân trả lời bằng clip '%s' (kịch bản: %s)" % [clip, intent])
			return
		_ask_pending = false
		if clip != "":
			p_npc.reset()
	# Không phát được clip ➔ coi như không nghe được câu trả lời.
	_on_continue_answer("")


func _on_continue_answer(intent: String) -> void:
	var p: CharacterBody3D = active_escorted_patient
	var res: String = escort_monitor.answer(intent)
	print("[ESCORT_HRI] intent trả lời='%s' ➔ %s" % [intent, res])
	if p == null:
		return
	match res:
		"continue":
			p.set_meta("escort_hold_s", 0.0)
			robot.arms.point_at(_leg_target(_rl_bridge.goal_position) + Vector3(0, 1.0, 0), 2.5)
		"rest":
			p.set_meta("escort_hold_s", _rng.randf_range(REST_HOLD_S.x, REST_HOLD_S.y))   # nghỉ chân ngắn rồi đi tiếp
		"ask_again":
			_say_then("ask_again", {}, _start_answer_turn.bind(p))
		"end":
			_end_escort_declined(p)


## Bệnh nhân không muốn dẫn tiếp / không trả lời: chào, thả bệnh nhân, robot về / tìm người kế tiếp.
func _end_escort_declined(p: CharacterBody3D) -> void:
	var cur_floor: int = Room.get_floor(robot.global_position)
	print("[ESCORT_HRI] kết thúc ca dẫn theo ý bệnh nhân (tầng %d)" % cur_floor)
	_log_id_stats("bệnh nhân dừng")
	if p != null:
		p.set_meta("escort_hold_s", 0.0)
		p.set_meta("helped", true)
		p.set("pardon_enabled", true)
		if p.has_method("end_escort"):
			p.end_escort()
	active_escorted_patient = null
	_rl_bridge.task_phase = RLEnvBridgeScript.TaskPhase.SEARCH_PATIENT
	robot.set_command(0.0, 0.0)
	state = Config.RobotState.ARRIVED
	overlay.set_status("Bệnh nhân không cần dẫn tiếp")
	_say_then("stop_ok", {}, _finish_escort.bind(cur_floor))


## Tới nơi: chỉ vào phòng + "Đã tới … rồi ạ", nói xong bệnh nhân cảm ơn. Không nói được ➔ false.
func _announce_arrival(p: CharacterBody3D, cur_floor: int) -> bool:
	if announcer == null:
		return false
	var place: String = announcer.place_name(_rl_bridge.goal_key)
	if place == "" or not announcer.say("arrive", {"place": place}):
		return false
	robot.arms.point_at(_rl_bridge.goal_position + Vector3(0, 1.0, 0), 3.5)
	robot.set_command(0.0, 0.0)
	state = Config.RobotState.ARRIVED
	_after_line["arrive"] = _thanks_or_finish.bind(p, cur_floor)
	return true


func _thanks_or_finish(p: CharacterBody3D, cur_floor: int) -> void:
	# Bệnh nhân nói lời cảm ơn bằng clip thật ➔ robot nghe (STT/NLU) và đáp (TTS); xong lượt mới về / reset.
	if p != null and _start_thanks_turn(p):
		_after_thanks = _finish_escort.bind(cur_floor)
		return
	if dialogue != null and dialogue.has_method("show_subtitle"):
		dialogue.show_subtitle("Bệnh nhân", "Cảm ơn robot, tôi đã tới nơi khám!", 4.0)
	_finish_escort(cur_floor)


## Đích chặng hiện tại lọt vào tầm nhìn (≤ 7 m, phía trước) ➔ chỉ tay + "Phòng … ở phía trước bên phải ạ", một lần mỗi đích.
func _maybe_announce_goal() -> void:
	if announcer == null or _rl_bridge == null:
		return
	var key: String = _rl_bridge.goal_key
	if key == "" or key == _goal_announced_key:
		return
	var gp: Vector3 = _rl_bridge.goal_position
	var d: float = Human.flat_distance(robot.global_position, gp)
	if d > GOAL_ANNOUNCE_MAX_M or d < GOAL_ANNOUNCE_MIN_M or not _point_visible(gp + Vector3(0, 1.0, 0)):
		return
	var local: Vector3 = robot.to_local(gp)
	var ang: float = atan2(local.x, -local.z)
	if absf(ang) > GOAL_ANNOUNCE_MAX_ANGLE:
		return
	var place: String = announcer.place_name(key)
	if place == "":
		return
	_goal_announced_key = key
	var line: String = Main.goal_side_line(ang)
	_say(line, {"place": place})
	robot.arms.point_at(gp + Vector3(0, 1.0, 0), 4.0)


static func goal_side_line(angle_rad: float) -> String:
	if absf(angle_rad) < deg_to_rad(25.0):
		return "goal_ahead"
	return "goal_right" if angle_rad > 0.0 else "goal_left"


## Tia từ đầu robot tới điểm `world_pos` không bị tường / đồ vật chắn (bỏ qua robot và bệnh nhân).
func _point_visible(world_pos: Vector3) -> bool:
	var space := get_world_3d().direct_space_state
	if space == null:
		return true
	var q := PhysicsRayQueryParameters3D.create(robot.global_position + Vector3(0, 1.2, 0), world_pos)
	var ex: Array[RID] = [robot.get_rid()]
	if active_escorted_patient != null:
		ex.append(active_escorted_patient.get_rid())
	q.exclude = ex
	return space.intersect_ray(q).is_empty()


func _say(key: String, args: Dictionary = {}) -> bool:
	return announcer != null and announcer.say(key, args)


## Nói câu `key` rồi gọi `then`; không nói được ➔ gọi `then` ngay.
func _say_then(key: String, args: Dictionary, then: Callable) -> void:
	if _say(key, args):
		_after_line[key] = then
	else:
		then.call()


func _on_line_started(key: String) -> void:
	if key == "start" and _rl_bridge != null and robot.arms != null:
		var tgt: Vector3 = _leg_target(_rl_bridge.goal_position) + Vector3(0, 1.0, 0)
		if _point_visible(tgt):
			robot.arms.point_at(tgt, 3.0)
		else:
			robot.arms.wave(1.6)


func _on_line_finished(key: String) -> void:
	if _after_line.has(key):
		var cb: Callable = _after_line[key]
		_after_line.erase(key)
		if cb.is_valid():
			cb.call()


func _pick_weighted(weights: Dictionary) -> String:
	var total: float = 0.0
	for k: String in weights:
		total += float(weights[k])
	var r: float = _rng.randf() * total
	for k: String in weights:
		r -= float(weights[k])
		if r <= 0.0:
			return k
	return str(weights.keys()[0])


# ===================== Nhận diện đúng bệnh nhân (docs/UPGRADE_HRI_PLAN.md D3–D5) =====================

func _now_s() -> float:
	return Time.get_ticks_msec() / 1000.0


## Client YOLO thứ hai cho camera sau: chỉ chạy khi đang dẫn, 10 Hz, frame nào cũng có ReID.
## Kết quả chỉ dùng để nhận ra / đo khoảng cách bệnh nhân, không đưa vào policy.
func _setup_rear_vision() -> void:
	if rear_vision != null or not robot.has_method("get_rear_viewport"):
		return
	rear_vision = VisionClient.new()
	rear_vision.name = "RearVisionClient"
	rear_vision.enabled = false
	rear_vision.capture_hz = REAR_CAPTURE_HZ
	rear_vision.reid_every_n = 1
	add_child(rear_vision)
	rear_vision.setup(robot.get_rear_viewport())
	rear_vision.detections_received.connect(_on_rear_detections)
	overlay.set_rear_viewport(robot.get_rear_viewport())


func _update_vision_modes() -> void:
	vision.reid_mode = _wanted_reid_mode()
	var escorting: bool = active_escorted_patient != null
	if rear_vision != null:
		rear_vision.enabled = escorting
		rear_vision.reid_mode = "track" if escorting else ""
	if robot.has_method("set_rear_camera_enabled") and robot.is_rear_camera_enabled() != escorting:
		robot.set_rear_camera_enabled(escorting)
		if not escorting:
			_latest_rear_dets = []
			_id_rear_det = {}
		overlay.set_rear_active(escorting)


## "enroll" khi đang nói chuyện với một bệnh nhân (chưa dẫn ai), "track" khi đang dẫn, "" còn lại (không tốn ReID).
func _wanted_reid_mode() -> String:
	if active_escorted_patient != null:
		return "track"
	if Main.is_dialogue_state(state):
		return "enroll"
	if _served_active():
		return "track"           # nhận ra người vừa giúp (không hỏi lại họ)
	return ""


func _served_active() -> bool:
	return not _served_id.templates.is_empty() and _now_s() < _served_until_s


func _remember_served() -> void:
	_served_id.reset()
	if identity.templates.is_empty():
		return
	_served_id.templates = identity.templates.duplicate()
	_served_id.last_box = identity.last_box.duplicate()
	_served_id.last_seen_s = identity.last_seen_s.duplicate()
	_served_until_s = _now_s() + SERVED_FORGET_S


func _forget_served() -> void:
	_served_id.reset()
	_served_until_s = -INF
	_served_det = {}


## Giám sát khoảng cách theo camera (thật) thay vì vị trí trong sim: cần server có ReID và đã ghi nhớ được người.
func _vision_tether() -> bool:
	return _reid_supported and identity.is_enrolled()


func _check_reid_support(dets: Array) -> void:
	if _reid_supported:
		return
	for dv: Variant in dets:
		if (dv as Dictionary).has("emb"):
			_reid_supported = true
			return


func _update_identity(dets: Array) -> void:
	_check_reid_support(dets)
	var now: float = _now_s()
	_id_det = {}
	_served_det = {}
	if active_escorted_patient == null and not Main.is_dialogue_state(state):
		if _served_active():
			var sr: Dictionary = _served_id.match(dets, now, "front")
			if not sr.is_empty():
				_served_det = sr["det"]
		elif not _served_id.templates.is_empty():
			_forget_served()
	match vision.reid_mode:
		"enroll":
			var spk: CharacterBody3D = get_active_speaking_patient()
			# Chỉ ghi nhớ khi biết hướng người nói (micro mảng) và hướng đó nằm trong khung. Người nói còn ngoài khung
			# ➔ chờ robot quay tới; không lấy "người có mặt gần giữa khung" (hay là người đứng cạnh, vd bác sĩ).
			var spk_cx: float = _speaker_cx(spk) if spk != null else -1.0
			if spk_cx < 0.0:
				if _has_enroll_frame(dets):
					_enroll_misses += 1
					_enroll_reasons["speaker_out_of_frame"] = int(_enroll_reasons.get("speaker_out_of_frame", 0)) + 1
			elif identity.enroll_from(dets, now, true, spk_cx):
				_remember_face_crop(PatientIdentity.box_of_last(identity, dets), spk)   # ô "ghi nhớ" = mẫu mới nhất
				print("[ESCORT_ID] ghi nhớ mẫu %d (lúc nói chuyện): %s" % [identity.n_enroll,
						_template_truth(get_active_speaking_patient(), PatientIdentity.box_of_last(identity, dets), false)])
			elif spk_cx >= 0.0 and _has_enroll_frame(dets) and identity.last_reject != "":
				_enroll_misses += 1
				_enroll_reasons[identity.last_reject] = int(_enroll_reasons.get(identity.last_reject, 0)) + 1
		"track":
			# Chưa ghi nhớ được lúc nói chuyện ➔ robot đứng chờ (ENROLL_WAIT_S), quay về phía bệnh nhân; vẫn chỉ ghi nhớ
			# khi thấy mặt ở đúng hướng người đó (bỏ kiểu ghi nhớ bù "người to nhất giữa khung": hay nhầm người).
			if not identity.is_enrolled() and _escort_t <= ENROLL_WAIT_S and active_escorted_patient != null \
					and identity.enroll_from(dets, now, true, _speaker_cx(active_escorted_patient)):
				_remember_face_crop(PatientIdentity.box_of_last(identity, dets), active_escorted_patient)
				print("[ESCORT_ID] ghi nhớ đầu ca dẫn (thấy mặt): %s" % _template_truth(active_escorted_patient, PatientIdentity.box_of_last(identity, dets), false))
			var n_t: int = identity.templates.size()
			var r: Dictionary = identity.match(dets, now, "front")
			if not r.is_empty():
				_id_det = r["det"]
				_on_identity_seen(_id_det, false, now)
				if identity.templates.size() != n_t or identity.templates_changed:
					print("[ESCORT_ID] học thêm mẫu (trước, %s %.2f): %s" % [r["how"], r["score"], _template_truth(active_escorted_patient, _id_det, false)])
			_record_id_stats(dets, _id_det, "front")
	overlay.set_tracked(_id_det, _id_rear_det)


func _on_rear_detections(_frame_id: int, dets: Array, _infer_ms: float) -> void:
	_latest_rear_dets = dets
	_check_reid_support(dets)
	_id_rear_det = {}
	if active_escorted_patient != null:
		var now: float = _now_s()
		var n_t: int = identity.templates.size()
		var r: Dictionary = identity.match(dets, now, "rear")
		if not r.is_empty():
			_id_rear_det = r["det"]
			_on_identity_seen(_id_rear_det, true, now)
			if identity.templates.size() != n_t or identity.templates_changed:
				print("[ESCORT_ID] học thêm mẫu (sau, %s %.2f): %s" % [r["how"], r["score"], _template_truth(active_escorted_patient, _id_rear_det, true)])
		_record_id_stats(dets, _id_rear_det, "rear")
	overlay.show_rear_detections(dets)
	overlay.show_faces(vision.last_faces, rear_vision.last_faces)
	overlay.set_tracked(_id_det, _id_rear_det)


func _on_identity_seen(det: Dictionary, rear: bool, now: float) -> void:
	_id_seen_s = now
	var b: Dictionary = PatientIdentity.box_of(det)
	_id_dist_m = clampf(1.20 / maxf(float(b["h"]), 0.05), 0.5, 8.0)   # như tether của policy
	_id_est_pos = _det_world_pos(b["cx"], _id_dist_m, rear)


## Frame ghi nhớ có box patient kèm embedding (để đếm lần ghi nhớ hụt: không thấy mặt / lệch khung).
static func _has_enroll_frame(dets: Array) -> bool:
	for dv: Variant in dets:
		var d: Dictionary = dv
		if PatientIdentity.is_patient(d) and d.has("emb"):
			return true
	return false


## Ước lượng vị trí người từ tâm box (góc ngang trong FOV camera) và khoảng cách. Camera sau nhìn ngược lại.
func _det_world_pos(cx: float, dist_m: float, rear: bool = false) -> Vector3:
	var a: float = atan((cx * 2.0 - 1.0) * tan(ROBOT_CAM_HFOV * 0.5))    # > 0: bên phải của camera
	var yaw: float = robot.rotation.y
	var fwd := Vector3(-sin(yaw), 0.0, -cos(yaw))
	var right := Vector3(cos(yaw), 0.0, -sin(yaw))
	if rear:
		fwd = -fwd
		right = -right
	return robot.global_position + (fwd * cos(a) + right * sin(a)) * dist_m


## Box đưa cho policy RL: khi đang dẫn và nhận diện được, chỉ box của bệnh nhân mình giữ lớp `patient`;
## bệnh nhân khác thành vật cản (class 100) ➔ policy không bám nhầm người, không phải train lại.
func _policy_dets() -> Array:
	if active_escorted_patient == null:
		return Main.mark_served(_latest_yolo_dets, _served_det)
	if not _vision_tether():
		return _latest_yolo_dets
	var out: Array = []
	for dv: Variant in _latest_yolo_dets:
		var d: Dictionary = dv
		if PatientIdentity.is_patient(d) and not is_same(d, _id_det):
			var o: Dictionary = d.duplicate()
			o.erase("emb")
			o["class_id"] = 100
			o["class_name"] = "patient_other"
			out.append(o)
		else:
			out.append(d)
	return out


## Box của người vừa được giúp ➔ vật cản (như bệnh nhân khác lúc đang dẫn): policy không tiếp cận / hỏi lại họ.
static func mark_served(dets: Array, served: Dictionary) -> Array:
	if served.is_empty():
		return dets
	var out: Array = []
	for dv: Variant in dets:
		var d: Dictionary = dv
		if PatientIdentity.is_patient(d) and is_same(d, served):
			var o: Dictionary = d.duplicate()
			o.erase("emb")
			o["class_id"] = 100
			o["class_name"] = "patient_other"
			o["served"] = true
			out.append(o)
		else:
			out.append(d)
	return out


func _new_id_stats() -> Dictionary:
	var cam := func() -> Dictionary:
		return {"frames": 0, "visible": 0, "matched": 0, "wrong_other": 0, "missed": 0, "base_frames": 0, "base_wrong_other": 0}
	return {"t": 0.0, "t_seen": 0.0, "t_gt": 0.0, "t_gt_front": 0.0, "front": cam.call(), "rear": cam.call()}


## Toạ độ chuẩn hoá của bệnh nhân thật trong ảnh camera (chiếu vị trí thật); không thấy ➔ (-1, -1).
func _gt_uv(p: CharacterBody3D, rear: bool, height_m: float = 0.9) -> Vector2:
	var cam: Camera3D = robot.get_rear_camera() if rear else robot.get_camera_viewport().get_node(^"RobotCamera") as Camera3D
	var world: Vector3 = p.global_position + Vector3(0, height_m, 0)
	if cam.is_position_behind(world) or not has_line_of_sight(robot, p):
		return Vector2(-1, -1)
	var px: Vector2 = cam.unproject_position(world)
	var uv := Vector2(px.x / float(Config.IMG_W), px.y / float(Config.IMG_H))
	return uv if uv.x >= 0.0 and uv.x <= 1.0 and uv.y >= 0.0 and uv.y <= 1.0 else Vector2(-1, -1)


## Chẩn đoán: box `d` có chứa người `p` thật không ("đúng người" / "NHẦM người" / "không rõ").
func _template_truth(p: CharacterBody3D, d: Dictionary, rear: bool) -> String:
	if p == null or d.is_empty():
		return "không rõ"
	# Hông / ngực / mặt: robot ngẩng đầu nhìn mặt người đứng gần ➔ hông có thể nằm dưới đáy ảnh.
	var any_visible: bool = false
	for h: float in [0.9, 1.3, 1.55]:
		var uv: Vector2 = _gt_uv(p, rear, h)
		if uv.x < 0.0:
			continue
		any_visible = true
		if Main._uv_in(uv, d):
			return "đúng người"
	return "NHẦM người" if any_visible else "không rõ (người thật ngoài khung)"


static func _uv_in(uv: Vector2, d: Dictionary) -> bool:
	return uv.x >= float(d["x1"]) and uv.x <= float(d["x2"]) and uv.y >= float(d["y1"]) and uv.y <= float(d["y2"])


## Số liệu D5 mỗi frame: chọn nhầm người khác khi bệnh nhân thật đang trong khung? So với cách cũ (box patient conf cao nhất).
func _record_id_stats(dets: Array, matched: Dictionary, cam_key: String) -> void:
	var p: CharacterBody3D = active_escorted_patient
	if p == null or _id_stats.is_empty():
		return
	var s: Dictionary = _id_stats[cam_key]
	s["frames"] += 1
	var uv: Vector2 = _gt_uv(p, cam_key == "rear")
	var visible: bool = uv.x >= 0.0
	if visible:
		s["visible"] += 1
	if not matched.is_empty():
		s["matched"] += 1
		if visible and not Main._uv_in(uv, matched):
			s["wrong_other"] += 1
	elif visible:
		s["missed"] += 1
	var base: Dictionary = {}
	for dv: Variant in dets:
		var d: Dictionary = dv
		if PatientIdentity.is_patient(d) and (base.is_empty() or float(d["conf"]) > float(base["conf"])):
			base = d
	if not base.is_empty():
		s["base_frames"] += 1
		if visible and not Main._uv_in(uv, base):
			s["base_wrong_other"] += 1


## Mỗi frame vật lý khi đang dẫn: robot có "thấy" bệnh nhân không, so với lúc bệnh nhân thật nằm trong tầm camera.
func _record_coverage(p: CharacterBody3D, delta: float) -> void:
	if _id_stats.is_empty():
		return
	_id_stats["t"] += delta
	if _now_s() - _id_seen_s <= ID_SEEN_S:
		_id_stats["t_seen"] += delta
	var in_front: bool = _gt_uv(p, false).x >= 0.0
	if in_front:
		_id_stats["t_gt_front"] += delta
	if in_front or _gt_uv(p, true).x >= 0.0:
		_id_stats["t_gt"] += delta


func _log_id_stats(tag: String) -> void:
	if _id_stats.is_empty():
		return
	var s: Dictionary = _id_stats
	var t: float = maxf(float(s["t"]), 1e-3)
	print("[ESCORT_ID] %s: %.0fs | bệnh nhân trong tầm camera trước %.0f%%, trước+sau %.0f%% | robot thấy (khớp) %.0f%% | mẫu=%d, ghi nhớ hụt=%d" % [
		tag, t, 100.0 * s["t_gt_front"] / t, 100.0 * s["t_gt"] / t, 100.0 * s["t_seen"] / t,
		identity.templates.size(), _enroll_misses])
	for k: String in ["front", "rear"]:
		var c: Dictionary = s[k]
		print("[ESCORT_ID]   %s: frames=%d visible=%d matched=%d nhầm_người=%d bỏ_sót=%d | cũ(conf max): chọn=%d nhầm_người=%d" % [
			k, c["frames"], c["visible"], c["matched"], c["wrong_other"], c["missed"], c["base_frames"], c["base_wrong_other"]])


## Gắn mặt cho mọi NPC có texture (chờ 2 frame để xương có tư thế). Trả số NPC được gắn.
func _apply_face_paste() -> int:
	await get_tree().process_frame
	await get_tree().process_frame
	var targets: Array = []
	if human != null:
		targets.append([human, "human_m06"])
	var furn: Node = room.get_node_or_null(^"Furniture") if room != null else null
	if furn != null:
		for c: Node in furn.get_children():
			if c is HospitalNpc:
				var look: String = str((c as HospitalNpc).tex_dir).get_base_dir().get_file()
				targets.append([c, FACE_LOOK_ALIASES.get(look, look)])
	var n: int = 0
	for t: Array in targets:
		if NpcFace.attach(t[0], NpcFace.load_face(t[1])) != null:
			n += 1
	print("[FacePaste] dán mặt người thật cho %d/%d NPC%s" % [n, targets.size(),
			"" if n > 0 else " (không có texture: chạy sim_vision/python -m identity.make_npc_faces)"])
	return n



# ===================== Ô "MẶT" bên trái camera trước =====================

## Vùng cắt quanh box mặt (toạ độ chuẩn hoá) trong ảnh cỡ `size`: rộng thêm 30%, tỉ lệ 4:5, kẹp trong ảnh.
static func face_crop_rect(face: Dictionary, size: Vector2i) -> Rect2i:
	var cx: float = (float(face["x1"]) + float(face["x2"])) * 0.5 * size.x
	var cy: float = (float(face["y1"]) + float(face["y2"])) * 0.5 * size.y
	var w: float = maxf((float(face["x2"]) - float(face["x1"])) * size.x, (float(face["y2"]) - float(face["y1"])) * size.y * 0.8) * 1.3
	var h: float = w * 1.25
	var r := Rect2i(int(round(cx - w * 0.5)), int(round(cy - h * 0.5)), maxi(int(round(w)), 2), maxi(int(round(h)), 2))
	return r.intersection(Rect2i(Vector2i.ZERO, size))


func _crop_camera(face: Dictionary, rear: bool = false) -> Texture2D:
	# Ảnh đúng frame đã phát hiện mặt (không phải ảnh viewport lúc kết quả về: robot quay ➔ lệch, cắt ra tường).
	var vc: VisionClient = rear_vision if rear else vision
	var img: Image = vc.last_image if vc != null else null
	if img == null or img.is_empty():
		var vp: SubViewport = robot.get_rear_viewport() if rear else robot.get_camera_viewport()
		img = vp.get_texture().get_image()
	if img == null or img.is_empty():
		return null
	var r: Rect2i = Main.face_crop_rect(face, img.get_size())
	if r.size.x < 2 or r.size.y < 2:
		return null
	return ImageTexture.create_from_image(img.get_region(r))


func _update_face_panel(now_ms: int) -> void:
	if now_ms - _last_face_crop_ms < FACE_CROP_EVERY_MS:
		return
	_last_face_crop_ms = now_ms
	var rear_on: bool = rear_vision != null and rear_vision.enabled
	overlay.set_live_faces(_live_face_items(vision.last_faces, _id_det, false),
			_live_face_items(rear_vision.last_faces, _id_rear_det, true) if rear_on else [])


## Ô mặt: các mặt lớn nhất, mỗi mặt gắn "đang dắt" (1) nếu là mặt của box ReID đang khớp bệnh nhân, "khác" (0) nếu
## đang dẫn mà không phải, -1 khi không dẫn ai.
func _live_face_items(faces_in: Array, tracked_det: Dictionary, rear: bool) -> Array:
	var faces: Array = faces_in.duplicate()
	faces.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) > (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
	var mine_face: Dictionary = Main.face_in_box(faces, tracked_det)
	var escorting: bool = active_escorted_patient != null
	var items: Array = []
	for f: Dictionary in faces.slice(0, ViewOverlay.LIVE_FACES_MAX):
		var t: Texture2D = _crop_camera(f, rear)
		if t != null:
			items.append({"tex": t, "conf": f["conf"], "mine": (1 if f == mine_face and not mine_face.is_empty() else 0) if escorting else -1})
	return items


## Lúc ghi nhớ: cắt mặt nằm trong nửa trên box bệnh nhân vừa ghi nhớ ➔ hiện ở ô "ghi nhớ".
func _remember_face_crop(det: Dictionary, who: CharacterBody3D = null) -> void:
	var f: Dictionary = Main.face_in_box(vision.last_faces, det)
	if f.is_empty():
		print("[ESCORT_ID] mặt ghi nhớ: không có mặt trong box")
		return
	var owner: CharacterBody3D = _face_owner(f)
	print("[ESCORT_ID] mặt ghi nhớ: %s (mặt của %s, người nói %s)" % [
			"đúng" if who != null and owner == who else ("không rõ" if owner == null or who == null else "NHẦM"),
			owner.name if owner != null else "?", who.name if who != null else "?"])
	var t: Texture2D = _crop_camera(f)
	if t != null:
		overlay.set_remembered_face(t)


## Mặt của người trong box `det`: tâm mặt nằm trong nửa trên box, gần đỉnh-giữa box nhất (box chồng nhau thì không
## lấy nhầm mặt người đứng sau). Không có ➔ {}.
static func face_in_box(faces: Array, det: Dictionary) -> Dictionary:
	if det.is_empty():
		return {}
	var x1: float = float(det["x1"])
	var x2: float = float(det["x2"])
	var y1: float = float(det["y1"])
	var y2: float = float(det["y2"])
	var best: Dictionary = {}
	var best_d: float = INF
	for f: Dictionary in faces:
		var fx: float = (float(f["x1"]) + float(f["x2"])) * 0.5
		var fy: float = (float(f["y1"]) + float(f["y2"])) * 0.5
		if fx < x1 or fx > x2 or fy < y1 or fy > y1 + 0.5 * (y2 - y1):
			continue
		var d: float = Vector2(fx - 0.5 * (x1 + x2), (fy - y1) * 0.5).length()
		if d < best_d:
			best_d = d
			best = f
	return best



# ===================== Quay về phía người đang nói (bước 1, bản luật trước khi có RL V3) =====================

## Hướng giọng nói ➔ quay tại chỗ (v = 0). Ngoài đời hướng lấy từ micro mảng (DOA); trong sim lấy vị trí NPC đang nói.
func _face_speaker() -> void:
	var spk: CharacterBody3D = get_active_speaking_patient()
	if spk == null:
		robot.set_command(0.0, 0.0)
		return
	_turn_towards(spk.global_position)


func _turn_towards(target: Vector3) -> void:
	var to: Vector3 = target - robot.global_position
	var err: float = wrapf(atan2(-to.x, -to.z) - robot.rotation.y, -PI, PI)
	robot.set_command(0.0, clampf(SPEAKER_TURN_GAIN * err, -Config.OMEGA_MAX, Config.OMEGA_MAX) if absf(err) > 0.08 else 0.0)


## Hướng người `p` chiếu lên ảnh camera trước (tâm ngang 0..1); ngoài tầm nhìn ➔ -1 (chưa ghi nhớ, chờ robot quay).
func _speaker_cx(p: Node3D) -> float:
	var local: Vector3 = robot.to_local(p.global_position)
	return Main.bearing_to_cx(atan2(local.x, -local.z))


static func bearing_to_cx(a: float) -> float:
	var half: float = ROBOT_CAM_HFOV * 0.5
	if absf(a) > half * 0.9:
		return -1.0
	return 0.5 + 0.5 * tan(a) / tan(half)



# ===================== RL V3: quan sát tương tác + hành động ngẩng đầu / hỏi (docs/RL_V3_DESIGN.md) =====================

const ASK_RANGE_M: float = 2.5
const ASK_CENTRE_CX: float = 0.3
const ASK_COOLDOWN_S: float = 3.0
## true khi policy server trả hành động V3 (tilt, ask) ➔ luật quay về người nói tắt, policy chạy cả lúc nói chuyện.
var policy_v3: bool = false
var _last_ask_s: float = -INF


func _on_policy_engage(tilt: float, ask: bool) -> void:
	if not policy_v3:
		policy_v3 = true
		print("[V3] policy V3: tự quay / ngẩng đầu / hỏi (luật quay về người nói tắt)")
	robot.set_head_tilt_action(tilt)
	if ask:
		_robot_ask()


## Người đang nói: NPC trong lượt hội thoại, hoặc NPC đang phát clip (gọi robot / bắt đầu nói). Null nếu im lặng.
func _current_speaker() -> CharacterBody3D:
	if Main.is_dialogue_state(state):
		return get_active_speaking_patient()
	for p: CharacterBody3D in _all_people():
		var n: Npc = _npc_of(p)
		if n != null and n.state in [Npc.NOTICE, Npc.TALK]:
			return p
	return null


func _all_people() -> Array[CharacterBody3D]:
	var out: Array[CharacterBody3D] = []
	if human != null:
		out.append(human)
	var furn: Node = room.get_node_or_null(^"Furniture") if room != null else null
	if furn != null:
		for c: Node in furn.get_children():
			if c is HospitalNpc and (c as HospitalNpc).visible:
				out.append(c as CharacterBody3D)
	return out


static func _xz(p: Vector3) -> String:
	return "(%.2f, %.2f)" % [p.x, p.z]


func _npc_of(p: Node) -> Npc:
	return (p.get_npc() if p.has_method("get_npc") else p.get_node_or_null(^"Npc")) as Npc


var _anon_seen_s: float = -INF
var _anon_dist_m: float = 1.0


## Chiều cao (0..1) box người (YOLO patient / doctor) to nhất trong `dets`; không có ➔ 0.
static func largest_patient_h(dets: Array) -> float:
	var best: float = 0.0
	for dv: Variant in dets:
		var d: Dictionary = dv
		if int(d.get("class_id", -1)) in [0, 1] or str(d.get("class_name", "")) in ["patient", "doctor"]:
			best = maxf(best, float(d.get("y2", 0.0)) - float(d.get("y1", 0.0)))
	return best


var _voice_mem_dir: Vector3 = Vector3.ZERO
var _voice_mem_t: float = -1e9


func _v3_features() -> Dictionary:
	var out: Dictionary = {"tilt": robot.get_head_tilt_norm(), "since_ask_s": _now_s() - _last_ask_s,
			"dialogue": Main.is_dialogue_state(state),
			# robot đang bị giữ đứng yên (nói chuyện / chờ người bệnh nói) ➔ "ra lệnh đi mà không đi" không phải là kẹt
			"held": Main.is_dialogue_state(state) or state == Config.RobotState.ARRIVED}
	# Người đang dẫn ở camera trước hoặc sau (họ thường đi sau lưng robot): [khoảng cách m, đang thấy, số giây mất dấu].
	# Có ReID ➔ đúng người đã ghi nhớ. Chưa ghi nhớ được (log run_1932: người nói ngoài khung) ➔ box người to nhất ở
	# camera trước / sau, như lúc train (tether tính cả camera sau). Trước đây trường hợp này chỉ có box camera trước
	# ➔ người đi sau lưng = "mất dấu" ➔ policy bò 0.1 m/s cả đoạn tới thang máy.
	if active_escorted_patient != null:
		if _vision_tether():
			var unseen_s: float = _now_s() - _id_seen_s
			out["tether"] = [_id_dist_m, 1.0 if unseen_s <= ID_SEEN_S else 0.0, unseen_s]
		else:
			var h: float = maxf(Main.largest_patient_h(_latest_yolo_dets), Main.largest_patient_h(_latest_rear_dets))
			if h > 0.0:
				_anon_seen_s = _now_s()
				_anon_dist_m = clampf(1.20 / maxf(h, 0.05), 0.5, 5.0)
			var unseen_a: float = _now_s() - _anon_seen_s
			out["tether"] = [_anon_dist_m, 1.0 if unseen_a <= ID_SEEN_S else 0.0, unseen_a]
	var spk: CharacterBody3D = _current_speaker()
	var expect_cx: float = -1.0
	if spk != null:
		var local: Vector3 = robot.to_local(spk.global_position)
		var bearing: float = atan2(local.x, -local.z)            # + = bên phải (như DOA của micro mảng)
		out["voice"] = [1.0, bearing]
		expect_cx = Main.bearing_to_cx(bearing)
		_voice_mem_dir = Vector3(spk.global_position.x - robot.global_position.x, 0.0,
				spk.global_position.z - robot.global_position.z).normalized()
		_voice_mem_t = _now_s()
	# Hướng giọng nói nghe lần cuối, giữ theo hệ thế giới (robot thật: DOA + odometry) ➔ người ngừng nói, robot vẫn
	# biết quay về đâu.
	if _voice_mem_t > -1e8:
		var ml: Vector3 = robot.global_transform.basis.inverse() * _voice_mem_dir
		out["voice_mem"] = [1.0, atan2(ml.x, -ml.z), _now_s() - _voice_mem_t]
	# Mặt: của người đang nói (gần hướng giọng nói), không thì mặt gần giữa khung nhất.
	var best: Dictionary = {}
	var best_score: float = INF
	var faces: Array = [] if (spk != null and expect_cx < 0.0) else vision.last_faces   # người nói ngoài khung ➔ không thấy mặt họ
	for f: Dictionary in faces:
		var cx: float = (float(f["x1"]) + float(f["x2"])) * 0.5
		var sc: float = absf(cx - expect_cx) if expect_cx >= 0.0 else absf(cx - 0.5)
		if expect_cx >= 0.0 and sc > 0.15:
			continue
		if sc < best_score:
			best_score = sc
			best = f
	if not best.is_empty():
		var owner: CharacterBody3D = _face_owner(best)
		var known: float = 1.0 if (owner != null and owner.get_meta("helped", false)) \
				or not Main.face_in_box([best], _served_det).is_empty() else 0.0
		out["face"] = [1.0, (float(best["x1"]) + float(best["x2"])) * 0.5 - 0.5,
				(float(best["y1"]) + float(best["y2"])) * 0.5 - 0.5, float(best["y2"]) - float(best["y1"]), known]
	return out


## Người có mặt `face` (camera trước): đầu thật chiếu lên ảnh gần tâm mặt nhất. Sim thay cho bộ nhớ ReID / mặt của
## những người robot đã giúp (robot thật: so embedding với gallery người đã gặp).
func _face_owner(face: Dictionary) -> CharacterBody3D:
	var c := Vector2((float(face["x1"]) + float(face["x2"])) * 0.5, (float(face["y1"]) + float(face["y2"])) * 0.5)
	var tol: float = maxf(0.08, float(face["y2"]) - float(face["y1"]))
	var best: CharacterBody3D = null
	var best_d: float = INF
	for p: CharacterBody3D in _all_people():
		var seated: bool = p.get("is_seated") == true
		var uv: Vector2 = _gt_uv(p, false, 1.2 if seated else 1.6)
		if uv.x < 0.0:
			continue
		var d: float = uv.distance_to(c)
		if d < tol and d < best_d and has_line_of_sight(robot, p):   # người sau tường không phải chủ mặt
			best_d = d
			best = p
	return best


const EXCUSE_RANGE_M: float = 1.5
const EXCUSE_HALF_ANGLE: float = deg_to_rad(40.0)


## Người đang đứng ngay trước mũi robot (không phải người đang được dẫn, không ngồi): chắn lối. Không có ➔ null.
func _blocking_person() -> CharacterBody3D:
	var best: CharacterBody3D = null
	var best_d: float = EXCUSE_RANGE_M
	for p: CharacterBody3D in _all_people():
		if p == active_escorted_patient or p.get("is_seated") == true:
			continue
		var local: Vector3 = robot.to_local(p.global_position)
		var d: float = Vector2(local.x, local.z).length()
		if d <= best_d and absf(atan2(local.x, -local.z)) <= EXCUSE_HALF_ANGLE and has_line_of_sight(robot, p):
			best_d = d
			best = p
	return best


## Hành động "hỏi" của policy V3 (như lúc train, hospital_env_engage_v3._ask):
## - robot rảnh và có người chưa được giúp gần giữa khung trong ASK_RANGE_M ➔ "Cần tôi giúp gì không ạ?" + vẫy tay,
##   người đó quay lại và nói;
## - không thì với người đứng chắn ngay trước mặt (người vừa giao xong, người đi ngang) ➔ "Xin lỗi, cho tôi đi nhờ ạ",
##   người đó bước sang bên. Robot tự xin đường chứ không trông vào việc người ta tự né.
func _robot_ask() -> void:
	if Main.is_dialogue_state(state) or _now_s() - _last_ask_s < ASK_COOLDOWN_S:
		return
	_last_ask_s = _now_s()
	var busy: bool = is_task_busy()
	var target: CharacterBody3D = null
	var best: float = INF
	if not busy:
		for p: CharacterBody3D in _all_people():
			var local: Vector3 = robot.to_local(p.global_position)
			var d: float = Vector2(local.x, local.z).length()
			var cx: float = Main.bearing_to_cx(atan2(local.x, -local.z))
			if d > ASK_RANGE_M or cx < 0.0 or absf(cx - 0.5) > ASK_CENTRE_CX or p.get_meta("helped", false):
				continue
			if absf(cx - 0.5) < best and has_line_of_sight(robot, p):
				best = absf(cx - 0.5)
				target = p
	if target == null:
		var blocker: CharacterBody3D = _blocking_person()
		if blocker != null:
			print("[V3] robot xin đường \"Xin lỗi, cho tôi đi nhờ ạ\" ➔ %s" % blocker.name)
			_say_then("excuse_me", {}, func() -> void:
				if is_instance_valid(blocker) and blocker.has_method("make_way"):
					var ok: bool = blocker.make_way(robot)
					var tgt: Variant = blocker.get("_pardon_target")
					print("[V3] %s %s (đứng %s ➔ %s, robot %s)" % [blocker.name, "nhường đường" if ok else "KHÔNG nhường được",
							_xz(blocker.global_position), _xz(tgt) if ok and tgt is Vector3 else "-", _xz(robot.global_position)]))
			return
		if busy:
			return                            # đang bận mà không ai chắn lối: không nói gì
	print("[V3] robot hỏi \"Cần tôi giúp gì không ạ?\" ➔ %s" % (target.name if target != null else "không có ai"))
	if robot.arms != null:
		robot.arms.wave(1.5)
	_say_then("ask_help", {}, _ask_answered.bind(target))


func _ask_answered(target: CharacterBody3D) -> void:
	if target == null or Main.is_dialogue_state(state) or is_task_busy():
		return
	var n: Npc = _npc_of(target)
	if n != null and n.state == Npc.IDLE and n.robot != null:
		current_target_patient = target
		n._notice()                       # quay về robot rồi nói yêu cầu (luồng hội thoại thường)
