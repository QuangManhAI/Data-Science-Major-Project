class_name Main
extends Node3D
## Scene chính (G8): vision_client ➔ controller.step ➔ apply_safety ➔ robot.set_command.
## Hội thoại (G16): ở LISTEN / THINK / RESPOND robot đứng yên (v = 0, omega = 0), không gọi controller.
## Phím R đặt lại vị trí ngẫu nhiên, C đổi góc nhìn camera, Esc thoát.

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
## Trạm thang máy camera FOLLOW đang nhìn vào trong lúc đi thang (null = bám robot như thường).
var _elev_cam_station: ElevatorStation = null
const ELEV_ROBOT_SPEED: float = 0.7      # m/s khi vào / ra thang máy
const ELEV_PATIENT_SPEED: float = 0.75
const ELEV_PATIENT_DELAY_S: float = 0.6  # bệnh nhân bước theo sau robot
var _after_thanks: Callable = Callable()
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
		_policy.sectors_received.connect(overlay.set_visual_sectors)
		if inspection_panel != null:
			_policy.sectors_received.connect(inspection_panel.update_visual_sectors)
		if dialogue != null:
			dialogue.turn_completed.connect(_on_dialogue_turn_completed)
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
	reset_positions()
	_update_viewer_camera()


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
		])

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
				esc_p.set_meta("helped", true)
			active_escorted_patient = null
			var cur_floor: int = Room.get_floor(robot.global_position)
			# Bệnh nhân nói lời cảm ơn bằng clip thật ➔ robot nghe (STT/NLU) và đáp (TTS); xong lượt mới về / reset.
			if has_escort and _start_thanks_turn(esc_p):
				_after_thanks = _finish_escort.bind(cur_floor)
				return
			if dialogue != null and dialogue.has_method("show_subtitle"):
				dialogue.show_subtitle("Bệnh nhân", "Cảm ơn robot, tôi đã tới nơi khám!", 4.0)
			_finish_escort(cur_floor)
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
	if _policy_active():
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


## PPO lái robot ở mọi pha ngoài hội thoại; gửi trạng thái mỗi STEP_DT_S như dt lúc train.
## Pha tìm bệnh nhân: tới ≤ PATIENT_ENGAGE_DIST_M ➔ ARRIVED, dừng chờ NPC bắt đầu hội thoại.
func _step_policy(delta: float) -> void:
	if Main.is_dialogue_state(state):
		return

	# Parse multi-class YOLO detections for patient channel
	var yolo_feat: Array = []
	var best_patient_det: Dictionary = {}
	for d_var: Variant in _latest_yolo_dets:
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

	if _is_search_phase():
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
				robot.set_command(0.0, 0.0)
				return
			state = Config.RobotState.SEARCH  # bệnh nhân đi xa trước khi nói ➔ tìm lại
		elif has_los and (d_human <= PATIENT_ENGAGE_DIST_M or yolo_patient_close):
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
	var tracked: Node = active_escorted_patient if active_escorted_patient != null else (current_target_patient if state == Config.RobotState.APPROACH else null)
	robot.depth_ignore = tracked as CollisionObject3D
	_policy.send_state(
		robot.global_position, robot.rotation.y, robot.get_real_velocity().length(), robot.omega,
		robot.get_lidar_readings(), pat_pos,
		RLEnvBridgeScript.TaskPhase.SEARCH_PATIENT if search else _rl_bridge.task_phase,
		pat_pos if search else _leg_target(_rl_bridge.goal_position),
		_rl_bridge.patients_helped,
		yolo_feat,
		_latest_yolo_dets,
		robot.get_depth_readings()
	)


func _on_policy_action(v: float, omega: float, phase: int) -> void:
	if not _policy_active() or Main.is_dialogue_state(state) or _is_in_elevator_transit:
		return
	if _is_search_phase():
		if state == Config.RobotState.ARRIVED:
			return
		state = Config.RobotState.APPROACH if phase == RLEnvBridgeScript.TaskPhase.APPROACH_PATIENT else Config.RobotState.SEARCH
	else:
		state = Config.RobotState.APPROACH
	robot.set_command(v, omega)


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

	if escort_only_mode and act != "guide":
		act = "guide"
		var fallback_targets = ["110", "112", "nha_thuoc", "201", "204", "205", "210", "220", "301", "302", "303", "315"]
		target_key = fallback_targets[_rng.randi_range(0, fallback_targets.size() - 1)]
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
	if ceiling_mesh != null:
		ceiling_mesh.visible = (camera_mode != CameraMode.TOP_DOWN)
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
			wall.material_override = null

	var trans_mat: StandardMaterial3D = _get_transparent_wall_mat()
	for wall: MeshInstance3D in new_occluding:
		if not _occluding_walls.has(wall):
			wall.material_override = trans_mat

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
			wall.material_override = null
	_occluding_walls.clear()


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


func _setup_hospital_npcs() -> void:
	current_target_patient = pick_next_target_patient()
	if human != null:
		human.setup_robot(robot)
		var h_npc_node: Node = human.get_node_or_null(^"Npc")
		if h_npc_node != null and h_npc_node.has_method("setup"):
			h_npc_node.setup(robot, func() -> int:
				if is_task_busy() or Main.is_dialogue_state(state):
					return Config.RobotState.SEARCH
				if current_target_patient != null and current_target_patient != human:
					return Config.RobotState.SEARCH
				return state
			)
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
					h_npc.setup_robot(robot, func() -> int:
						if is_task_busy() or Main.is_dialogue_state(state):
							return Config.RobotState.SEARCH
						if current_target_patient != null and current_target_patient != pat_body:
							return Config.RobotState.SEARCH
						return state
					)
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
	var now: int = Time.get_ticks_msec()
	_last_result_ms = now
	if not dets.is_empty():
		_last_seen_ms = now
	var cmd: Dictionary
	if Main.is_dialogue_state(state):
		cmd = {"state": state, "v": 0.0, "omega": 0.0, "target": {}}
		robot.set_command(0.0, 0.0)
	elif _rl_bridge != null and _rl_bridge.task_phase in [RLEnvBridgeScript.TaskPhase.GUIDE_ESCORT, RLEnvBridgeScript.TaskPhase.CALL_STAFF]:
		var cur_cmd: Vector2 = robot.get_command() if robot.has_method("get_command") else Vector2(robot.v, robot.omega)
		cmd = {"state": state, "v": cur_cmd.x, "omega": cur_cmd.y, "target": {}}
	elif _policy_active():
		var pol_cmd: Vector2 = robot.get_command()
		cmd = {"state": state, "v": pol_cmd.x, "omega": pol_cmd.y, "target": {}}
	else:
		cmd = _apply(_controller.step(state, dets, _time_since_seen_s(now)))
	overlay.show_detections(dets, _display_state_name(), _connected)
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
