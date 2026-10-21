class_name Main
extends Node3D
## Scene chính (G8): vision_client ➔ controller.step ➔ apply_safety ➔ robot.set_command.
## Phím R đặt lại vị trí ngẫu nhiên, Esc thoát.

## Phát sau mỗi chu kỳ nhận diện (sau apply_safety); G9 ghi log từ đây.
signal cycle_completed(info: Dictionary)

const START_CLEARANCE: float = 0.6      # G8.1: get_free_position(0.6)
const MIN_START_SEPARATION: float = 3.0 # G8.1: robot cách người ≥ 3 m
const MAX_PLACEMENT_TRIES: int = 100

@warning_ignore("shadowed_global_identifier")
@export var seed: int = -1              # −1 = ngẫu nhiên
@export var controller_path: NodePath = ^"Controller"

var state: int = Config.RobotState.SEARCH

var _rng := RandomNumberGenerator.new()
var _controller: Node
var _connected: bool = false
var _last_seen_ms: int = -1             # −1 = chưa thấy người lần nào
var _last_result_ms: int = -1
var _start_ms: int = 0

@onready var room: Room = $Room
@onready var human: Human = $Human
@onready var robot: Robot = $Robot
@onready var vision: VisionClient = $VisionClient
@onready var overlay: ViewOverlay = $UI/ViewOverlay
@onready var logger: RunLogger = get_node_or_null(^"RunLogger")  # G9; không có thì không ghi log


func _ready() -> void:
	_controller = get_node(controller_path)
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
	reset_positions()


func _unhandled_input(event: InputEvent) -> void:
	var key := event as InputEventKey
	if key == null or not key.pressed or key.echo:
		return
	if key.keycode == KEY_R:
		reset_positions()
	elif key.keycode == KEY_ESCAPE:
		get_tree().quit()


func _process(_delta: float) -> void:
	# Quá LOST_TIMEOUT_S không có kết quả mới (nhưng vẫn kết nối) ➔ coi như không thấy ai.
	var now: int = Time.get_ticks_msec()
	if _connected and _last_result_ms >= 0 and now - _last_result_ms > int(Config.LOST_TIMEOUT_S * 1000.0):
		_apply(_controller.step(state, [], _time_since_seen_s(now)))


## Đặt robot và người ở vị trí ngẫu nhiên mới, robot hướng ngẫu nhiên, về SEARCH.
func reset_positions() -> void:
	var p: Dictionary = Main.pick_start_positions(room, _rng)
	robot.global_position = p["robot"]
	robot.rotation.y = p["robot_yaw"]
	human.global_position = p["human"]
	human.rotation.y = p["human_yaw"]
	robot.set_command(0.0, 0.0)
	state = Config.RobotState.SEARCH
	_last_seen_ms = -1
	_last_result_ms = Time.get_ticks_msec()
	overlay.show_detections([], Config.state_name(state), _connected)


func human_distance() -> float:
	return Human.flat_distance(robot.global_position, human.global_position)


## Góc thật tới người so với hướng trước của robot (−Z), dương khi người bên trái.
func human_angle_deg() -> float:
	return RunLogger.signed_angle_deg(-robot.global_transform.basis.z, human.global_position - robot.global_position)


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
	var now: int = Time.get_ticks_msec()
	_last_result_ms = now
	if not dets.is_empty():
		_last_seen_ms = now
	var cmd: Dictionary = _apply(_controller.step(state, dets, _time_since_seen_s(now)))
	overlay.show_detections(dets, Config.state_name(state), _connected)
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
	overlay.show_detections([], Config.state_name(state), connected)


## Áp an toàn rồi gửi lệnh cho robot; trả lệnh cuối cùng.
func _apply(cmd: Dictionary) -> Dictionary:
	var safe: Dictionary = Main.apply_safety(cmd, human_distance(), robot.front_obstacle_distance())
	state = safe["state"]
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


## Vị trí ban đầu: robot và người đều lấy từ room.get_free_position(START_CLEARANCE),
## cách nhau ≥ MIN_START_SEPARATION; hướng robot và người ngẫu nhiên. Cùng rng.seed ➔ cùng kết quả.
static func pick_start_positions(r: Room, rng: RandomNumberGenerator) -> Dictionary:
	var robot_pos := Vector3.ZERO
	var human_pos := Vector3.ZERO
	var ok: bool = false
	for i: int in MAX_PLACEMENT_TRIES:
		robot_pos = r.get_free_position(START_CLEARANCE, rng)
		human_pos = r.get_free_position(START_CLEARANCE, rng)
		if Human.flat_distance(robot_pos, human_pos) >= MIN_START_SEPARATION:
			ok = true
			break
	if not ok:
		push_error("Main.pick_start_positions: no pair ≥ %.1f m apart" % MIN_START_SEPARATION)
	return {"robot": robot_pos, "human": human_pos,
			"robot_yaw": rng.randf_range(-PI, PI), "human_yaw": rng.randf_range(-PI, PI)}
