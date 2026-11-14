class_name RLPolicyClient
extends Node
## Client WebSocket tới rl_policy_server.py (PPO đã train). Gửi trạng thái thô 5 Hz (khớp dt = 0.2 s
## lúc train), nhận lệnh (v, omega) thực tế. Mất kết nối thì thử lại mỗi RECONNECT_S;
## Main quay về controller cũ khi chưa kết nối.

signal action_received(v: float, omega: float, phase: int)
## Policy V3 (docs/RL_V3_DESIGN.md): góc đầu mục tiêu [−1, 1] và có hỏi "Cần tôi giúp gì không ạ?" không.
signal engage_received(tilt: float, ask: bool)
## Vòng 11: policy tự quay lại hỏi bệnh nhân dừng lại ➔ EscortMonitor không tự hỏi nữa.
var policy_returns_to_patient: bool = false
signal sectors_received(sectors: Array)
signal connection_changed(connected: bool)

const URL: String = "ws://127.0.0.1:8768"
const STEP_DT_S: float = 0.2

var _ws: WebSocketPeer = null
var _connected: bool = false
var _last_attempt_ms: int = -1
var _next_id: int = 0
var _waiting: bool = false


func is_server_connected() -> bool:
	return _connected


func _process(_delta: float) -> void:
	var now: int = Time.get_ticks_msec()
	var state: int = WebSocketPeer.STATE_CLOSED
	if _ws != null:
		_ws.poll()
		state = _ws.get_ready_state()
	_set_connected(state == WebSocketPeer.STATE_OPEN)
	if state == WebSocketPeer.STATE_CLOSED \
			and (_last_attempt_ms < 0 or now - _last_attempt_ms >= int(Config.RECONNECT_S * 1000.0)):
		_last_attempt_ms = now
		_ws = WebSocketPeer.new()
		_ws.connect_to_url(SimClock.env_url if SimClock.env_url != "" else URL)
	_poll()


func _ready() -> void:
	SimClock.register_poller(_poll)


func _poll() -> void:
	if _ws == null:
		return
	_ws.poll()
	if _ws.get_ready_state() != WebSocketPeer.STATE_OPEN:
		SimClock.release("policy", true)
	while _ws != null and _ws.get_available_packet_count() > 0:
		SimClock.release("policy")
		_handle_message(_ws.get_packet().get_string_from_utf8())


func _set_connected(value: bool) -> void:
	if value != _connected:
		_connected = value
		_waiting = false
		connection_changed.emit(value)


## Gửi một bước quan sát; bỏ qua nếu bước trước chưa có trả lời (không dồn hàng đợi).
func send_state(robot_pos: Vector3, robot_yaw: float, v: float, omega: float, lidar: PackedFloat32Array,
		patient_pos: Vector3, phase: int, goal_pos: Vector3, task_id: int, yolo: Array = [], dets: Array = [],
		depth: PackedFloat32Array = PackedFloat32Array(), v3: Dictionary = {}, train: Dictionary = {}) -> void:
	if not _connected or _waiting:
		return
	var msg: Dictionary = {
		"type": "rl_state", "id": _next_id,
		"robot": [robot_pos.x, robot_pos.z, robot_yaw, v, omega],
		"lidar": Array(lidar),
		"patient": [patient_pos.x, patient_pos.z],
		"phase": phase, "goal": [goal_pos.x, goal_pos.z], "task_id": task_id,
	}
	if not yolo.is_empty():
		msg["yolo"] = yolo
	if not dets.is_empty():
		msg["dets"] = dets
	if not depth.is_empty():
		msg["depth"] = Array(depth)  # V2: 15 khoảng cách depth (m) trong HFOV camera
	if not v3.is_empty():
		msg["v3"] = v3               # V3: tiếng nói / mặt / góc đầu / hỏi / đang nói chuyện
	if not train.is_empty():
		msg["train"] = train         # train trong Godot: sự kiện + trạng thái thật để tính phần thưởng (policy không thấy)
	if _ws.send_text(JSON.stringify(msg)) == OK:
		_waiting = true
		SimClock.hold("policy")
		_next_id += 1


func _handle_message(text: String) -> void:
	_waiting = false
	var data: Variant = JSON.parse_string(text)
	if not (data is Dictionary) or str(data.get("type", "")) != "rl_action":
		push_warning("RLPolicyClient: bad reply %s" % text.left(200))
		return
	if data.has("sectors") and data["sectors"] is Array:
		sectors_received.emit(data["sectors"])
	if data.has("tilt"):
		policy_returns_to_patient = bool(data.get("returns", false))
		engage_received.emit(float(data["tilt"]), bool(data.get("ask", false)))
	action_received.emit(float(data["v"]), float(data["omega"]), int(data.get("phase", 0)))
