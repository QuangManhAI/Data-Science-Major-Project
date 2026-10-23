class_name RLController
extends Controller
## Bộ điều khiển RL (kế thừa Controller để giữ nguyên giao diện `step` tương thích với Main).
## Hỗ trợ:
## 1) Nhận lệnh trực tiếp từ mạng chính sách PPO qua `set_rl_action(a_v, a_omega)`
## 2) Hoạt động độc lập với bộ điều hướng pha nhiệm vụ (`SEARCH`, `APPROACH`, `GUIDE_ESCORT`, `CALL_STAFF`)
##    kết hợp tránh vật cản bằng LiDAR 17 tia khi chạy suy luận nội bộ.

const RLEnvBridgeRef = preload("res://scripts/rl_env_bridge.gd")

var has_external_action: bool = false
var external_v: float = 0.0
var external_omega: float = 0.0
var external_state: int = Config.RobotState.SEARCH


## Nạp hành động đã chuẩn hóa [-1, 1]^2 từ mô hình PPO (a_v, a_omega) sang (v, omega) thực tế.
static func decode_action(a_v: float, a_omega: float) -> Vector2:
	var v: float = clampf((a_v + 1.0) * 0.5 * Config.V_MAX, 0.0, Config.V_MAX)
	var omega: float = clampf(a_omega * Config.OMEGA_MAX, -Config.OMEGA_MAX, Config.OMEGA_MAX)
	return Vector2(v, omega)


func set_rl_action(a_v: float, a_omega: float, next_state: int = Config.RobotState.APPROACH) -> void:
	var cmd: Vector2 = RLController.decode_action(a_v, a_omega)
	external_v = cmd.x
	external_omega = cmd.y
	external_state = next_state
	has_external_action = true


func clear_rl_action() -> void:
	has_external_action = false


## Tính lệnh điều hướng tới điểm đích `goal_pos` kết hợp phản xạ tránh vật cản từ 17 tia LiDAR
## (dùng cho dẫn đường GUIDE_ESCORT / CALL_STAFF hoặc khi chạy chính sách lai).
static func step_towards_goal_with_lidar(
	robot_pos: Vector3,
	robot_yaw: float,
	goal_pos: Vector3,
	lidar: PackedFloat32Array,
	d_tether: float = 0.0,
	is_escorting: bool = false
) -> Vector2:
	var dist: float = Human.flat_distance(robot_pos, goal_pos)
	if dist <= RLEnvBridge.GOAL_ARRIVE_ROBOT_M:
		return Vector2.ZERO
	if is_escorting and d_tether > RLEnvBridge.TETHER_MAX_M:
		# Đợi bệnh nhân cao tuổi theo kịp
		return Vector2.ZERO

	var fwd := Vector3(-sin(robot_yaw), 0.0, -cos(robot_yaw))
	var to_goal := Vector3(goal_pos.x - robot_pos.x, 0.0, goal_pos.z - robot_pos.z)
	var angle_rad: float = deg_to_rad(RunLogger.signed_angle_deg(fwd, to_goal))

	# Đẩy góc lái tránh tường/vật cản từ nửa trái và nửa phải của LiDAR 17 tia
	var repel_turn: float = 0.0
	if lidar.size() >= Robot.LIDAR_NUM_RAYS:
		var mid: int = Robot.LIDAR_NUM_RAYS / 2
		var left_clear: float = 0.0
		var right_clear: float = 0.0
		for i: int in range(0, mid):
			left_clear += lidar[i]
		for i: int in range(mid + 1, Robot.LIDAR_NUM_RAYS):
			right_clear += lidar[i]
		var center_min: float = minf(lidar[mid], minf(lidar[mid - 1], lidar[mid + 1]))
		if center_min < 0.35:
			repel_turn = 0.9 if left_clear >= right_clear else -0.9

	var omega: float = clampf(1.6 * angle_rad + repel_turn, -Config.OMEGA_MAX, Config.OMEGA_MAX)
	var max_v: float = RLEnvBridge.ESCORT_WALK_SPEED if is_escorting else Config.V_MAX
	var v: float = clampf(0.45 * dist, 0.0, max_v)
	if absf(angle_rad) > 0.65:
		v *= 0.25
	return Vector2(v, omega)


func step(state: int, dets: Array, time_since_seen_s: float) -> Dictionary:
	if has_external_action:
		var target_det: Dictionary = Controller.pick_target(dets) if not dets.is_empty() else {}
		return _cmd(external_state, external_v, external_omega, target_det)
	return super.step(state, dets, time_since_seen_s)
