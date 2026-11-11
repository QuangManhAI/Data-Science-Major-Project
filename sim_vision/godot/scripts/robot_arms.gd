class_name RobotArms
extends Node3D
## Hai tay robot để giao tiếp (HRI, docs/UPGRADE_HRI_PLAN.md mục A): chỉ vào phòng / thang máy / hướng đi, vẫy chào.
## Mỗi tay: khớp vai (Node3D) ➔ cánh tay trên ➔ khuỷu ➔ cẳng tay ➔ bàn tay; trục tay là −Z của khớp vai.
## Tay chỉ để nhìn: không có collision (không chạm tia depth / LiDAR) và nằm ở VISUAL_LAYER, camera robot không
## render layer này ➔ tay không lọt vào ảnh gửi YOLO.

const VISUAL_LAYER_BIT: int = 9                 # layer 10 trong editor
const VISUAL_LAYER: int = 1 << VISUAL_LAYER_BIT
const SHOULDER_X: float = 0.2
const SHOULDER_Y: float = 0.86
const UPPER_LEN: float = 0.24
const FORE_LEN: float = 0.22
const ARM_RADIUS: float = 0.032
const MAX_YAW_RAD: float = deg_to_rad(110.0)    # không chỉ ra sau lưng
const PITCH_MIN_RAD: float = deg_to_rad(-30.0)
const PITCH_MAX_RAD: float = deg_to_rad(-4.0)   # luôn hơi chúc xuống: tay không giơ lên trước camera
const REST_ELBOW_RAD: float = deg_to_rad(28.0)
const TURN_RATE: float = 7.0                    # 1/s, làm mượt khớp
const WAVE_HZ: float = 1.6
const WAVE_AMP_RAD: float = deg_to_rad(22.0)

enum Side { LEFT = -1, NONE = 0, RIGHT = 1 }

var _shoulders: Dictionary = {}   # Side ➔ Node3D
var _elbows: Dictionary = {}      # Side ➔ Node3D
var _point_side: int = Side.NONE
var _point_world: Vector3 = Vector3.ZERO
var _point_left_s: float = 0.0    # > 0: tự hạ tay khi hết; 0: giữ đến lower()
var _wave_left_s: float = 0.0
var _wave_t: float = 0.0


func _ready() -> void:
	name = "Arms"
	if _shoulders.is_empty():
		_build()


func _build() -> void:
	var shell := StandardMaterial3D.new()
	shell.albedo_color = Color(0.95, 0.96, 0.97)
	shell.roughness = 0.25
	var joint := StandardMaterial3D.new()
	joint.albedo_color = Color(0.13, 0.14, 0.16)
	joint.metallic = 0.35
	joint.roughness = 0.45
	var accent := StandardMaterial3D.new()
	accent.albedo_color = Color(0.0, 0.62, 0.66)
	for side: int in [Side.LEFT, Side.RIGHT]:
		var sh := Node3D.new()
		sh.name = "ShoulderL" if side == Side.LEFT else "ShoulderR"
		sh.position = Vector3(SHOULDER_X * side, SHOULDER_Y, 0.0)
		add_child(sh)
		_add_mesh(sh, _sphere(ARM_RADIUS * 1.5, joint), Vector3.ZERO)
		_add_mesh(sh, _capsule(ARM_RADIUS, UPPER_LEN, shell), Vector3(0, 0, -UPPER_LEN * 0.5), true)
		var el := Node3D.new()
		el.name = "Elbow"
		el.position = Vector3(0, 0, -UPPER_LEN)
		sh.add_child(el)
		_add_mesh(el, _sphere(ARM_RADIUS * 1.15, joint), Vector3.ZERO)
		_add_mesh(el, _capsule(ARM_RADIUS * 0.9, FORE_LEN, shell), Vector3(0, 0, -FORE_LEN * 0.5), true)
		_add_mesh(el, _sphere(ARM_RADIUS * 1.35, accent), Vector3(0, 0, -FORE_LEN))
		_shoulders[side] = sh
		_elbows[side] = el
		sh.quaternion = rest_quat(side)
		el.quaternion = Quaternion(Vector3.RIGHT, REST_ELBOW_RAD)


func _add_mesh(parent: Node3D, mesh: Mesh, pos: Vector3, along_z: bool = false) -> void:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.position = pos
	if along_z:
		mi.rotation = Vector3(PI * 0.5, 0, 0)   # capsule dọc Y ➔ nằm dọc trục Z của khớp
	mi.layers = VISUAL_LAYER
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	parent.add_child(mi)


static func _sphere(r: float, mat: Material) -> SphereMesh:
	var m := SphereMesh.new()
	m.radius = r
	m.height = r * 2.0
	m.material = mat
	return m


static func _capsule(r: float, length: float, mat: Material) -> CapsuleMesh:
	var m := CapsuleMesh.new()
	m.radius = r
	m.height = length + r * 2.0
	m.material = mat
	return m


## Tư thế nghỉ: buông dọc thân, hơi chĩa ra ngoài và ra trước.
static func rest_quat(side: int) -> Quaternion:
	return Basis.looking_at(Vector3(0.18 * side, -1.0, -0.12).normalized(), Vector3(0, 0, -1)).get_rotation_quaternion()


## Tay dùng để chỉ: đích bên phải (x > 0, robot nhìn về −Z) ➔ tay phải.
static func choose_side(local_dir: Vector3) -> int:
	return Side.RIGHT if local_dir.x >= 0.0 else Side.LEFT


## Hướng chỉ (hệ robot) sau khi kẹp góc: yaw trong ±MAX_YAW_RAD, pitch trong [PITCH_MIN, PITCH_MAX].
static func clamp_point_dir(local_dir: Vector3) -> Vector3:
	var horiz: float = Vector2(local_dir.x, local_dir.z).length()
	var yaw: float = atan2(local_dir.x, -local_dir.z) if horiz > 1e-6 else 0.0
	yaw = clampf(yaw, -MAX_YAW_RAD, MAX_YAW_RAD)
	var pitch: float = clampf(atan2(local_dir.y, maxf(horiz, 1e-6)), PITCH_MIN_RAD, PITCH_MAX_RAD)
	return Vector3(sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch))


static func point_quat(local_dir: Vector3) -> Quaternion:
	return Basis.looking_at(clamp_point_dir(local_dir), Vector3.UP).get_rotation_quaternion()


## Chỉ vào một điểm trong thế giới (theo dõi điểm đó khi robot đang đi). duration_s = 0 ➔ giữ đến lower().
func point_at(world_pos: Vector3, duration_s: float = 0.0) -> void:
	_point_world = world_pos
	_point_side = choose_side(_local_dir_from_shoulder(world_pos, Side.RIGHT))
	_point_left_s = duration_s
	_wave_left_s = 0.0


func lower() -> void:
	_point_side = Side.NONE
	_point_left_s = 0.0
	_wave_left_s = 0.0


func wave(duration_s: float = 1.6) -> void:
	_point_side = Side.NONE
	_wave_left_s = duration_s
	_wave_t = 0.0


func pointing_side() -> int:
	return _point_side


func is_waving() -> bool:
	return _wave_left_s > 0.0


func _local_dir_from_shoulder(world_pos: Vector3, side: int) -> Vector3:
	var robot_local: Vector3 = to_local(world_pos)
	return robot_local - Vector3(SHOULDER_X * side, SHOULDER_Y, 0.0)


func _process(delta: float) -> void:
	if _shoulders.is_empty():
		return
	if _point_left_s > 0.0:
		_point_left_s -= delta
		if _point_left_s <= 0.0:
			_point_side = Side.NONE
	if _wave_left_s > 0.0:
		_wave_left_s -= delta
		_wave_t += delta
	var k: float = 1.0 - exp(-TURN_RATE * delta)
	for side: int in [Side.LEFT, Side.RIGHT]:
		var sh: Node3D = _shoulders[side]
		var el: Node3D = _elbows[side]
		var target_sh: Quaternion = rest_quat(side)
		var target_el: Quaternion = Quaternion(Vector3.RIGHT, REST_ELBOW_RAD)
		if side == _point_side:
			target_sh = point_quat(_local_dir_from_shoulder(_point_world, side))
			target_el = Quaternion.IDENTITY
		elif side == Side.RIGHT and _wave_left_s > 0.0:
			# Vẫy: cánh tay giơ ngang sang bên, cẳng tay dựng lên và lắc qua lại.
			target_sh = Basis.looking_at(Vector3(1.0, 0.25, -0.35).normalized(), Vector3.UP).get_rotation_quaternion()
			target_el = Quaternion(Vector3.RIGHT, deg_to_rad(80.0)) \
					* Quaternion(Vector3.UP, WAVE_AMP_RAD * sin(TAU * WAVE_HZ * _wave_t))
		sh.quaternion = sh.quaternion.slerp(target_sh, k)
		el.quaternion = el.quaternion.slerp(target_el, k)
