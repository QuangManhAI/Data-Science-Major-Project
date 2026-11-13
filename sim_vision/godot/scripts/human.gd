class_name Human
extends CharacterBody3D
## Người tả thực (Rocketbox Male_Adult_06 hoặc biến thể cao tuổi). Mặc định đứng yên (idle);
## `walks = true` ➔ đi chậm qua lần lượt các `waypoints` (lặp vòng).
## Hỗ trợ `pardon_enabled`: tự động bước sang một bên nhường đường khi robot đi ngang qua.
## Mặt người hướng về +Z cục bộ.

const WALK_SPEED: float = 0.5     # m/s (G4.1)
const ARRIVE_TOL: float = 0.05    # m, coi như đã tới waypoint
const PARDON_TRIGGER_DIST: float = 1.4
const PARDON_CLEAR_DIST: float = 1.85
## Locomotion: walk.res đã gỡ root motion (tools/extract_human_anims.gd); clip gốc tiến 1.40 m/s.
const WALK_ANIM_SPEED: float = 1.40       # m/s ➔ speed_scale = v_thật / 1.40 để chân không trượt
const WALK_ANIM_SCALE_MIN: float = 0.3
const WALK_ANIM_SCALE_MAX: float = 1.6
const WALK_START_SPEED: float = 0.12      # m/s (đã lọc) ➔ chuyển sang walk
const WALK_STOP_SPEED: float = 0.06       # m/s ➔ về idle (trễ/hysteresis, tránh nháy walk↔idle)
const SPEED_SMOOTH_RATE: float = 10.0     # 1/s, lọc thông thấp vận tốc đo được
const TURN_RATE: float = 9.0              # 1/s, xoay thân mượt thay vì quay ngoắt
const TELEPORT_DIST: float = 0.5          # m/khung: dịch chuyển tức thời ➔ không tính là đi

@export var walks: bool = false
@export var waypoints: Array[Vector3] = []
@export var pardon_enabled: bool = false
@export var use_t_pose: bool = false

var robot_ref: Node3D = null

var _wp_index: int = 0
var _is_pardoning: bool = false
var _make_way_left_s: float = 0.0
var _pardon_target: Vector3 = Vector3.ZERO
var _anchor_pos: Vector3 = Vector3.ZERO
var _escorted: bool = false  # đang được robot dẫn đường: main điều khiển bước đi, bỏ qua waypoint / pardon
var _last_pos: Vector3 = Vector3.ZERO
var _speed: float = 0.0       # vận tốc phẳng đã lọc (m/s)
var _moving: bool = false

@onready var _anim: AnimationPlayer = $AnimationPlayer


func _ready() -> void:
	_anchor_pos = global_position
	_last_pos = global_position
	_moving = _is_walking()
	_speed = WALK_SPEED if _moving else 0.0
	if not use_t_pose:
		_anim.play(&"human/walk" if _moving else &"human/idle")
	elif _anim != null and _anim.is_playing():
		_anim.stop()


func setup_robot(robot_node: Node3D) -> void:
	robot_ref = robot_node


func is_pardoning() -> bool:
	return _is_pardoning


## Robot xin đường (xem HospitalNpc.make_way).
func make_way(robot: Node3D) -> bool:
	if robot == null or _escorted:
		return false
	var target: Variant = HospitalNpc.pick_clear_pardon_target(self, global_position, robot)
	if target == null:
		return false
	_anchor_pos = global_position
	_is_pardoning = true
	_pardon_target = target
	_make_way_left_s = HospitalNpc.MAKE_WAY_S
	return true


## Thay bộ texture (body/head/opacity) của `$Human` sang biến thể bệnh nhân cao tuổi bất kỳ.
func apply_variant_textures(tex_dir: String) -> void:
	var model := get_node_or_null(^"Model")
	if model == null:
		return
	var mesh_node := HospitalNpc._find_mesh(model)
	if mesh_node == null or mesh_node.mesh == null:
		return
	for s: int in mesh_node.mesh.get_surface_count():
		var orig_mat: Material = mesh_node.mesh.surface_get_material(s)
		var mname: String = orig_mat.resource_name.to_lower() if orig_mat != null else ""
		var mat := StandardMaterial3D.new()
		if "opacity" in mname or s == 2:
			var o_path: String = "%s/opacity_color.png" % tex_dir
			if ResourceLoader.exists(o_path):
				mat.albedo_texture = load(o_path)
			mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_HASH
			mat.cull_mode = BaseMaterial3D.CULL_DISABLED
			mat.roughness = 0.7
		else:
			var part: String = "head" if ("head" in mname or s == 1) else "body"
			var c_path: String = "%s/%s_color.jpg" % [tex_dir, part]
			var n_path: String = "%s/%s_normal.png" % [tex_dir, part]
			var r_path: String = "%s/%s_roughness.jpg" % [tex_dir, part]
			if ResourceLoader.exists(c_path):
				mat.albedo_texture = load(c_path)
			if ResourceLoader.exists(r_path):
				mat.roughness_texture = load(r_path)
			if ResourceLoader.exists(n_path):
				mat.normal_enabled = true
				mat.normal_texture = load(n_path)
		mesh_node.set_surface_override_material(s, mat)


## Bước theo robot khi được dẫn đường (gọi mỗi physics frame từ RLEnvBridge.step_escorted_patient).
func escort_step_towards(target: Vector3, max_step: float, delta: float) -> void:
	_escorted = true
	var to := Vector3(target.x - global_position.x, 0.0, target.z - global_position.z)
	var d: float = to.length()
	if d < 1e-4 or max_step <= 0.0:
		escort_stop()
		return
	var disp: Vector3 = to / d * minf(max_step, d)
	face_yaw(atan2(disp.x, disp.z), delta)
	velocity = disp / maxf(delta, 1e-4)
	move_and_slide()
	velocity = Vector3.ZERO


func escort_stop() -> void:
	_escorted = true
	velocity = Vector3.ZERO


## Kết thúc dẫn đường (reset / hoàn thành nhiệm vụ) ➔ hành vi thường.
func end_escort() -> void:
	_escorted = false
	_anchor_pos = global_position


func _physics_process(delta: float) -> void:
	_update_locomotion(delta)
	# Đang nói chuyện với robot (kể cả lúc được dẫn / đang ngồi): nhìn về robot để robot thấy mặt.
	var npc: Node = get_node_or_null(^"Npc")
	if npc != null and ("state" in npc) and npc.state in [Npc.NOTICE, Npc.TALK, Npc.WAIT_REPLY] \
			and robot_ref != null and is_instance_valid(robot_ref) and not npc.is_turning():
		velocity = Vector3.ZERO
		face_yaw(atan2(robot_ref.global_position.x - global_position.x,
				robot_ref.global_position.z - global_position.z), delta)
		return
	if _escorted:
		return
	if pardon_enabled and robot_ref != null and is_instance_valid(robot_ref):
		var dist_robot: float = Human.flat_distance(global_position, robot_ref.global_position)
		if not _is_pardoning and dist_robot <= PARDON_TRIGGER_DIST \
				and HospitalNpc.should_pardon(global_position, robot_ref):
			_anchor_pos = global_position
			var target: Variant = HospitalNpc.pick_clear_pardon_target(self, _anchor_pos, robot_ref)
			if target != null:
				_is_pardoning = true
				_pardon_target = target
		elif _is_pardoning and _make_way_left_s <= 0.0 and dist_robot >= PARDON_CLEAR_DIST:
			_is_pardoning = false

	if _make_way_left_s > 0.0:
		_make_way_left_s -= delta
		if _make_way_left_s <= 0.0:
			_is_pardoning = false
	if _is_pardoning:
		if Human.flat_distance(global_position, _pardon_target) > ARRIVE_TOL:
			var p_step: Vector3 = Human.step_towards(global_position, _pardon_target, WALK_SPEED * delta)
			var p_dir := Vector3(p_step.x - global_position.x, 0.0, p_step.z - global_position.z)
			if p_dir.length() > 0.0:
				face_yaw(atan2(p_dir.x, p_dir.z), delta)
				velocity = p_dir / delta
				move_and_slide()
		else:
			velocity = Vector3.ZERO
			if robot_ref != null and is_instance_valid(robot_ref):
				face_yaw(atan2(
					robot_ref.global_position.x - global_position.x,
					robot_ref.global_position.z - global_position.z
				), delta)
		return

	if not _is_walking():
		return
	var target: Vector3 = waypoints[_wp_index]
	if Human.flat_distance(global_position, target) <= ARRIVE_TOL:
		_wp_index = Human.next_waypoint_index(_wp_index, waypoints.size())
		target = waypoints[_wp_index]
	var step: Vector3 = Human.step_towards(global_position, target, WALK_SPEED * delta)
	var dir := Vector3(step.x - global_position.x, 0.0, step.z - global_position.z)
	if dir.length() > 0.0:
		face_yaw(atan2(dir.x, dir.z), delta)  # quay mặt (+Z) dần theo hướng đi
		velocity = dir / delta
		move_and_slide()


func _is_walking() -> bool:
	return walks and waypoints.size() >= 2


## Xoay thân mượt về `yaw` (không quay ngoắt 180° ở đầu vòng tuần).
func face_yaw(yaw: float, delta: float) -> void:
	rotation.y = lerp_angle(rotation.y, yaw, Human.smooth_factor(TURN_RATE, delta))


## Chọn walk/idle theo vận tốc THẬT (đo từ dịch chuyển mỗi khung, đã lọc + trễ) và chỉnh nhịp
## bước `speed_scale` = v / WALK_ANIM_SPEED ➔ bàn chân khớp mặt sàn, không trượt, không nháy.
func _update_locomotion(delta: float) -> void:
	var moved: float = Human.flat_distance(global_position, _last_pos)
	_last_pos = global_position
	if delta <= 0.0:
		return
	var inst_speed: float = 0.0 if moved > TELEPORT_DIST else moved / delta
	_speed = lerpf(_speed, inst_speed, Human.smooth_factor(SPEED_SMOOTH_RATE, delta))
	_moving = Human.next_moving(_moving, _speed)
	_play(&"human/walk" if _moving else &"human/idle")
	if _anim != null:
		_anim.speed_scale = Human.walk_anim_scale(_speed) if _moving else 1.0


func _play(anim_name: StringName) -> void:
	if use_t_pose:
		if _anim != null and _anim.is_playing():
			_anim.stop()
		return
	if _anim != null and _anim.current_animation != anim_name:
		_anim.play(anim_name, 0.3)


static func smooth_factor(rate: float, delta: float) -> float:
	return 1.0 - exp(-rate * maxf(delta, 0.0))


## Trễ (hysteresis) walk↔idle: bắt đầu đi khi > WALK_START_SPEED, dừng khi < WALK_STOP_SPEED.
static func next_moving(moving: bool, speed: float) -> bool:
	return speed > WALK_STOP_SPEED if moving else speed > WALK_START_SPEED


static func walk_anim_scale(speed: float) -> float:
	return clampf(speed / WALK_ANIM_SPEED, WALK_ANIM_SCALE_MIN, WALK_ANIM_SCALE_MAX)


static func flat_distance(a: Vector3, b: Vector3) -> float:
	return Vector2(a.x - b.x, a.z - b.z).length()


static func next_waypoint_index(i: int, count: int) -> int:
	return (i + 1) % count if count > 0 else 0


## Điểm mới sau khi tiến tối đa `max_dist` về `target` trên mặt phẳng XZ (giữ nguyên y).
static func step_towards(from: Vector3, target: Vector3, max_dist: float) -> Vector3:
	var d: float = flat_distance(from, target)
	if d <= max_dist or d == 0.0:
		return Vector3(target.x, from.y, target.z)
	var t: float = max_dist / d
	return Vector3(from.x + (target.x - from.x) * t, from.y, from.z + (target.z - from.z) * t)
