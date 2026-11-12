class_name Robot
extends CharacterBody3D
## Robot differential drive (G5). Hướng trước là −Z cục bộ.
## omega > 0 ➔ quay trái (ngược chiều kim đồng hồ nhìn từ trên xuống), đúng Mục 2.4.
## Trang bị cảm biến: Camera RGB trước, FrontRay (0.8m), và LiDAR 2D 17 tia (200°, tầm xa 6.0 m) cho RL.

const LIDAR_NUM_RAYS: int = 17
const LIDAR_FOV_DEG: float = 200.0
const LIDAR_MAX_RANGE: float = 6.0
const LIDAR_HEIGHT_Y: float = 0.35

var _v: float = 0.0
var _omega: float = 0.0
var _lidar_rays: Array[RayCast3D] = []
var _depth_rays: Array[RayCast3D] = []
## Người đang được theo dõi (bệnh nhân mục tiêu / đang dẫn): pixel depth trong box YOLO của họ bị che,
## vì họ là mục tiêu chứ không phải vật cản (khớp HospitalPureVisionV2Env.depth_rays).
var depth_ignore: CollisionObject3D = null
## Tay giao tiếp (chỉ hướng / vẫy), xem robot_arms.gd.
var arms: RobotArms = null

## Đầu ngẩng / cúi (RL V3, docs/RL_V3_DESIGN.md): camera trước ngẩng theo đầu. Khớp HospitalEngageV3Env:
## −15° (cúi) .. +30° (ngẩng), tối đa 60°/s. head_tilt > 0 là ngẩng lên.
const HEAD_TILT_DOWN: float = deg_to_rad(15.0)
const HEAD_TILT_UP: float = deg_to_rad(30.0)
const HEAD_TILT_RATE: float = deg_to_rad(60.0)
var head_tilt: float = 0.0
var head_tilt_target: float = 0.0

# Camera depth (RGB-D / depth ước lượng trên robot thật): 15 tia ngang trong HFOV camera, tầm 4 m.
# Khớp DEPTH_* trong src/training/hospital_env_pure_vision_v2.py
const DEPTH_NUM_RAYS: int = 15
const DEPTH_HFOV_DEG: float = 86.4
const DEPTH_MAX_RANGE: float = 4.0

var v: float:
	get:
		return _v
	set(val):
		set_command(val, _omega)

var omega: float:
	get:
		return _omega
	set(val):
		set_command(_v, val)


@onready var _viewport: SubViewport = $CameraViewport
## Camera sau (docs/UPGRADE_HRI_PLAN.md D): chỉ để thấy bệnh nhân đi phía sau / tụt lại khi dẫn đường.
## Không đưa vào policy RL (policy train với camera trước). Tắt render khi không dẫn đường.
@onready var _rear_viewport: SubViewport = $RearViewport
@onready var _front_ray: RayCast3D = $FrontRay


func _ready() -> void:
	_viewport.size = Vector2i(Config.IMG_W, Config.IMG_H)
	_rear_viewport.size = Vector2i(Config.IMG_W, Config.IMG_H)
	_setup_lidar_fan()
	_setup_depth_fan()
	_setup_arms()


func _setup_arms() -> void:
	if arms != null:
		return
	arms = RobotArms.new()
	add_child(arms)
	# Camera robot (ảnh cho YOLO) không render tay của chính nó.
	for c: Node in [_viewport.get_node_or_null(^"RobotCamera"), _rear_viewport.get_node_or_null(^"RearCamera")]:
		if c is Camera3D:
			(c as Camera3D).cull_mask &= ~RobotArms.VISUAL_LAYER
			RenderLayers.setup_robot_camera(c as Camera3D)


func _setup_lidar_fan() -> void:
	if not _lidar_rays.is_empty():
		return
	var holder := Node3D.new()
	holder.name = "LidarFan"
	holder.position = Vector3(0.0, LIDAR_HEIGHT_Y, 0.0)
	add_child(holder)
	for i: int in LIDAR_NUM_RAYS:
		var frac: float = float(i) / float(LIDAR_NUM_RAYS - 1) - 0.5  # [-0.5, +0.5]
		var angle_rad: float = deg_to_rad(frac * LIDAR_FOV_DEG)
		var ray := RayCast3D.new()
		ray.name = "LidarRay_%02d" % i
		# Hướng trước cục bộ là -Z; góc dương quay sang phải, góc âm sang trái (quét từ trái sang phải)
		ray.target_position = Vector3(sin(angle_rad) * LIDAR_MAX_RANGE, 0.0, -cos(angle_rad) * LIDAR_MAX_RANGE)
		ray.enabled = true
		ray.add_exception(self)
		holder.add_child(ray)
		_lidar_rays.append(ray)


func _setup_depth_fan() -> void:
	if not _depth_rays.is_empty():
		return
	var holder := Node3D.new()
	holder.name = "DepthFan"
	holder.position = Vector3(0.0, LIDAR_HEIGHT_Y, 0.0)
	add_child(holder)
	for i: int in DEPTH_NUM_RAYS:
		var frac: float = float(i) / float(DEPTH_NUM_RAYS - 1) - 0.5  # trái -> phải
		var angle_rad: float = deg_to_rad(frac * DEPTH_HFOV_DEG)
		var ray := RayCast3D.new()
		ray.name = "DepthRay_%02d" % i
		ray.target_position = Vector3(sin(angle_rad) * DEPTH_MAX_RANGE, 0.0, -cos(angle_rad) * DEPTH_MAX_RANGE)
		ray.enabled = true
		ray.add_exception(self)
		holder.add_child(ray)
		_depth_rays.append(ray)


## 15 khoảng cách depth (m, trái -> phải, tối đa 4.0 m), bỏ qua `depth_ignore`.
func get_depth_readings() -> PackedFloat32Array:
	if _depth_rays.is_empty():
		_setup_depth_fan()
	var out := PackedFloat32Array()
	out.resize(DEPTH_NUM_RAYS)
	for i: int in DEPTH_NUM_RAYS:
		var ray: RayCast3D = _depth_rays[i]
		ray.clear_exceptions()
		ray.add_exception(self)
		if depth_ignore != null and is_instance_valid(depth_ignore):
			ray.add_exception(depth_ignore)
		ray.force_raycast_update()
		out[i] = ray.global_position.distance_to(ray.get_collision_point()) if ray.is_colliding() else DEPTH_MAX_RANGE
	return out


## Trả mảng 17 giá trị khoảng cách LiDAR đã chuẩn hóa về [0.0, 1.0] (1.0 = không chạm trong tầm 6.0m).
func get_lidar_readings() -> PackedFloat32Array:
	if _lidar_rays.is_empty():
		_setup_lidar_fan()
	var out := PackedFloat32Array()
	out.resize(LIDAR_NUM_RAYS)
	for i: int in LIDAR_NUM_RAYS:
		var ray: RayCast3D = _lidar_rays[i]
		ray.force_raycast_update()
		if not ray.is_colliding():
			out[i] = 1.0
		else:
			var dist: float = ray.global_position.distance_to(ray.get_collision_point())
			out[i] = clampf(dist / LIDAR_MAX_RANGE, 0.0, 1.0)
	return out


## Khoảng cách nhỏ nhất (m) từ mảng LiDAR 17 tia.
func min_lidar_distance() -> float:
	var readings: PackedFloat32Array = get_lidar_readings()
	var min_norm: float = 1.0
	for v: float in readings:
		min_norm = minf(min_norm, v)
	return min_norm * LIDAR_MAX_RANGE


## Nhận lệnh vận tốc; v kẹp về [0, V_MAX], omega kẹp về [-OMEGA_MAX, OMEGA_MAX].
func set_command(v: float, omega: float) -> void:
	var c: Vector2 = Robot.clamp_command(v, omega)
	_v = c.x
	_omega = c.y


func get_command() -> Vector2:
	return Vector2(_v, _omega)


func get_camera_viewport() -> SubViewport:
	return get_node(^"CameraViewport") as SubViewport  # dùng được cả trước _ready


func get_rear_viewport() -> SubViewport:
	return get_node(^"RearViewport") as SubViewport


func get_rear_camera() -> Camera3D:
	return get_node(^"RearViewport/RearCamera") as Camera3D


func set_rear_camera_enabled(on: bool) -> void:
	get_rear_viewport().render_target_update_mode = SubViewport.UPDATE_ALWAYS if on else SubViewport.UPDATE_DISABLED


func is_rear_camera_enabled() -> bool:
	return get_rear_viewport().render_target_update_mode == SubViewport.UPDATE_ALWAYS


## Khoảng cách từ đầu raycast phía trước tới vật cản; không chạm ➔ INF.
func front_obstacle_distance() -> float:
	_front_ray.force_raycast_update()
	if not _front_ray.is_colliding():
		return INF
	return _front_ray.global_position.distance_to(_front_ray.get_collision_point())


## Mục tiêu góc đầu từ hành động chuẩn hoá a ∈ [−1, 1] (a > 0 ngẩng tới 30°, a < 0 cúi tới 15°).
func set_head_tilt_action(a: float) -> void:
	a = clampf(a, -1.0, 1.0)
	head_tilt_target = a * (HEAD_TILT_UP if a > 0.0 else HEAD_TILT_DOWN)


## Góc đầu chuẩn hoá như quan sát V3: −1 = cúi 15°, 0 = ngang, 1 = ngẩng 30°.
func get_head_tilt_norm() -> float:
	return head_tilt / HEAD_TILT_UP if head_tilt >= 0.0 else head_tilt / HEAD_TILT_DOWN


func _apply_head_tilt(delta: float) -> void:
	var step: float = HEAD_TILT_RATE * delta
	head_tilt = clampf(head_tilt + clampf(head_tilt_target - head_tilt, -step, step), -HEAD_TILT_DOWN, HEAD_TILT_UP)
	var head := get_node_or_null(^"Head") as Node3D
	if head != null:
		head.rotation.x = head_tilt
	var mount := get_node_or_null(^"CameraMount") as Node3D
	if mount != null:
		mount.rotation.x = head_tilt          # +X: camera nhìn −Z ngẩng lên


func _physics_process(delta: float) -> void:
	_apply_head_tilt(delta)
	# Không còn thao tác "lùi + xoay" hardcode khi kẹt (D-071): policy V2 thấy tường/vật cản qua depth
	# và tự xoay tại chỗ (v = 0, omega != 0) giống hệt môi trường huấn luyện.
	rotation.y += _omega * delta
	var forward: Vector3 = -global_transform.basis.z
	velocity = Vector3(forward.x, 0.0, forward.z).normalized() * _v
	move_and_slide()


static func clamp_command(v: float, omega: float) -> Vector2:
	return Vector2(clampf(v, 0.0, Config.V_MAX), clampf(omega, -Config.OMEGA_MAX, Config.OMEGA_MAX))
