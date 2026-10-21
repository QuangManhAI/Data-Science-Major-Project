class_name Robot
extends CharacterBody3D
## Robot differential drive (G5). Hướng trước là −Z cục bộ.
## omega > 0 ➔ quay trái (ngược chiều kim đồng hồ nhìn từ trên xuống), đúng Mục 2.4.

var _v: float = 0.0
var _omega: float = 0.0

@onready var _viewport: SubViewport = $CameraViewport
@onready var _front_ray: RayCast3D = $FrontRay


func _ready() -> void:
	_viewport.size = Vector2i(Config.IMG_W, Config.IMG_H)


## Nhận lệnh vận tốc; v kẹp về [0, V_MAX], omega kẹp về [-OMEGA_MAX, OMEGA_MAX].
func set_command(v: float, omega: float) -> void:
	var c: Vector2 = Robot.clamp_command(v, omega)
	_v = c.x
	_omega = c.y


func get_command() -> Vector2:
	return Vector2(_v, _omega)


func get_camera_viewport() -> SubViewport:
	return get_node(^"CameraViewport") as SubViewport  # dùng được cả trước _ready


## Khoảng cách từ đầu raycast phía trước tới vật cản; không chạm ➔ INF.
func front_obstacle_distance() -> float:
	_front_ray.force_raycast_update()
	if not _front_ray.is_colliding():
		return INF
	return _front_ray.global_position.distance_to(_front_ray.get_collision_point())


func _physics_process(delta: float) -> void:
	rotation.y += _omega * delta
	var forward: Vector3 = -global_transform.basis.z
	velocity = Vector3(forward.x, 0.0, forward.z).normalized() * _v
	move_and_slide()


static func clamp_command(v: float, omega: float) -> Vector2:
	return Vector2(clampf(v, 0.0, Config.V_MAX), clampf(omega, -Config.OMEGA_MAX, Config.OMEGA_MAX))
