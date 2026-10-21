class_name Human
extends CharacterBody3D
## Người tả thực (Rocketbox Male_Adult_06). Mặc định đứng yên (idle);
## `walks = true` ➔ đi chậm qua lần lượt các `waypoints` (lặp vòng).
## Mặt người hướng về +Z cục bộ.

const WALK_SPEED: float = 0.5     # m/s (G4.1)
const ARRIVE_TOL: float = 0.05    # m, coi như đã tới waypoint

@export var walks: bool = false
@export var waypoints: Array[Vector3] = []

var _wp_index: int = 0

@onready var _anim: AnimationPlayer = $AnimationPlayer


func _ready() -> void:
	_anim.play(&"human/walk" if _is_walking() else &"human/idle")


func _physics_process(delta: float) -> void:
	if not _is_walking():
		_play(&"human/idle")
		return
	_play(&"human/walk")
	var target: Vector3 = waypoints[_wp_index]
	if Human.flat_distance(global_position, target) <= ARRIVE_TOL:
		_wp_index = Human.next_waypoint_index(_wp_index, waypoints.size())
		target = waypoints[_wp_index]
	var step: Vector3 = Human.step_towards(global_position, target, WALK_SPEED * delta)
	var dir := Vector3(step.x - global_position.x, 0.0, step.z - global_position.z)
	if dir.length() > 0.0:
		rotation.y = atan2(dir.x, dir.z)  # quay mặt (+Z) theo hướng đi
		velocity = dir / delta
		move_and_slide()


func _is_walking() -> bool:
	return walks and waypoints.size() >= 2


func _play(anim_name: StringName) -> void:
	if _anim.current_animation != anim_name:
		_anim.play(anim_name, 0.3)


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
