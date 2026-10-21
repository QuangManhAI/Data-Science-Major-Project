class_name Room
extends Node3D
## Phòng 8 m × 6 m, trần 3 m, gốc tọa độ ở tâm sàn (G3).
## Đồ vật là các StaticBody3D con của node `Furniture`, collision dạng BoxShape3D.

const SIZE_X: float = 8.0
const SIZE_Z: float = 6.0
const HEIGHT: float = 3.0
const MAX_TRIES: int = 100

@onready var _furniture: Node3D = $Furniture


## Điểm ngẫu nhiên trên sàn (y = 0), cách tường và mọi đồ vật ≥ `min_clearance`.
## Truyền `rng` để lặp lại được (G8 dùng seed); bỏ trống ➔ ngẫu nhiên.
## Thử tối đa MAX_TRIES lần; không tìm được ➔ push_error và trả Vector3.ZERO.
func get_free_position(min_clearance: float, rng: RandomNumberGenerator = null) -> Vector3:
	if rng == null:
		rng = RandomNumberGenerator.new()
		rng.randomize()
	var hx: float = SIZE_X / 2.0 - min_clearance
	var hz: float = SIZE_Z / 2.0 - min_clearance
	if hx >= 0.0 and hz >= 0.0:
		var footprints: Array[Rect2] = get_furniture_footprints()
		for i: int in MAX_TRIES:
			var p := Vector3(rng.randf_range(-hx, hx), 0.0, rng.randf_range(-hz, hz))
			if is_clear(p, footprints, min_clearance):
				return p
	push_error("Room.get_free_position: no free position for clearance %.2f" % min_clearance)
	return Vector3.ZERO


## Hình chiếu XZ (Rect2: x ➔ x, y ➔ z) của collision từng đồ vật, theo tọa độ thế giới.
func get_furniture_footprints() -> Array[Rect2]:
	var rects: Array[Rect2] = []
	for node: Node in _furniture.find_children("*", "CollisionShape3D", true, false):
		var cs := node as CollisionShape3D
		var box := cs.shape as BoxShape3D
		if box == null:
			continue
		var h: Vector3 = box.size / 2.0
		var rect := Rect2()
		for i: int in 8:
			var corner := Vector3(h.x if i & 1 else -h.x, h.y if i & 2 else -h.y, h.z if i & 4 else -h.z)
			var w: Vector3 = cs.global_transform * corner
			var p := Vector2(w.x, w.z)
			rect = Rect2(p, Vector2.ZERO) if i == 0 else rect.expand(p)
		rects.append(rect)
	return rects


static func distance_to_rect(p: Vector2, r: Rect2) -> float:
	var dx: float = maxf(maxf(r.position.x - p.x, 0.0), p.x - r.end.x)
	var dz: float = maxf(maxf(r.position.y - p.y, 0.0), p.y - r.end.y)
	return sqrt(dx * dx + dz * dz)


static func is_clear(p: Vector3, footprints: Array[Rect2], clearance: float) -> bool:
	var q := Vector2(p.x, p.z)
	for r: Rect2 in footprints:
		if distance_to_rect(q, r) < clearance:
			return false
	return true
