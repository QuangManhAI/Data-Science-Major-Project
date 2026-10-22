class_name Room
extends Node3D
## Bệnh viện tầng 1 kích thước 24 m × 18 m, trần 3.2 m, gốc tọa độ ở tâm sàn (0, 0, 0).
## Gồm 6 khu vực nối với Hành lang chính qua cửa trượt tự động (SlidingDoor):
##   - 101 / cap_cuu : Khoa Cấp Cứu (Tây Bắc)
##   - 110           : Phòng Xét Nghiệm Máu (Bắc)
##   - 112           : Phòng Khám Nội (Đông Bắc)
##   - nha_thuoc     : Nhà Thuốc (Tây Nam)
##   - tiep_don      : Sảnh Chính - Quầy Tiếp Đón & Khu Ghế Chờ (Nam)
##   - thang_may     : Khu Thang Máy & Nhà Vệ Sinh (Đông Nam)
##   - hanh_lang     : Hành Lang Chính Đông - Tây (Giữa)

const SIZE_X: float = 24.0
const SIZE_Z: float = 18.0
const HEIGHT: float = 3.2
const MAX_TRIES: int = 100

## Vùng mặc định cho get_free_position() (Hành lang trung tâm + Sảnh chính thông nhau).
const DEFAULT_SPAWN_RECT := Rect2(Vector2(-5.6, -1.8), Vector2(11.2, 10.2))

const ROOMS: Dictionary = {
	"101": {
		"name": "Khoa cấp cứu",
		"rect": Rect2(Vector2(-12.0, -9.0), Vector2(8.0, 7.0)),
		"door": Vector3(-8.0, 0.0, -2.0),
		"waypoint": Vector3(-8.0, 0.0, -5.2),
	},
	"cap_cuu": {
		"name": "Khoa cấp cứu",
		"rect": Rect2(Vector2(-12.0, -9.0), Vector2(8.0, 7.0)),
		"door": Vector3(-8.0, 0.0, -2.0),
		"waypoint": Vector3(-8.0, 0.0, -5.2),
	},
	"110": {
		"name": "Phòng xét nghiệm máu",
		"rect": Rect2(Vector2(-4.0, -9.0), Vector2(8.0, 7.0)),
		"door": Vector3(0.0, 0.0, -2.0),
		"waypoint": Vector3(0.0, 0.0, -5.2),
	},
	"112": {
		"name": "Phòng khám nội",
		"rect": Rect2(Vector2(4.0, -9.0), Vector2(8.0, 7.0)),
		"door": Vector3(8.0, 0.0, -2.0),
		"waypoint": Vector3(8.0, 0.0, -5.2),
	},
	"nha_thuoc": {
		"name": "Nhà thuốc",
		"rect": Rect2(Vector2(-12.0, 2.0), Vector2(6.0, 7.0)),
		"door": Vector3(-9.0, 0.0, 2.0),
		"waypoint": Vector3(-9.0, 0.0, 5.2),
	},
	"tiep_don": {
		"name": "Sảnh chính - Quầy tiếp đón",
		"rect": Rect2(Vector2(-6.0, 2.0), Vector2(12.0, 7.0)),
		"door": Vector3(0.0, 0.0, 2.0),
		"waypoint": Vector3(0.0, 0.0, 5.2),
	},
	"ghe_cho": {
		"name": "Khu ghế chờ",
		"rect": Rect2(Vector2(-6.0, 2.0), Vector2(12.0, 7.0)),
		"door": Vector3(0.0, 0.0, 2.0),
		"waypoint": Vector3(-2.5, 0.0, 4.5),
	},
	"thu_ngan": {
		"name": "Quầy thu ngân",
		"rect": Rect2(Vector2(-6.0, 2.0), Vector2(12.0, 7.0)),
		"door": Vector3(0.0, 0.0, 2.0),
		"waypoint": Vector3(1.5, 0.0, 6.2),
	},
	"bhyt": {
		"name": "Quầy bảo hiểm y tế",
		"rect": Rect2(Vector2(-6.0, 2.0), Vector2(12.0, 7.0)),
		"door": Vector3(0.0, 0.0, 2.0),
		"waypoint": Vector3(-1.5, 0.0, 6.2),
	},
	"thang_may": {
		"name": "Thang máy & Nhà vệ sinh",
		"rect": Rect2(Vector2(6.0, 2.0), Vector2(6.0, 7.0)),
		"door": Vector3(9.0, 0.0, 2.0),
		"waypoint": Vector3(9.0, 0.0, 5.2),
	},
	"nha_ve_sinh": {
		"name": "Nhà vệ sinh",
		"rect": Rect2(Vector2(6.0, 2.0), Vector2(6.0, 7.0)),
		"door": Vector3(9.0, 0.0, 2.0),
		"waypoint": Vector3(9.5, 0.0, 6.5),
	},
	"hanh_lang": {
		"name": "Hành lang chính",
		"rect": Rect2(Vector2(-12.0, -2.0), Vector2(24.0, 4.0)),
		"door": Vector3(0.0, 0.0, 0.0),
		"waypoint": Vector3(0.0, 0.0, 0.0),
	},
}

const CANONICAL_ROOM_KEYS: Array[String] = [
	"101", "110", "112", "nha_thuoc", "tiep_don", "thang_may", "hanh_lang"
]

@onready var _furniture: Node3D = $Furniture
@onready var _interior_walls: Node3D = get_node_or_null(^"InteriorWalls")
@onready var _ceiling: MeshInstance3D = get_node_or_null(^"Shell/Ceiling")


## Ẩn hoặc hiện trần nhà (hữu ích cho camera nhìn từ trên xuống TOP_DOWN kiểu RimWorld).
func set_ceiling_visible(v: bool) -> void:
	if _ceiling != null:
		_ceiling.visible = v


func is_ceiling_visible() -> bool:
	return _ceiling != null and _ceiling.visible


## Điểm ngẫu nhiên trên sàn (y = 0) trong Sảnh chính + Hành lang trung tâm,
## cách tường và mọi đồ vật ≥ `min_clearance`.
func get_free_position(min_clearance: float, rng: RandomNumberGenerator = null) -> Vector3:
	return _sample_in_rect(DEFAULT_SPAWN_RECT, min_clearance, rng)


## Điểm ngẫu nhiên trên sàn (y = 0) nằm trong phòng `room_key`, cách tường phòng và đồ vật ≥ `min_clearance`.
func get_free_position_in_room(room_key: String, min_clearance: float, rng: RandomNumberGenerator = null) -> Vector3:
	if not ROOMS.has(room_key):
		push_error("Room.get_free_position_in_room: unknown room_key '%s'" % room_key)
		return Vector3.ZERO
	var r: Rect2 = ROOMS[room_key]["rect"]
	return _sample_in_rect(r, min_clearance, rng)


func _sample_in_rect(bounds: Rect2, min_clearance: float, rng: RandomNumberGenerator = null) -> Vector3:
	if rng == null:
		rng = RandomNumberGenerator.new()
		rng.randomize()
	var x_min: float = bounds.position.x + min_clearance
	var x_max: float = bounds.end.x - min_clearance
	var z_min: float = bounds.position.y + min_clearance
	var z_max: float = bounds.end.y - min_clearance
	if x_min <= x_max and z_min <= z_max:
		var footprints: Array[Rect2] = get_obstacle_footprints()
		for i: int in MAX_TRIES:
			var p := Vector3(rng.randf_range(x_min, x_max), 0.0, rng.randf_range(z_min, z_max))
			if is_clear(p, footprints, min_clearance):
				return p
	push_error("Room.get_free_position: no free position for clearance %.2f" % min_clearance)
	return Vector3.ZERO


## Trả tọa độ đích (waypoint) cho RL / dẫn đường ứng với mã phòng hoặc cơ sở.
func get_room_waypoint(room_key: String) -> Vector3:
	if ROOMS.has(room_key):
		return ROOMS[room_key]["waypoint"]
	return Vector3.ZERO


## Trả mã phòng chứa điểm `pos` (chiếu xuống mặt phẳng XZ). Không thuộc phòng nào ➔ "".
func get_room_at(pos: Vector3) -> String:
	var p := Vector2(pos.x, pos.z)
	for key: String in CANONICAL_ROOM_KEYS:
		var r: Rect2 = ROOMS[key]["rect"]
		if r.has_point(p):
			return key
	return ""


## Hình chiếu XZ (Rect2) của collision từng đồ vật trong node `Furniture`.
func get_furniture_footprints() -> Array[Rect2]:
	return _collect_footprints(_furniture)


## Hình chiếu XZ (Rect2) của các tường ngăn nội thất trong node `InteriorWalls`.
func get_wall_footprints() -> Array[Rect2]:
	return _collect_footprints(_interior_walls)


## Hình chiếu XZ (Rect2) của cả đồ vật (`Furniture`) và tường ngăn nội thất (`InteriorWalls`).
func get_obstacle_footprints() -> Array[Rect2]:
	var rects: Array[Rect2] = _collect_footprints(_furniture)
	if _interior_walls != null:
		rects.append_array(_collect_footprints(_interior_walls))
	return rects


static func _collect_footprints(root: Node3D) -> Array[Rect2]:
	var rects: Array[Rect2] = []
	if root == null:
		return rects
	for node: Node in root.find_children("*", "CollisionShape3D", true, false):
		var cs := node as CollisionShape3D
		if cs == null or cs.disabled:
			continue
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
