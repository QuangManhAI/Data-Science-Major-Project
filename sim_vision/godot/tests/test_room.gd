extends GutTest

const RoomScene: PackedScene = preload("res://scenes/room.tscn")

var room: Room


func before_each() -> void:
	room = RoomScene.instantiate() as Room
	add_child_autofree(room)


func test_has_four_furniture_footprints_inside_room() -> void:
	var rects: Array[Rect2] = room.get_furniture_footprints()
	assert_eq(rects.size(), 4)
	for r: Rect2 in rects:
		assert_true(r.position.x >= -4.0 and r.end.x <= 4.0, "x inside room: %s" % r)
		assert_true(r.position.y >= -3.0 and r.end.y <= 3.0, "z inside room: %s" % r)


func test_free_position_100_calls() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 42
	var rects: Array[Rect2] = room.get_furniture_footprints()
	for i: int in 100:
		var p: Vector3 = room.get_free_position(0.5, rng)
		assert_eq(p.y, 0.0)
		assert_true(absf(p.x) <= 3.5 and absf(p.z) <= 2.5, "inside room: %s" % p)
		for r: Rect2 in rects:
			assert_true(Room.distance_to_rect(Vector2(p.x, p.z), r) >= 0.5, "clear of %s: %s" % [r, p])


func test_free_position_center_is_free() -> void:
	# Đồ vật đặt sát tường nên giữa phòng phải trống.
	assert_true(Room.is_clear(Vector3.ZERO, room.get_furniture_footprints(), 1.0))


func test_same_seed_same_position() -> void:
	var a := RandomNumberGenerator.new()
	var b := RandomNumberGenerator.new()
	a.seed = 7
	b.seed = 7
	assert_eq(room.get_free_position(0.6, a), room.get_free_position(0.6, b))


func test_impossible_clearance_returns_zero_and_errors() -> void:
	assert_eq(room.get_free_position(10.0), Vector3.ZERO)
	assert_push_error("no free position")


func test_distance_to_rect() -> void:
	var r := Rect2(Vector2(0, 0), Vector2(2, 1))
	assert_almost_eq(Room.distance_to_rect(Vector2(1, 0.5), r), 0.0, 1e-6)
	assert_almost_eq(Room.distance_to_rect(Vector2(3, 0.5), r), 1.0, 1e-6)
	assert_almost_eq(Room.distance_to_rect(Vector2(1, -2), r), 2.0, 1e-6)
	assert_almost_eq(Room.distance_to_rect(Vector2(5, 5), r), 5.0, 1e-6)
