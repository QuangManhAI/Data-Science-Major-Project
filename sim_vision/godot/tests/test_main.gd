extends GutTest

const RoomScene: PackedScene = preload("res://scenes/room.tscn")
const CMD := {"state": Config.RobotState.APPROACH, "v": 0.5, "omega": 0.2, "target": {}}


func test_safety_too_close_to_human() -> void:
	var out: Dictionary = Main.apply_safety(CMD, 0.7, 2.0)
	assert_eq(out["v"], 0.0)
	assert_almost_eq(out["omega"], 0.2, 1e-6)
	assert_eq(out["state"], Config.RobotState.APPROACH)


func test_safety_obstacle_ahead() -> void:
	var out: Dictionary = Main.apply_safety(CMD, 2.0, 0.4)
	assert_eq(out["v"], 0.0)
	assert_almost_eq(out["omega"], 0.2, 1e-6)


func test_safety_clear_unchanged() -> void:
	assert_eq(Main.apply_safety(CMD, 2.0, 2.0), CMD)


func test_safety_no_obstacle_inf() -> void:
	assert_eq(Main.apply_safety(CMD, 2.0, INF), CMD)


func test_safety_thresholds_strict() -> void:
	# Điều kiện là "<": đúng bằng ngưỡng thì vẫn được tiến.
	assert_almost_eq(Main.apply_safety(CMD, Config.SAFE_DIST_HUMAN, 2.0)["v"], 0.5, 1e-6)
	assert_almost_eq(Main.apply_safety(CMD, 2.0, Config.SAFE_DIST_OBSTACLE)["v"], 0.5, 1e-6)


func test_safety_does_not_mutate_input() -> void:
	var cmd: Dictionary = CMD.duplicate()
	Main.apply_safety(cmd, 0.1, 0.1)
	assert_eq(cmd, CMD)


func _room() -> Room:
	var r := RoomScene.instantiate() as Room
	add_child_autofree(r)
	return r


func test_start_positions_separated_and_free() -> void:
	var r: Room = _room()
	var rects: Array[Rect2] = r.get_furniture_footprints()
	for s: int in 50:
		var rng := RandomNumberGenerator.new()
		rng.seed = s
		var p: Dictionary = Main.pick_start_positions(r, rng)
		assert_true(Human.flat_distance(p["robot"], p["human"]) >= 3.0, "seed %d separation" % s)
		for key: String in ["robot", "human"]:
			var pos: Vector3 = p[key]
			assert_true(absf(pos.x) <= 3.4 and absf(pos.z) <= 2.4, "seed %d %s inside" % [s, key])
			assert_true(Room.is_clear(pos, rects, 0.6), "seed %d %s clear" % [s, key])
		assert_between(p["robot_yaw"], -PI, PI)


func test_start_positions_same_seed_same_result() -> void:
	var r: Room = _room()
	var a := RandomNumberGenerator.new()
	var b := RandomNumberGenerator.new()
	a.seed = 123
	b.seed = 123
	assert_eq(Main.pick_start_positions(r, a), Main.pick_start_positions(r, b))
	var c := RandomNumberGenerator.new()
	c.seed = 124
	assert_ne(Main.pick_start_positions(r, a), Main.pick_start_positions(r, c), "seed khác ➔ kết quả khác")
