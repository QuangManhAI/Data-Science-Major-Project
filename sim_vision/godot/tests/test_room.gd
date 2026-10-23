extends GutTest

const RoomScene: PackedScene = preload("res://scenes/room.tscn")

var room: Room


func before_each() -> void:
	room = RoomScene.instantiate() as Room
	add_child_autofree(room)


func test_has_furniture_and_wall_footprints_inside_hospital() -> void:
	var furniture: Array[Rect2] = room.get_furniture_footprints()
	var walls: Array[Rect2] = room.get_wall_footprints()
	assert_true(furniture.size() >= 15, "expected >= 15 furniture footprints, got %d" % furniture.size())
	assert_true(walls.size() >= 8, "expected >= 8 interior wall footprints, got %d" % walls.size())
	for r: Rect2 in room.get_obstacle_footprints():
		assert_true(r.position.x >= -12.01 and r.end.x <= 12.01, "x inside hospital: %s" % r)
		assert_true(r.position.y >= -9.01 and r.end.y <= 9.01, "z inside hospital: %s" % r)


func test_has_five_sliding_doors() -> void:
	var doorsNode: Node = room.get_node_or_null(^"Doors")
	assert_not_null(doorsNode)
	assert_eq(doorsNode.get_child_count(), 5)
	for child: Node in doorsNode.get_children():
		assert_true(child is SlidingDoor, "%s should be SlidingDoor" % child.name)


func test_ceiling_visibility_toggle() -> void:
	assert_true(room.is_ceiling_visible())
	room.set_ceiling_visible(false)
	assert_false(room.is_ceiling_visible())
	room.set_ceiling_visible(true)
	assert_true(room.is_ceiling_visible())


func test_room_waypoints_and_room_at() -> void:
	var expected_Map := {
		"101": "101",
		"cap_cuu": "101",
		"110": "110",
		"112": "112",
		"nha_thuoc": "nha_thuoc",
		"tiep_don": "tiep_don",
		"ghe_cho": "tiep_don",
		"thang_may": "thang_may",
		"nha_ve_sinh": "thang_may",
		"hanh_lang": "hanh_lang",
	}
	for key: String in expected_Map.keys():
		var wp: Vector3 = room.get_room_waypoint(key)
		assert_eq(room.get_room_at(wp), expected_Map[key], "waypoint for %s should lie in %s" % [key, expected_Map[key]])


func test_free_position_100_calls() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 42
	var obstacles: Array[Rect2] = room.get_obstacle_footprints()
	for i: int in 100:
		var p: Vector3 = room.get_free_position(0.5, rng)
		assert_eq(p.y, 0.0)
		assert_true(absf(p.x) <= 12.0 and absf(p.z) <= 9.0, "inside hospital: %s" % p)
		for r: Rect2 in obstacles:
			assert_true(Room.distance_to_rect(Vector2(p.x, p.z), r) >= 0.5, "clear of %s: %s" % [r, p])


func test_free_position_in_each_room() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 99
	var obstacles: Array[Rect2] = room.get_obstacle_footprints()
	for key: String in ["101", "110", "112", "nha_thuoc", "tiep_don", "thang_may", "hanh_lang"]:
		var p: Vector3 = room.get_free_position_in_room(key, 0.4, rng)
		assert_ne(p, Vector3.ZERO, "found free pos in %s" % key)
		assert_eq(room.get_room_at(p), key, "position %s is inside room %s" % [p, key])
		for r: Rect2 in obstacles:
			assert_true(Room.distance_to_rect(Vector2(p.x, p.z), r) >= 0.4, "clear in %s" % key)


func test_free_position_center_is_free() -> void:
	# Giao lộ hành lang chính tại (0, 0, 0) phải hoàn toàn trống.
	assert_true(Room.is_clear(Vector3.ZERO, room.get_obstacle_footprints(), 1.0))


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


func test_has_medical_staff_and_waiting_patients_and_new_props() -> void:
	var furniture: Node = room.get_node_or_null(^"Furniture")
	assert_not_null(furniture)
	for staff_name: String in ["MedicalStaffNurse", "MedicalStaffDoctor"]:
		var staff := furniture.get_node_or_null(NodePath(staff_name)) as HospitalNpc
		assert_not_null(staff, "expected %s in Furniture" % staff_name)
		assert_true(staff.staff_role in ["nurse", "doctor"])
		assert_not_null(staff.get_node_or_null(^"RoleBadge"))
	for i: int in range(1, 7):
		var pat_name: String = "WaitingPatient%d" % i
		var pat := furniture.get_node_or_null(NodePath(pat_name)) as HospitalNpc
		assert_not_null(pat, "expected %s in Furniture" % pat_name)
		assert_true(pat.is_seated, "%s should be seated" % pat_name)
		assert_eq(pat.staff_role, "patient")
	for prop_name: String in [
		"LobbyQueueKiosk",
		"LobbyWaterDispenser",
		"LobbyInfoStandee",
		"LobbySanitizerStand",
		"CorridorSanitizerStand",
		"EmergencyVitalMonitor",
		"EmergencyIVStand",
		"EmergencyStretcherCart",
		"ClinicVitalMonitor",
		"ClinicIVStand",
	]:
		assert_not_null(furniture.get_node_or_null(NodePath(prop_name)), "expected prop %s" % prop_name)

func test_npc_walking_and_pardon_step_aside() -> void:
	var furn: Node = room.get_node_or_null(^"Furniture")
	var doc := furn.get_node_or_null(^"MedicalStaffDoctor") as HospitalNpc
	assert_not_null(doc)
	assert_true(doc.walks, "doctor should patrol corridor")
	assert_true(doc.pardon_enabled, "doctor should have pardon enabled")
	var p_target: Vector3 = HospitalNpc.compute_pardon_target(
		Vector3(0.0, 0.0, 0.0), Vector3(-1.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), 0.55
	)
	assert_almost_eq(absf(p_target.z), 0.55, 1e-4, "steps aside perpendicular to robot heading")
	var fake_robot := Node3D.new()
	add_child_autofree(fake_robot)
	fake_robot.global_position = doc.global_position + Vector3(0.8, 0.0, 0.0)
	doc.setup_robot(fake_robot)
	await wait_physics_frames(3)
	assert_true(doc.is_pardoning(), "doctor enters pardon state when robot is within trigger distance")
	fake_robot.global_position = doc.global_position + Vector3(5.0, 0.0, 0.0)
	await wait_physics_frames(3)
	assert_false(doc.is_pardoning(), "doctor exits pardon state when robot moves away")
