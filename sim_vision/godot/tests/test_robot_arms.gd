extends GutTest

const RobotScene: PackedScene = preload("res://scenes/robot.tscn")


func _robot() -> Robot:
	var robot := RobotScene.instantiate() as Robot
	add_child_autofree(robot)
	return robot


func test_robot_has_two_arms() -> void:
	var robot := _robot()
	assert_not_null(robot.arms)
	assert_not_null(robot.arms.get_node_or_null(^"ShoulderL"))
	assert_not_null(robot.arms.get_node_or_null(^"ShoulderR"))


func test_robot_camera_does_not_render_arms() -> void:
	var robot := _robot()
	var cam := robot.get_camera_viewport().get_node("RobotCamera") as Camera3D
	assert_eq(cam.cull_mask & RobotArms.VISUAL_LAYER, 0, "ảnh YOLO không có tay robot")
	assert_ne(cam.cull_mask & 1, 0, "vẫn thấy layer thế giới")


func test_arm_meshes_only_on_arm_layer_and_no_collision() -> void:
	var robot := _robot()
	var meshes: Array[Node] = robot.arms.find_children("*", "MeshInstance3D", true, false)
	assert_gt(meshes.size(), 0)
	for m: Node in meshes:
		assert_eq((m as MeshInstance3D).layers, RobotArms.VISUAL_LAYER)
	assert_eq(robot.arms.find_children("*", "CollisionObject3D", true, false).size(), 0)


func test_choose_side() -> void:
	assert_eq(RobotArms.choose_side(Vector3(1, 0, -1)), RobotArms.Side.RIGHT)
	assert_eq(RobotArms.choose_side(Vector3(-1, 0, -1)), RobotArms.Side.LEFT)


func test_clamp_point_dir_limits_yaw_and_pitch() -> void:
	var behind: Vector3 = RobotArms.clamp_point_dir(Vector3(0.1, 0, 1))   # gần như ngay sau lưng, lệch phải
	var yaw: float = atan2(behind.x, -behind.z)
	assert_almost_eq(yaw, RobotArms.MAX_YAW_RAD, 1e-4)
	var up: Vector3 = RobotArms.clamp_point_dir(Vector3(0, 5, -1))
	assert_almost_eq(asin(up.y), RobotArms.PITCH_MAX_RAD, 1e-4, "không giơ tay lên trước camera")
	var ahead: Vector3 = RobotArms.clamp_point_dir(Vector3(0, -0.2, -3))
	assert_almost_eq(ahead.length(), 1.0, 1e-5)
	assert_lt(ahead.z, -0.9)


func test_point_at_moves_right_arm_toward_target() -> void:
	var robot := _robot()
	var target := Vector3(3.0, 0.5, -2.0)   # trước, bên phải
	robot.arms.point_at(target)
	assert_eq(robot.arms.pointing_side(), RobotArms.Side.RIGHT)
	for i: int in 60:
		robot.arms._process(1.0 / 30.0)
	var sh := robot.arms.get_node("ShoulderR") as Node3D
	var arm_dir: Vector3 = -sh.global_basis.z
	var want: Vector3 = (target - sh.global_position).normalized()
	assert_gt(arm_dir.dot(want), 0.95, "tay phải chỉ về đích")
	var left := robot.arms.get_node("ShoulderL") as Node3D
	assert_lt((-left.global_basis.z).y, -0.8, "tay trái vẫn buông")


func test_point_duration_then_lower() -> void:
	var robot := _robot()
	robot.arms.point_at(Vector3(-2, 0.5, -2), 0.5)
	assert_eq(robot.arms.pointing_side(), RobotArms.Side.LEFT)
	for i: int in 20:
		robot.arms._process(0.05)
	assert_eq(robot.arms.pointing_side(), RobotArms.Side.NONE)


func test_wave_then_stops() -> void:
	var robot := _robot()
	robot.arms.wave(0.3)
	assert_true(robot.arms.is_waving())
	for i: int in 10:
		robot.arms._process(0.05)
	assert_false(robot.arms.is_waving())


func test_arm_never_reaches_across_body() -> void:
	var left_target := Vector3(-1.0, -0.2, -0.5)                      # trước, bên trái
	var r: Vector3 = RobotArms.clamp_point_dir(left_target, RobotArms.Side.RIGHT)
	assert_gte(atan2(r.x, -r.z), -RobotArms.INWARD_YAW_RAD - 1e-4, "tay phải không quét sang trái qua thân")
	var l: Vector3 = RobotArms.clamp_point_dir(Vector3(1.0, -0.2, -0.5), RobotArms.Side.LEFT)
	assert_lte(atan2(l.x, -l.z), RobotArms.INWARD_YAW_RAD + 1e-4)


func test_side_switch_with_hysteresis() -> void:
	assert_eq(RobotArms.side_for_yaw(deg_to_rad(-5.0), RobotArms.Side.RIGHT), RobotArms.Side.RIGHT, "lệch chút ➔ giữ tay")
	assert_eq(RobotArms.side_for_yaw(deg_to_rad(-40.0), RobotArms.Side.RIGHT), RobotArms.Side.LEFT, "sang hẳn trái ➔ đổi tay")
	assert_eq(RobotArms.side_for_yaw(deg_to_rad(5.0), RobotArms.Side.LEFT), RobotArms.Side.LEFT)
	assert_eq(RobotArms.side_for_yaw(deg_to_rad(40.0), RobotArms.Side.LEFT), RobotArms.Side.RIGHT)


func test_point_switches_arm_when_robot_turns() -> void:
	var robot := _robot()
	var target := Vector3(2.0, 0.5, -2.0)       # trước-phải
	robot.arms.point_at(target)
	robot.arms._process(0.05)
	assert_eq(robot.arms.pointing_side(), RobotArms.Side.RIGHT)
	robot.rotation.y = deg_to_rad(-90.0)        # robot quay phải 90° ➔ đích giờ ở trước-trái
	robot.arms._process(0.05)
	assert_eq(robot.arms.pointing_side(), RobotArms.Side.LEFT, "đổi tay, không vươn tay phải qua thân")
