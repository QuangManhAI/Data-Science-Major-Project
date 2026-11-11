extends GutTest

const RobotScene: PackedScene = preload("res://scenes/robot.tscn")
const OverlayScene: PackedScene = preload("res://scenes/view_overlay.tscn")


func test_set_command_clamps_example() -> void:
	var robot := RobotScene.instantiate() as Robot
	add_child_autofree(robot)
	robot.set_command(2.0, -3.0)
	assert_almost_eq(robot.get_command().x, 0.8, 1e-6)
	assert_almost_eq(robot.get_command().y, -1.5, 1e-6)


func test_clamp_command_bounds() -> void:
	assert_eq(Robot.clamp_command(-0.5, 0.0), Vector2(0.0, 0.0), "không lùi")
	assert_almost_eq(Robot.clamp_command(0.3, 5.0).y, Config.OMEGA_MAX, 1e-6)
	assert_almost_eq(Robot.clamp_command(0.3, 0.5).x, 0.3, 1e-6)
	assert_almost_eq(Robot.clamp_command(0.3, 0.5).y, 0.5, 1e-6)


func test_camera_viewport() -> void:
	var robot := RobotScene.instantiate() as Robot
	add_child_autofree(robot)
	var vp: SubViewport = robot.get_camera_viewport()
	assert_eq(vp.size, Vector2i(Config.IMG_W, Config.IMG_H))
	assert_eq(vp.render_target_update_mode, SubViewport.UPDATE_ALWAYS)
	var cam := vp.get_node("RobotCamera") as Camera3D
	assert_almost_eq(cam.fov, 70.0, 1e-6)
	await wait_physics_frames(2)
	assert_almost_eq(cam.global_position.y, 1.2, 1e-4, "camera cao 1.2 m")


func test_front_obstacle_distance_inf_without_hit() -> void:
	var robot := RobotScene.instantiate() as Robot
	add_child_autofree(robot)
	await wait_physics_frames(2)
	assert_eq(robot.front_obstacle_distance(), INF)


func test_front_obstacle_distance_with_wall() -> void:
	# Đặt ở x = 50 để không chồng lên robot của test trước (chưa kịp giải phóng).
	var robot := RobotScene.instantiate() as Robot
	robot.position = Vector3(50, 0, 0)
	add_child_autofree(robot)
	var wall := StaticBody3D.new()
	var cs := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(2, 2, 0.2)
	cs.shape = box
	wall.add_child(cs)
	add_child_autofree(wall)
	wall.position = Vector3(50, 0.5, -1.05)  # mặt tường ở z = -0.95, đầu tia ở z = -0.25
	await wait_physics_frames(3)
	assert_almost_eq(robot.front_obstacle_distance(), 0.7, 1e-3)


func test_moves_forward_along_minus_z() -> void:
	var robot := RobotScene.instantiate() as Robot
	add_child_autofree(robot)
	robot.set_command(0.5, 0.0)
	await wait_seconds(1.0)
	assert_almost_eq(robot.global_position.x, 0.0, 1e-3)
	assert_between(-robot.global_position.z, 0.4, 0.6)


func test_positive_omega_turns_left() -> void:
	# Mục 2.4: omega > 0 ➔ quay trái ➔ hướng trước (−Z) lệch về −X.
	var robot := RobotScene.instantiate() as Robot
	add_child_autofree(robot)
	robot.set_command(0.0, 1.0)
	await wait_seconds(0.5)
	var forward: Vector3 = -robot.global_transform.basis.z
	assert_lt(forward.x, 0.0)


func test_norm_to_pixel_example() -> void:
	var r: Rect2 = ViewOverlay.norm_to_pixel({"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}, Vector2(320, 240))
	assert_eq(r.position, Vector2(80, 60))
	assert_eq(r.end, Vector2(240, 180))


func test_norm_to_pixel_full_frame() -> void:
	var r: Rect2 = ViewOverlay.norm_to_pixel({"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}, Vector2(320, 240))
	assert_eq(r, Rect2(0, 0, 320, 240))


func test_overlay_texts() -> void:
	assert_eq(ViewOverlay.label_text({"conf": 0.87, "class_name": "patient"}), "patient 0.87")
	assert_eq(ViewOverlay.label_text({"conf": 0.95, "class_name": "doctor"}), "doctor 0.95")
	assert_eq(ViewOverlay.label_text({"conf": 0.82, "class_name": "wheelchair"}), "wheelchair 0.82")
	assert_eq(ViewOverlay.connection_text(true), "YOLO: đã kết nối")
	assert_eq(ViewOverlay.connection_text(false), "YOLO: mất kết nối")


func test_overlay_layout_top_right() -> void:
	var overlay := OverlayScene.instantiate() as ViewOverlay
	add_child_autofree(overlay)
	assert_eq(overlay.size.x, ViewOverlay.TOTAL_W)
	assert_true(overlay.face_panel_rect().end.x <= overlay.front_image_rect().position.x, "ô mặt nằm bên trái camera trước")
	assert_eq(overlay.anchor_left, 1.0)
	assert_eq(overlay.offset_right, -16.0)
	assert_eq(overlay.offset_top, 16.0)
	assert_eq(overlay.front_image_rect().size, Vector2(320, 240), "ảnh camera giữ tỉ lệ 4:3")
	assert_true(overlay.front_image_rect().position.y >= ViewOverlay.HEADER_H, "tiêu đề nằm trên ảnh, không đè")
	assert_true(overlay.footer_rect().position.y >= overlay.sector_rect().end.y, "trạng thái ở thanh dưới, không đè ảnh")
	overlay.show_detections([{"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75, "conf": 0.87}], "SEARCH", true)
	overlay.set_status("Một dòng trạng thái rất dài rất dài rất dài rất dài rất dài rất dài rất dài rất dài")
	await wait_process_frames(2)
	pass_test("vẽ không lỗi")


func test_overlay_rear_panel_only_grows_when_active() -> void:
	var overlay := OverlayScene.instantiate() as ViewOverlay
	add_child_autofree(overlay)
	var robot := RobotScene.instantiate() as Robot
	add_child_autofree(robot)
	var h0: float = overlay.total_height()
	overlay.set_rear_viewport(robot.get_rear_viewport())
	var h_off: float = overlay.total_height()
	assert_gt(h_off, h0, "có thêm dòng tiêu đề camera sau")
	assert_lt(h_off - h0, ViewOverlay.IMG_H, "camera sau tắt ➔ không chiếm chỗ ảnh")
	overlay.set_rear_active(true)
	assert_almost_eq(overlay.total_height() - h_off, ViewOverlay.IMG_H, 0.01)
	assert_true(overlay.rear_image_rect().position.y >= overlay.rear_header_rect().end.y)
	var tracked := {"x1": 0.3, "y1": 0.2, "x2": 0.5, "y2": 0.9, "conf": 0.9, "class_id": 0}
	overlay.show_rear_detections([tracked, {"x1": 0.6, "y1": 0.2, "x2": 0.7, "y2": 0.8, "conf": 0.8, "class_id": 0}])
	overlay.set_tracked({}, tracked)
	await wait_process_frames(2)
	overlay.set_rear_active(false)
	assert_almost_eq(overlay.total_height(), h_off, 0.01)


func test_overlay_labels_do_not_overlap_or_leave_image() -> void:
	var area := Rect2(0, 22, 320, 240)
	var placed: Array[Rect2] = []
	var box := Rect2(100, 60, 60, 150)
	for i: int in 3:
		var r: Rect2 = ViewOverlay.place_label(box, Vector2(80, 16), area, placed)
		for p: Rect2 in placed:
			assert_false(p.intersects(r), "nhãn %d chồng nhãn trước" % i)
		assert_true(area.encloses(r), "nhãn nằm trong ảnh")
		placed.append(r)
	var edge: Rect2 = ViewOverlay.place_label(Rect2(300, 22, 20, 50), Vector2(80, 16), area, [] as Array[Rect2])
	assert_true(area.encloses(edge), "box sát mép phải ➔ nhãn dời vào trong")


func test_overlay_short_labels_and_ellipsize() -> void:
	var pat := {"class_id": 0, "conf": 0.91}
	assert_eq(ViewOverlay.short_label(pat, true, true), "đang dẫn")
	assert_eq(ViewOverlay.short_label(pat, false, true), "bệnh nhân khác")
	assert_eq(ViewOverlay.short_label(pat, false, false), "bệnh nhân 91%")
	assert_eq(ViewOverlay.short_label({"class_id": 1, "conf": 0.8}, false, false), "bác sĩ")
	assert_eq(ViewOverlay.short_label({"class_id": 5, "conf": 0.8}, false, false), "", "vật cản không ghi nhãn")
	var font: Font = ThemeDB.fallback_font
	var t: String = ViewOverlay.ellipsize(font, "Bắt đầu dẫn đường bệnh nhân tới: thang_may (Tầng 3) rất dài", 13, 120.0)
	assert_true(t.ends_with("…"))
	assert_lte(font.get_string_size(t, HORIZONTAL_ALIGNMENT_LEFT, -1, 13).x, 120.0)


func test_rear_camera_faces_backward_and_starts_off() -> void:
	var robot := RobotScene.instantiate() as Robot
	add_child_autofree(robot)
	await wait_physics_frames(2)
	assert_false(robot.is_rear_camera_enabled(), "chỉ bật khi dẫn đường")
	var cam: Camera3D = robot.get_rear_camera()
	assert_gt((-cam.global_basis.z).z, 0.99, "camera sau nhìn về +Z (sau lưng robot)")
	assert_almost_eq(cam.global_position.y, 1.2, 1e-4)
	assert_eq(cam.cull_mask & RobotArms.VISUAL_LAYER, 0, "không thấy tay robot")
	robot.set_rear_camera_enabled(true)
	assert_true(robot.is_rear_camera_enabled())


func test_depth_readings_see_wall_and_respect_ignore() -> void:
	# V2 depth fan: 15 tia trái -> phải, tối đa 4 m; tia giữa thấy tường cách tâm robot 1.0 m.
	var robot := RobotScene.instantiate() as Robot
	robot.position = Vector3(80, 0, 0)
	add_child_autofree(robot)
	var wall := StaticBody3D.new()
	var cs := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(0.6, 2, 0.2)
	cs.shape = box
	wall.add_child(cs)
	add_child_autofree(wall)
	wall.position = Vector3(80, 0.5, -1.1)  # mặt tường ở z = -1.0
	await wait_physics_frames(3)
	var d: PackedFloat32Array = robot.get_depth_readings()
	assert_eq(d.size(), Robot.DEPTH_NUM_RAYS)
	assert_almost_eq(d[7], 1.0, 0.02)
	assert_almost_eq(d[0], Robot.DEPTH_MAX_RANGE, 1e-3)  # tia trái ngoài cùng không chạm
	robot.depth_ignore = wall  # người đang được theo dõi bị che khỏi depth
	assert_almost_eq(robot.get_depth_readings()[7], Robot.DEPTH_MAX_RANGE, 1e-3)


func test_no_hardcoded_escape_when_blocked() -> void:
	# Bỏ D-071: kẹt vào tường thì robot KHÔNG tự lùi/xoay; chỉ chạy theo lệnh (v, omega).
	var robot := RobotScene.instantiate() as Robot
	robot.position = Vector3(110, 0, 0)
	add_child_autofree(robot)
	var wall := StaticBody3D.new()
	var cs := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(2, 2, 0.2)
	cs.shape = box
	wall.add_child(cs)
	add_child_autofree(wall)
	wall.position = Vector3(110, 0.5, -0.5)
	var yaw0: float = robot.rotation.y
	robot.set_command(0.5, 0.0)
	await wait_seconds(1.5)
	assert_almost_eq(robot.rotation.y, yaw0, 1e-4)
	assert_lt(robot.global_position.z, 0.01)  # không bị đẩy lùi về +Z


func test_overlay_face_boxes_dashed_and_parsed() -> void:
	var r: Dictionary = VisionClient.parse_message(JSON.stringify({"type": "detections", "frame_id": 3, "infer_ms": 5.0,
			"detections": [], "faces": [{"x1": 0.4, "y1": 0.2, "x2": 0.46, "y2": 0.3, "conf": 0.9}, {"x1": 2.0}]}))
	assert_true(r["ok"])
	assert_eq(r["faces"].size(), 1, "box mặt sai bị bỏ")
	var old: Dictionary = VisionClient.parse_message(JSON.stringify({"type": "detections", "frame_id": 4, "infer_ms": 5.0, "detections": []}))
	assert_eq(old["faces"], [], "server cũ không có faces ➔ rỗng")
	assert_eq(ViewOverlay.rect_edges(Rect2(0, 0, 10, 20)).size(), 4)
	var overlay := OverlayScene.instantiate() as ViewOverlay
	add_child_autofree(overlay)
	overlay.show_faces(r["faces"])
	await wait_process_frames(2)
	pass_test("vẽ box mặt nét đứt không lỗi")



func test_face_crop_rect_inside_image() -> void:
	var r: Rect2i = Main.face_crop_rect({"x1": 0.45, "y1": 0.2, "x2": 0.5, "y2": 0.28}, Vector2i(640, 480))
	assert_true(Rect2i(0, 0, 640, 480).encloses(r))
	assert_almost_eq(float(r.size.y) / r.size.x, 1.25, 0.05, "tỉ lệ 4:5")
	assert_gt(r.size.x, int(0.05 * 640), "rộng hơn box mặt")
	var edge: Rect2i = Main.face_crop_rect({"x1": 0.97, "y1": 0.0, "x2": 1.0, "y2": 0.05}, Vector2i(640, 480))
	assert_true(Rect2i(0, 0, 640, 480).encloses(edge), "sát mép ➔ kẹp trong ảnh")
