extends GutTest
## docs/UPGRADE_HRI_PLAN.md mục B–C: câu thoại robot, giám sát khoảng cách, bệnh nhân dừng lại.

const DT: float = 0.1


func _run(m: EscortMonitor, dist: float, seen: bool, seconds: float) -> Array[String]:
	var out: Array[String] = []
	for i: int in int(round(seconds / DT)):
		out.append_array(m.step(dist, seen, DT))
	return out


func test_close_patient_never_triggers() -> void:
	var m := EscortMonitor.new()
	assert_eq(_run(m, 1.2, true, 60.0), [] as Array[String])
	assert_false(m.hold_robot())


func test_short_gap_is_tolerated() -> void:
	var m := EscortMonitor.new()
	var ev := _run(m, 3.5, true, EscortMonitor.FAR_DWELL_S - 0.3)
	ev.append_array(_run(m, 1.5, true, 2.0))
	assert_eq(ev, [] as Array[String], "xa thoáng qua (rẽ góc) không nhắc")


func test_far_reminds_holds_and_resumes() -> void:
	var m := EscortMonitor.new()
	var ev := _run(m, 3.5, true, 2.0)
	assert_eq(ev, ["remind"] as Array[String])
	assert_true(m.hold_robot(), "robot đứng chờ")
	ev = _run(m, 3.5, true, EscortMonitor.REMIND_EVERY_S + 0.2)
	assert_has(ev, "remind_again")
	ev = _run(m, 1.5, true, 0.2)
	assert_eq(ev, ["resumed"] as Array[String])
	assert_false(m.hold_robot())


func test_lost_by_distance_asks() -> void:
	var m := EscortMonitor.new()
	_run(m, 3.5, true, 2.0)
	var ev := _run(m, EscortMonitor.LOST_M + 0.5, true, 0.2)
	assert_eq(ev, ["ask_continue"] as Array[String])
	assert_eq(m.state, EscortMonitor.State.ASK)
	assert_eq(_run(m, 9.0, false, 30.0), [] as Array[String], "đang chờ trả lời thì không hỏi chồng")


func test_lost_by_unseen_asks() -> void:
	var m := EscortMonitor.new()
	_run(m, 3.0, false, 2.0)
	var ev := _run(m, 3.0, false, EscortMonitor.LOST_UNSEEN_S)
	assert_has(ev, "ask_continue")


func test_waiting_too_long_asks() -> void:
	var m := EscortMonitor.new()
	_run(m, 3.2, true, 2.0)
	var ev := _run(m, 3.2, true, EscortMonitor.ASK_AFTER_S + 0.5)
	assert_has(ev, "ask_continue")


func test_close_but_occluded_counts_as_following() -> void:
	var m := EscortMonitor.new()
	_run(m, 3.5, true, 2.0)
	var ev := _run(m, 1.4, false, 0.2)
	assert_eq(ev, ["resumed"] as Array[String], "đã quay lại gần, tia nhìn bị che vẫn là theo kịp")
	assert_eq(_run(m, 1.4, false, 20.0), [] as Array[String], "ở gần thì không đếm mất dấu")


func _asking() -> EscortMonitor:
	var m := EscortMonitor.new()
	_run(m, 3.5, true, 2.0)
	_run(m, 6.0, true, 0.2)
	assert_eq(m.state, EscortMonitor.State.ASK)
	return m


func test_answer_continue() -> void:
	var m := _asking()
	assert_eq(m.answer("tiep_tuc"), "continue")
	assert_false(m.hold_robot())


func test_answer_end() -> void:
	var m := _asking()
	assert_eq(m.answer("cam_on"), "end")
	assert_eq(m.state, EscortMonitor.State.ENDED)


func test_answer_rest_then_asks_again_or_resumes() -> void:
	var m := _asking()
	assert_eq(m.answer("dung_lai"), "rest")
	assert_true(m.hold_robot(), "bệnh nhân nghỉ ➔ robot chờ")
	assert_has(_run(m, 6.0, true, EscortMonitor.REST_WAIT_S + 0.2), "ask_continue")
	var m2 := _asking()
	m2.answer("dung_lai")
	assert_eq(_run(m2, 1.5, true, 0.2), ["resumed"] as Array[String], "nghỉ xong tự đi lại gần")


func test_unclear_answers_end_after_two() -> void:
	var m := _asking()
	assert_eq(m.answer(""), "ask_again")
	assert_eq(m.answer("unknown"), "end")


func test_ask_limit_gives_up() -> void:
	var m := EscortMonitor.new()
	var gave_up := false
	for i: int in EscortMonitor.MAX_ASKS + 1:
		_run(m, 3.5, true, 2.0)
		var ev := _run(m, 6.0, true, 0.2)
		if ev.has("give_up"):
			gave_up = true
			break
		m.answer("dung_lai")
		_run(m, 6.0, true, EscortMonitor.REST_WAIT_S + 0.2)
		if m.state == EscortMonitor.State.ENDED:
			gave_up = true
			break
		m.answer("dung_lai")
	assert_true(gave_up, "hỏi mãi không xong thì kết thúc ca")


# ---------- câu thoại ----------

func test_lines_file_has_all_keys_used_by_main() -> void:
	var a := RobotAnnouncer.new()
	a.load_lines(RobotAnnouncer.LINES_PATH)
	for key: String in ["start", "goal_ahead", "goal_left", "goal_right", "elevator_enter", "elevator_exit", "arrive",
			"remind", "remind_again", "ask_continue", "ask_again", "continue_ok", "stop_ok"]:
		var t: String = a.line_text(key, {"place": "Phòng 205", "floor": 2})
		assert_ne(t, "", "thiếu câu '%s'" % key)
		assert_false(t.contains("{"), key)
	a.free()


func test_format_line_and_place_names() -> void:
	var lines := {"arrive": "Đã tới {place} rồi ạ."}
	assert_eq(RobotAnnouncer.format_line(lines, "arrive", {"place": "Phòng 205"}), "Đã tới Phòng 205 rồi ạ.")
	assert_eq(RobotAnnouncer.format_line(lines, "arrive", {}), "", "thiếu tên nơi ➔ không nói câu cụt")
	assert_eq(RobotAnnouncer.format_line(lines, "nope", {}), "")
	assert_eq(RobotAnnouncer.place_name_of("205", {}), "Phòng 205")
	assert_eq(RobotAnnouncer.place_name_of("nha_thuoc", {"nha_thuoc": "Nhà thuốc"}), "Nhà thuốc")
	assert_eq(RobotAnnouncer.place_name_of("la", {}), "")


func test_announcer_subtitle_only_queues_and_finishes() -> void:
	var a := RobotAnnouncer.new()
	var busy := [true]
	a.is_busy = func() -> bool: return busy[0]
	add_child_autofree(a)
	watch_signals(a)
	assert_true(a.say("remind"))
	assert_false(a.is_speaking(), "đang hội thoại ➔ xếp hàng, chưa nói")
	busy[0] = false
	a._process(0.0)
	assert_true(a.is_speaking())
	assert_signal_emitted_with_parameters(a, "line_started", ["remind"])
	a._done_at_ms = 0
	a._process(0.0)
	assert_false(a.is_speaking())
	assert_signal_emitted_with_parameters(a, "line_finished", ["remind"])


func test_goal_side_line() -> void:
	assert_eq(Main.goal_side_line(0.1), "goal_ahead")
	assert_eq(Main.goal_side_line(deg_to_rad(60.0)), "goal_right")
	assert_eq(Main.goal_side_line(deg_to_rad(-60.0)), "goal_left")


# ---------- bệnh nhân dừng lại ----------

func test_patient_hold_keeps_patient_still_and_counts_down() -> void:
	var human: Human = (load("res://scenes/human.tscn") as PackedScene).instantiate()
	add_child_autofree(human)
	human.global_position = Vector3(0, 0, 3)
	RLEnvBridge.reset_escort_follow()
	human.set_meta("escort_hold_s", 1.0)
	var trail: Array[Vector3] = []
	var start: Vector3 = human.global_position
	for i: int in 5:
		RLEnvBridge.step_escorted_patient(human, Vector3(0, 0, -3), 0.1, trail, 0.0)
	assert_almost_eq(human.global_position.distance_to(start), 0.0, 1e-4, "đang nghỉ ➔ đứng yên")
	assert_almost_eq(float(human.get_meta("escort_hold_s")), 0.5, 1e-4)
