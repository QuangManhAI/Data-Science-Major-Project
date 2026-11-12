extends GutTest
## docs/UPGRADE_HRI_PLAN.md D3–D4: ghi nhớ / bám đúng bệnh nhân bằng embedding.


## Vector đơn vị 512-D: thành phần `i` = cos góc, `j` = sin góc ➔ cosine với e(i) bằng cos(góc).
func _emb(i: int, j: int = -1, cos_to_i: float = 1.0) -> PackedFloat32Array:
	var v := PackedFloat32Array()
	v.resize(512)
	v[i] = cos_to_i
	if j >= 0:
		v[j] = sqrt(maxf(0.0, 1.0 - cos_to_i * cos_to_i))
	return v


func _det(cx: float, h: float, emb: PackedFloat32Array = PackedFloat32Array(), face: bool = false, cid: int = 0) -> Dictionary:
	var d: Dictionary = {"x1": cx - 0.08, "x2": cx + 0.08, "y1": 0.5 - h / 2.0, "y2": 0.5 + h / 2.0, "conf": 0.9,
			"class_id": cid, "class_name": "patient" if cid == 0 else "doctor"}
	if not emb.is_empty():
		d["emb"] = emb
	d["face"] = face
	return d


func _enrolled() -> PatientIdentity:
	var p := PatientIdentity.new()
	assert_true(p.enroll_from([_det(0.5, 0.6, _emb(0), true)], 0.0))
	return p


func test_enroll_needs_face_center_and_patient_class() -> void:
	var p := PatientIdentity.new()
	assert_false(p.enroll_from([_det(0.5, 0.6, _emb(0), false)], 0.0), "không thấy mặt")
	assert_false(p.enroll_from([_det(0.95, 0.6, _emb(0), true)], 0.0), "lệch mép khung, không phải người đang nói")
	assert_false(p.enroll_from([_det(0.5, 0.6, _emb(0), true, 1)], 0.0), "bác sĩ")
	assert_false(p.enroll_from([_det(0.5, 0.6, PackedFloat32Array(), true)], 0.0), "frame không có embedding")
	assert_true(p.enroll_from([_det(0.45, 0.3, _emb(1), true), _det(0.55, 0.6, _emb(0), true)], 0.0))
	assert_almost_eq(p.score(_emb(0)), 1.0, 1e-5, "chọn người to nhất (gần robot nhất)")


func test_enroll_rate_and_cap() -> void:
	var p := PatientIdentity.new()
	var d := [_det(0.5, 0.6, _emb(0), true)]
	assert_true(p.enroll_from(d, 0.0))
	assert_false(p.enroll_from(d, 0.1), "quá dày")
	assert_true(p.enroll_from(d, 0.4))
	assert_true(p.enroll_from(d, 0.8))
	assert_false(p.enroll_from(d, 1.2), "đủ MAX_ENROLL")
	assert_eq(p.templates.size(), PatientIdentity.MAX_ENROLL)


func test_match_picks_enrolled_person_not_most_confident() -> void:
	var p := _enrolled()
	var stranger := _det(0.3, 0.7, _emb(1))
	stranger["conf"] = 0.99
	var mine := _det(0.7, 0.4, _emb(0, 2, 0.9))
	mine["conf"] = 0.6
	var r := p.match([stranger, mine], 5.0)
	assert_true(is_same(r["det"], mine))
	assert_eq(r["how"], "appearance")


func test_no_match_when_only_strangers() -> void:
	var p := _enrolled()
	assert_eq(p.match([_det(0.5, 0.5, _emb(1)), _det(0.2, 0.5, _emb(2))], 5.0), {})


func test_ambiguous_pair_rejected_by_margin() -> void:
	var p := _enrolled()
	var a := _det(0.3, 0.5, _emb(0, 1, 0.80))
	var b := _det(0.8, 0.5, _emb(0, 2, 0.78))
	assert_eq(p.match([a, b], 5.0), {}, "hai người giống nhau quá ➔ không đoán")


func test_continuity_on_frames_without_embedding_then_expires() -> void:
	var p := _enrolled()
	assert_false(p.match([_det(0.52, 0.6, _emb(0, 1, 0.9))], 1.0).is_empty())
	var r := p.match([_det(0.55, 0.62), _det(0.9, 0.6)], 1.1)   # frame không có emb
	assert_eq(r["how"], "continuity")
	assert_almost_eq(float(r["det"]["x1"]), 0.47, 1e-5)
	assert_eq(p.match([_det(0.56, 0.62)], 1.1 + PatientIdentity.CONT_MAX_S + 0.1), {}, "mất dấu lâu ➔ không bám mù")


func test_continuity_rejects_stranger_like_score() -> void:
	var p := _enrolled()
	p.match([_det(0.5, 0.6, _emb(0))], 1.0)
	assert_eq(p.match([_det(0.52, 0.6, _emb(0, 4, 0.55))], 1.1), {}, "điểm kiểu người lạ (0.55) ➔ không bám dù đứng đúng chỗ")


func test_continuity_rejects_jump_and_low_score() -> void:
	var p := _enrolled()
	p.match([_det(0.5, 0.6, _emb(0))], 1.0)
	assert_eq(p.match([_det(0.8, 0.6)], 1.1), {}, "nhảy xa")
	p.match([_det(0.5, 0.6, _emb(0))], 1.2)
	assert_eq(p.match([_det(0.52, 0.6, _emb(3))], 1.3), {}, "cùng chỗ nhưng là người khác hẳn")


func test_learns_back_view_while_tracking_alone() -> void:
	var p := _enrolled()
	var back := _emb(0, 5, 0.73)         # lưng: giống mẫu trước vừa đủ để bám liên tục, chưa đủ ACCEPT_SIM
	assert_eq(p.match([_det(0.5, 0.6, back)], 1.0).get("how", ""), "", "chưa bám ➔ chưa nhận")
	p.match([_det(0.5, 0.6, _emb(0))], 2.0)                       # vẫn đang nhìn mặt trước
	var r := p.match([_det(0.55, 0.6, back)], 2.3)   # xoay người, bám liên tục (mẫu cuối thêm lúc 0 s)
	assert_eq(r["how"], "continuity")
	assert_eq(p.templates.size(), 2, "học thêm mẫu lưng")
	var later := p.match([_det(0.2, 0.4, back), _det(0.8, 0.5, _emb(7))], 30.0)   # mất dấu lâu, gặp lại từ sau lưng
	assert_eq(later.get("how", ""), "appearance")
	assert_almost_eq(float(later["det"]["x1"]), 0.12, 1e-5)


func test_does_not_learn_when_other_patients_in_view() -> void:
	var p := _enrolled()
	p.match([_det(0.5, 0.6, _emb(0))], 2.0)
	p.match([_det(0.55, 0.6, _emb(0, 5, 0.73)), _det(0.1, 0.5, _emb(9))], 2.3)
	assert_eq(p.templates.size(), 1, "có người khác trong khung ➔ không học mẫu mơ hồ")


func test_continuity_is_per_camera() -> void:
	var p := _enrolled()
	p.match([_det(0.5, 0.6, _emb(0))], 1.0, "front")
	assert_eq(p.match([_det(0.52, 0.6)], 1.1, "rear"), {}, "camera sau chưa từng thấy ➔ không bám theo vị trí của camera trước")
	assert_eq(p.match([_det(0.3, 0.5, _emb(0, 3, 0.9))], 1.2, "rear").get("how", ""), "appearance", "nhận ra bằng mẫu dùng chung")
	assert_eq(p.match([_det(0.32, 0.5)], 1.3, "rear").get("how", ""), "continuity")


func test_close_range_clipped_box_recognised() -> void:
	var p := _enrolled()
	p.match([_det(0.5, 0.6, _emb(0))], 1.0)                         # vừa thấy ở gần
	var clipped := _det(0.5, 0.98, _emb(0, 6, 0.55))                  # sát robot: box cắt, điểm thấp
	assert_eq(p.match([clipped], 4.0).get("how", ""), "close")
	assert_eq(p.match([clipped, _det(0.4, 0.95, _emb(0, 7, 0.55))], 4.5), {}, "hai box to ➔ không đoán")
	assert_eq(p.match([_det(0.5, 0.98, _emb(8))], 5.0), {}, "người lạ sát robot (điểm 0) ➔ không nhận")
	var q := _enrolled()
	q.match([_det(0.5, 0.6, _emb(0))], 1.0)
	assert_eq(q.match([clipped], 1.0 + PatientIdentity.CLOSE_RECENT_S + 1.0), {}, "lâu rồi không thấy ➔ không đoán")


func test_no_duplicate_templates() -> void:
	var p := _enrolled()
	p.match([_det(0.5, 0.6, _emb(0, 3, 0.95))], 2.0)
	assert_eq(p.templates.size(), 1, "gần trùng mẫu đã có ➔ không thêm")


func test_bootstrap_enroll_without_face() -> void:
	var p := PatientIdentity.new()
	assert_false(p.enroll_from([_det(0.5, 0.6, _emb(0), false)], 0.0))
	assert_true(p.enroll_from([_det(0.5, 0.6, _emb(0), false)], 0.0, false))


func test_template_cap_keeps_enrolled() -> void:
	var p := _enrolled()
	var t: float = 1.0
	for k: int in 20:
		p.match([_det(0.5, 0.6, _emb(0, 10 + k, 0.88))], t)
		t += PatientIdentity.ADD_EVERY_S + 0.01
	assert_eq(p.templates.size(), PatientIdentity.MAX_TEMPLATES)
	assert_almost_eq(PatientIdentity.cosine(p.templates[0], _emb(0)), 1.0, 1e-5, "mẫu lúc nói chuyện vẫn còn")


func test_enroll_picks_box_in_speaker_direction() -> void:
	var p := PatientIdentity.new()
	var near_center := _det(0.5, 0.7, _emb(1), true)     # người khác, to, giữa khung
	var speaker := _det(0.8, 0.5, _emb(0), true)         # người đang nói, lệch phải
	assert_true(p.enroll_from([near_center, speaker], 0.0, true, 0.78))
	assert_almost_eq(p.score(_emb(0)), 1.0, 1e-5, "ghi nhớ người ở hướng giọng nói, không phải người to nhất")
	var q := PatientIdentity.new()
	assert_false(q.enroll_from([near_center], 0.0, true, 0.78), "hướng giọng nói không có ai thấy mặt ➔ chưa ghi nhớ")
	assert_eq(q.last_reject, "not_speaker_dir")
