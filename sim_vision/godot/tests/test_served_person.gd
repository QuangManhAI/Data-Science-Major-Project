extends GutTest
## Người vừa được dẫn tới nơi đứng cạnh robot: box của họ thành vật cản, không phải "bệnh nhân cần giúp" mới.


func test_served_box_becomes_obstacle_others_kept() -> void:
	var served := {"class_id": 0, "class_name": "patient", "x1": 0.3, "x2": 0.7, "y1": 0.0, "y2": 1.0, "emb": PackedFloat32Array([1.0])}
	var other := {"class_id": 0, "class_name": "patient", "x1": 0.8, "x2": 0.9, "y1": 0.3, "y2": 0.7}
	var chair := {"class_id": 7, "class_name": "chair", "x1": 0.1, "x2": 0.2, "y1": 0.5, "y2": 0.9}
	var out: Array = Main.mark_served([served, other, chair], served)
	assert_eq(out.size(), 3)
	assert_eq(int(out[0]["class_id"]), 100, "người vừa giúp ➔ vật cản")
	assert_true(out[0]["served"])
	assert_false(out[0].has("emb"), "không gửi embedding cho policy")
	assert_eq(int(out[1]["class_id"]), 0, "bệnh nhân khác vẫn là bệnh nhân")
	assert_eq(int(out[2]["class_id"]), 7)


func test_no_served_box_returns_dets_unchanged() -> void:
	var d := {"class_id": 0, "class_name": "patient", "x1": 0.3, "x2": 0.7, "y1": 0.0, "y2": 1.0}
	var dets: Array = [d]
	assert_true(is_same(Main.mark_served(dets, {}), dets))


func test_served_box_equal_values_but_different_object_not_marked() -> void:
	var a := {"class_id": 0, "class_name": "patient", "x1": 0.3, "x2": 0.7, "y1": 0.0, "y2": 1.0}
	var b := a.duplicate()
	assert_eq(int(Main.mark_served([a], b)[0]["class_id"]), 0, "chỉ đúng box ReID khớp (cùng đối tượng)")
