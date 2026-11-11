extends GutTest


func _f(x1: float, y1: float, x2: float, y2: float) -> Dictionary:
	return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": 0.9}


func test_picks_face_near_top_centre_of_box() -> void:
	var det := {"x1": 0.3, "y1": 0.1, "x2": 0.6, "y2": 0.9}
	var own := _f(0.42, 0.12, 0.48, 0.2)        # đầu người trong box
	var behind := _f(0.31, 0.4, 0.35, 0.46)     # mặt người đứng sau, lọt vào nửa trên box
	assert_eq(Main.face_in_box([behind, own], det), own)


func test_no_face_in_lower_half_or_outside() -> void:
	var det := {"x1": 0.3, "y1": 0.1, "x2": 0.6, "y2": 0.9}
	assert_true(Main.face_in_box([_f(0.4, 0.7, 0.45, 0.75), _f(0.7, 0.1, 0.75, 0.2)], det).is_empty())
	assert_true(Main.face_in_box([_f(0.4, 0.1, 0.45, 0.2)], {}).is_empty())
