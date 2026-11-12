from voice.escort_lines import escort_texts, load_lines, place_name, room_numbers


def test_place_name_matches_godot_rule():
    assert place_name("205", {}) == "Phòng 205"
    assert place_name("nha_thuoc", {"nha_thuoc": "Nhà thuốc"}) == "Nhà thuốc"
    assert place_name("x", {}) == ""


def test_room_numbers_from_godot_code():
    nums = room_numbers()
    assert {"101", "205", "301"} <= set(nums)


def test_escort_texts_expanded_without_placeholders():
    texts = escort_texts()
    assert all("{" not in t for t in texts)
    assert "Đã tới Phòng 205 rồi ạ." in texts
    assert "Đến tầng 2 rồi, mời anh chị đi tiếp theo tôi ạ." in texts
    assert load_lines()["lines"]["ask_continue"] in texts


def test_escort_texts_small_fixture():
    data = {"lines": {"a": "Tới {place}.", "b": "Tầng {floor}.", "c": "Xin chào."},
            "places": {"k": "Khoa K"}, "floors": [1, 2]}
    assert escort_texts(data, rooms=["101"]) == sorted(["Tới Khoa K.", "Tới Phòng 101.", "Tầng 1.", "Tầng 2.", "Xin chào."])
