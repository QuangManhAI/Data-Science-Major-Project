extends GutTest

const HumanScene: PackedScene = preload("res://scenes/human.tscn")
const PatientF02Scene: PackedScene = preload("res://scenes/props/npc_patient_f02.tscn")
const MainScene: PackedScene = preload("res://scenes/main.tscn")


func _solid_tex(col: Color, sz: int = 32) -> ImageTexture:
	var img := Image.create(sz, sz, false, Image.FORMAT_RGB8)
	img.fill(col)
	return ImageTexture.create_from_image(img)


func test_attach_turns_off_box_head_and_adds_front_back_shirt_plates() -> void:
	var h := HumanScene.instantiate() as Human
	add_child_autofree(h)
	await wait_process_frames(2)

	var tex := _solid_tex(Color(0.1, 0.9, 0.2))
	var front_mi: MeshInstance3D = NpcFace.attach(h, tex)
	assert_not_null(front_mi)
	assert_eq(front_mi.name, &"ShirtFrontFace")

	var skel := h.get_node("Model/Skeleton3D") as Skeleton3D
	assert_null(h.get_node_or_null(^"BoxHead"), "tắt đầu hộp vuông")
	assert_null(h.get_node_or_null(^"FaceMask"), "tắt mặt phẳng trước đầu")
	assert_null(skel.get_node_or_null(^"FaceAttachment"))

	var back_mi := h.get_node_or_null(^"ShirtBackFace") as MeshInstance3D
	assert_not_null(back_mi, "phải gắn tấm mặt sau áo")
	var att := skel.get_node_or_null(^"ShirtFaceAttachment") as BoneAttachment3D
	assert_not_null(att)
	assert_eq(att.bone_name, &"Bip01 Spine2")

	var f_mat := (front_mi.mesh as QuadMesh).material as StandardMaterial3D
	var b_mat := (back_mi.mesh as QuadMesh).material as StandardMaterial3D
	assert_eq(f_mat.albedo_texture, tex, "mặt trước áo dùng ảnh gốc nguyên bản")
	assert_eq(b_mat.albedo_texture, tex, "mặt sau áo mặc định dùng chính ảnh mặt trước")
	assert_gt(front_mi.global_basis.z.dot(h.global_basis.z), 0.9, "tấm trước hướng ra phía trước ngực")
	assert_gt(back_mi.global_basis.z.dot(-h.global_basis.z), 0.9, "tấm sau hướng ra phía sau lưng")
	assert_gt(front_mi.global_position.z, back_mi.global_position.z)


func test_attach_distinct_back_texture_supported() -> void:
	var h := HumanScene.instantiate() as Human
	add_child_autofree(h)
	await wait_process_frames(2)

	var front_tex := _solid_tex(Color(0.1, 0.9, 0.2))
	var back_tex := _solid_tex(Color(0.1, 0.2, 0.95))
	var front_mi: MeshInstance3D = NpcFace.attach(h, front_tex, back_tex)
	assert_not_null(front_mi)
	var back_mi := h.get_node_or_null(^"ShirtBackFace") as MeshInstance3D
	assert_eq(((front_mi.mesh as QuadMesh).material as StandardMaterial3D).albedo_texture, front_tex)
	assert_eq(((back_mi.mesh as QuadMesh).material as StandardMaterial3D).albedo_texture, back_tex)


func test_attach_replaces_cleanly_and_removes_legacy_box_head() -> void:
	var h := HumanScene.instantiate() as Human
	add_child_autofree(h)
	await wait_process_frames(2)

	for legacy_name: String in ["BoxHead", "FaceMask"]:
		var legacy := MeshInstance3D.new()
		legacy.name = legacy_name
		h.add_child(legacy)

	var tex1 := _solid_tex(Color(0.1, 0.9, 0.2))
	var tex2 := _solid_tex(Color(0.95, 0.1, 0.9))
	NpcFace.attach(h, tex1)
	var front_mi: MeshInstance3D = NpcFace.attach(h, tex2)
	await wait_process_frames(1)

	assert_null(h.get_node_or_null(^"BoxHead"), "dọn BoxHead cũ nếu có")
	assert_null(h.get_node_or_null(^"FaceMask"), "dọn FaceMask cũ nếu có")
	assert_eq(((front_mi.mesh as QuadMesh).material as StandardMaterial3D).albedo_texture, tex2)
	var back_mi := h.get_node_or_null(^"ShirtBackFace") as MeshInstance3D
	assert_eq(((back_mi.mesh as QuadMesh).material as StandardMaterial3D).albedo_texture, tex2)


func test_attach_works_on_seated_and_rotated_hospital_npc() -> void:
	var npc := PatientF02Scene.instantiate() as HospitalNpc
	npc.can_speak = false
	npc.rotation.y = deg_to_rad(135.0)
	add_child_autofree(npc)
	await wait_process_frames(2)

	var tex := _solid_tex(Color(0.95, 0.15, 0.2))
	var front_mi: MeshInstance3D = NpcFace.attach(npc, tex)
	assert_not_null(front_mi)
	assert_null(npc.get_node_or_null(^"BoxHead"))
	var back_mi := npc.get_node_or_null(^"ShirtBackFace") as MeshInstance3D
	assert_not_null(back_mi)
	assert_gt(front_mi.global_basis.z.dot(npc.global_basis.z), 0.8)
	assert_gt(back_mi.global_basis.z.dot(-npc.global_basis.z), 0.8)


func test_main_apply_face_paste_has_no_box_head() -> void:
	if NpcFace.face_path("patient_m02") == "":
		pass_test("chưa có assets/textures/npc_faces trên máy")
		return
	var main := MainScene.instantiate() as Main
	main.seed = 1
	for n: String in ["RunLogger", "DialogueLogger"]:
		var node: Node = main.get_node_or_null(NodePath(n))
		if node != null:
			node.free()
	add_child_autofree(main)
	main.vision.set_process(false)
	await wait_process_frames(4)
	assert_null(main.human.get_node_or_null(^"BoxHead"))
	assert_null(main.human.get_node_or_null(^"FaceMask"))
	assert_not_null(main.human.get_node_or_null(^"ShirtFrontFace"), "Main đã dán mặt lên trước áo Human")
	assert_not_null(main.human.get_node_or_null(^"ShirtBackFace"), "Main đã dán mặt lên sau áo Human")
