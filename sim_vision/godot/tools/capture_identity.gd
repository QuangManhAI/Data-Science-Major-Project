extends SceneTree
## Chụp ảnh từ camera robot để đo nhận diện người (docs/UPGRADE_HRI_PLAN.md mục D2).
## Hành lang tầng 1 thật (room.tscn, đủ ánh sáng), camera giống robot: cao 1.2 m, FOV dọc 70°, 640×480, nhìn ngang.
## Mỗi người (10 ngoại hình NPC) ✕ khoảng cách ✕ góc quay người ✕ tư thế (đứng / đang đi) ➔ PNG + dòng JSON.
## Chạy (cần cửa sổ để render, không --headless):
##   godot --path sim_vision/godot --resolution 640x480 -s tools/capture_identity.gd -- <thư_mục_ra> [--faces]
## --faces: dán ảnh mặt người thật (npc_face.gd, texture từ identity/make_npc_faces.py) lên đầu từng NPC.

const IDENTITIES: Dictionary = {
	"patient_f02": "res://scenes/props/npc_patient_f02.tscn",
	"patient_f08": "res://scenes/props/npc_patient_f08.tscn",
	"patient_m02": "res://scenes/props/npc_patient_m02.tscn",
	"patient_m05": "res://scenes/props/npc_patient_m05.tscn",
	"patient_elder_f09": "res://scenes/props/npc_patient_elder_f09.tscn",
	"patient_elder_m03": "res://scenes/props/npc_patient_elder_m03.tscn",
	"patient_elder_m14": "res://scenes/props/npc_patient_elder_m14.tscn",
	"medical_doctor": "res://scenes/props/npc_medical_doctor.tscn",
	"medical_nurse": "res://scenes/props/npc_medical_nurse.tscn",
	"human_m06": "res://scenes/human.tscn",
}
const DISTANCES: Array[float] = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
const VIEWS_DEG: Array[float] = [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0]   # 0 = nhìn thẳng vào camera
const POSES: Array[String] = ["idle", "walk"]
const PERSON_POS: Vector3 = Vector3(-3.0, 0.0, 0.0)   # giữa hành lang tầng 1 (trục X)
const CAM_H: float = 1.2


func _init() -> void:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	var out_dir: String = args[0] if args.size() > 0 else "user://identity_capture"
	var with_faces: bool = "--faces" in args
	DirAccess.make_dir_recursive_absolute(out_dir)
	var root3d := Node3D.new()
	get_root().add_child(root3d)
	var room: Node3D = (load("res://scenes/room.tscn") as PackedScene).instantiate()
	root3d.add_child(room)
	var furn: Node = room.get_node_or_null(^"Furniture")
	if furn != null:
		for c: Node in furn.get_children():
			if c is HospitalNpc:
				c.queue_free()
	var cam := Camera3D.new()
	cam.fov = 70.0
	cam.near = 0.05
	root3d.add_child(cam)
	cam.current = true
	cam.rotation = Vector3(0.0, deg_to_rad(90.0), 0.0)   # nhìn về −X (dọc hành lang, về phía người)
	var meta := FileAccess.open(out_dir.path_join("frames.jsonl"), FileAccess.WRITE)
	var rng := RandomNumberGenerator.new()
	rng.seed = 7
	var n: int = 0
	await process_frame
	for ident: String in IDENTITIES:
		var person: Node3D = (load(IDENTITIES[ident]) as PackedScene).instantiate()
		if "can_speak" in person:
			person.set("can_speak", false)
		if "is_seated" in person:
			person.set("is_seated", false)
		if "walks" in person:
			person.set("walks", false)
		root3d.add_child(person)
		person.global_position = PERSON_POS
		person.set_physics_process(false)            # tắt máy trạng thái: chỉ giữ tư thế
		for i: int in 3:
			await process_frame
		if with_faces and NpcFace.attach(person, NpcFace.load_face(ident)) == null:
			push_error("không dán được mặt cho %s" % ident)
		var anim: AnimationPlayer = _anim_player(person)
		for pose: String in POSES:
			for d: float in DISTANCES:
				for view: float in VIEWS_DEG:
					person.rotation = Vector3(0.0, deg_to_rad(90.0 + view), 0.0)   # NPC nhìn theo +Z; 90° ➔ nhìn về +X (camera)
					_pose(anim, pose, rng)
					cam.global_position = PERSON_POS + Vector3(d, CAM_H, 0.0)
					await process_frame
					await process_frame
					var fname: String = "%s_%s_d%.1f_v%+04d.png" % [ident, pose, d, int(view)]
					get_root().get_texture().get_image().save_png(out_dir.path_join(fname))
					var c: Vector2 = cam.unproject_position(PERSON_POS + Vector3(0, 0.9, 0))
					var head: Vector2 = cam.unproject_position(PERSON_POS + Vector3(0, 1.6, 0))
					var feet: Vector2 = cam.unproject_position(PERSON_POS)
					meta.store_line(JSON.stringify({"file": fname, "identity": ident, "pose": pose, "distance_m": d,
							"view_deg": view, "center": [c.x, c.y], "head": [head.x, head.y], "feet": [feet.x, feet.y]}))
					n += 1
		person.queue_free()
		await process_frame
	meta.close()
	print("captured %d frames ➔ %s" % [n, out_dir])
	quit()


func _anim_player(n: Node) -> AnimationPlayer:
	var found: Array[Node] = n.find_children("*", "AnimationPlayer", true, false)
	return found[0] as AnimationPlayer if not found.is_empty() else null


func _pose(anim: AnimationPlayer, pose: String, rng: RandomNumberGenerator) -> void:
	if anim == null:
		return
	var name: String = ""
	for lib_anim: StringName in anim.get_animation_list():
		if String(lib_anim).ends_with(pose):
			name = String(lib_anim)
			break
	if name == "":
		return
	anim.play(name)
	anim.seek(rng.randf_range(0.0, anim.current_animation_length), true)
	anim.pause()
