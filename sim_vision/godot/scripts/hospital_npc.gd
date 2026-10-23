class_name HospitalNpc
extends StaticBody3D
## NPC tĩnh/ngồi chờ trong bệnh viện (Nhân viên y tế: Bác sĩ, Điều dưỡng; Bệnh nhân ngồi chờ).
## Kế thừa StaticBody3D kèm CollisionShape3D (BoxShape3D) để Room.get_furniture_footprints()
## tự động tránh spawn robot hoặc $Human chồng lên.

@export var tex_dir: String = "res://assets/models/medical_f01/textures"
@export var anim_path: String = "res://assets/models/human_m06/anims/f_idle.res"
@export var anim_offset: float = 0.0
@export var is_seated: bool = false
@export var staff_role: String = ""  # "nurse" | "doctor" | "patient"
@export var badge_text: String = ""
@export var has_cane: bool = false

var _anim_player: AnimationPlayer


func _ready() -> void:
	var model := get_node_or_null(^"Model") as Node3D
	if model == null:
		return
	if is_seated:
		model.position = Vector3(0.0, 0.0, 0.06)

	var mesh_node := _find_mesh(model)
	if mesh_node != null and mesh_node.mesh != null:
		var mat_body := _make_part_mat("body")
		var mat_head := _make_part_mat("head")
		var mat_opacity := _make_opacity_mat()
		var fallback: Array[Material] = [mat_body, mat_head, mat_opacity]
		for s: int in mesh_node.mesh.get_surface_count():
			var orig_mat: Material = mesh_node.mesh.surface_get_material(s)
			var mname: String = orig_mat.resource_name.to_lower() if orig_mat != null else ""
			if "opacity" in mname:
				mesh_node.set_surface_override_material(s, mat_opacity)
			elif "head" in mname:
				mesh_node.set_surface_override_material(s, mat_head)
			elif "body" in mname:
				mesh_node.set_surface_override_material(s, mat_body)
			elif s < fallback.size():
				mesh_node.set_surface_override_material(s, fallback[s])

	var skel := model.get_node_or_null(^"Skeleton3D") as Skeleton3D
	if skel != null and staff_role == "nurse":
		_attach_nurse_cap(skel)

	if badge_text != "":
		_create_role_badge()

	if has_cane:
		_create_walking_cane()

	if anim_path != "" and ResourceLoader.exists(anim_path):
		_anim_player = AnimationPlayer.new()
		_anim_player.name = "AnimationPlayer"
		model.add_child(_anim_player)
		var lib := AnimationLibrary.new()
		var anim_res: Animation = load(anim_path)
		lib.add_animation(&"loop", anim_res)
		_anim_player.add_animation_library(&"npc", lib)
		_anim_player.play(&"npc/loop")
		if anim_offset > 0.0:
			_anim_player.advance(anim_offset)


func _make_part_mat(part: String) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	var c_path: String = "%s/%s_color.jpg" % [tex_dir, part]
	var n_path: String = "%s/%s_normal.png" % [tex_dir, part]
	var r_path: String = "%s/%s_roughness.jpg" % [tex_dir, part]
	if ResourceLoader.exists(c_path):
		mat.albedo_texture = load(c_path)
	if ResourceLoader.exists(r_path):
		mat.roughness_texture = load(r_path)
	if ResourceLoader.exists(n_path):
		mat.normal_enabled = true
		mat.normal_texture = load(n_path)
	return mat


func _make_opacity_mat() -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	var o_path: String = "%s/opacity_color.png" % tex_dir
	if ResourceLoader.exists(o_path):
		mat.albedo_texture = load(o_path)
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_HASH
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	mat.roughness = 0.7
	return mat


static func _find_mesh(root: Node) -> MeshInstance3D:
	if root is MeshInstance3D:
		return root as MeshInstance3D
	for c: Node in root.get_children():
		var found := _find_mesh(c)
		if found != null:
			return found
	return null


func _attach_nurse_cap(skel: Skeleton3D) -> void:
	var head_idx: int = skel.find_bone("Bip01 Head")
	if head_idx == -1:
		return
	var attach := BoneAttachment3D.new()
	attach.name = "NurseCapAttachment"
	attach.bone_name = "Bip01 Head"
	skel.add_child(attach)

	var cap_white := StandardMaterial3D.new()
	cap_white.albedo_color = Color(0.96, 0.97, 0.99, 1.0)
	cap_white.roughness = 0.4

	var cap_blue := StandardMaterial3D.new()
	cap_blue.albedo_color = Color(0.09, 0.38, 0.68, 1.0)
	cap_blue.roughness = 0.35

	var cap_body := MeshInstance3D.new()
	var cyl := CylinderMesh.new()
	cyl.top_radius = 0.078
	cyl.bottom_radius = 0.072
	cyl.height = 0.055
	cyl.material = cap_white
	cap_body.mesh = cyl
	cap_body.position = Vector3(0.155, -0.015, 0.0)
	cap_body.rotation = Vector3(0.0, 0.0, -PI / 2.0)
	attach.add_child(cap_body)

	var stripe := MeshInstance3D.new()
	var stripe_cyl := CylinderMesh.new()
	stripe_cyl.top_radius = 0.08
	stripe_cyl.bottom_radius = 0.079
	stripe_cyl.height = 0.012
	stripe_cyl.material = cap_blue
	stripe.mesh = stripe_cyl
	stripe.position = Vector3(0.162, -0.015, 0.0)
	stripe.rotation = Vector3(0.0, 0.0, -PI / 2.0)
	attach.add_child(stripe)


func _create_role_badge() -> void:
	var lbl := Label3D.new()
	lbl.name = "RoleBadge"
	lbl.position = Vector3(0.0, 1.84 if not is_seated else 1.48, 0.0)
	lbl.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	lbl.pixel_size = 0.0014
	lbl.font_size = 26
	lbl.outline_size = 5
	lbl.text = badge_text
	if staff_role == "doctor":
		lbl.modulate = Color(0.55, 0.92, 1.0, 1.0)
	elif staff_role == "nurse":
		lbl.modulate = Color(0.62, 0.98, 0.78, 1.0)
	else:
		lbl.modulate = Color(0.95, 0.95, 0.9, 1.0)
	add_child(lbl)


func _create_walking_cane() -> void:
	var cane := Node3D.new()
	cane.name = "WalkingCane"
	# Position resting slightly in front and to the right hand side of the patient
	cane.position = Vector3(-0.18, 0.0, 0.56) if is_seated else Vector3(-0.24, 0.0, 0.22)
	cane.rotation = Vector3(0.08, 0.0, 0.0) if is_seated else Vector3(0.0, 0.0, 0.0)
	add_child(cane)

	var wood_mat := StandardMaterial3D.new()
	wood_mat.albedo_color = Color(0.36, 0.2, 0.09, 1.0)
	wood_mat.roughness = 0.35

	var brass_mat := StandardMaterial3D.new()
	brass_mat.albedo_color = Color(0.78, 0.64, 0.24, 1.0)
	brass_mat.metallic = 0.75
	brass_mat.roughness = 0.25

	var rubber_mat := StandardMaterial3D.new()
	rubber_mat.albedo_color = Color(0.14, 0.14, 0.15, 1.0)
	rubber_mat.roughness = 0.85

	var shaft := MeshInstance3D.new()
	var shaft_cyl := CylinderMesh.new()
	shaft_cyl.top_radius = 0.013
	shaft_cyl.bottom_radius = 0.011
	shaft_cyl.height = 0.78
	shaft_cyl.material = wood_mat
	shaft.mesh = shaft_cyl
	shaft.position = Vector3(0.0, 0.40, 0.0)
	cane.add_child(shaft)

	var tip := MeshInstance3D.new()
	var tip_cyl := CylinderMesh.new()
	tip_cyl.top_radius = 0.015
	tip_cyl.bottom_radius = 0.018
	tip_cyl.height = 0.045
	tip_cyl.material = rubber_mat
	tip.mesh = tip_cyl
	tip.position = Vector3(0.0, 0.0225, 0.0)
	cane.add_child(tip)

	var collar := MeshInstance3D.new()
	var collar_cyl := CylinderMesh.new()
	collar_cyl.top_radius = 0.0145
	collar_cyl.bottom_radius = 0.0145
	collar_cyl.height = 0.022
	collar_cyl.material = brass_mat
	collar.mesh = collar_cyl
	collar.position = Vector3(0.0, 0.78, 0.0)
	cane.add_child(collar)

	var handle := MeshInstance3D.new()
	var handle_box := BoxMesh.new()
	handle_box.size = Vector3(0.028, 0.026, 0.13)
	handle_box.material = wood_mat
	handle.mesh = handle_box
	handle.position = Vector3(0.0, 0.80, -0.04)
	cane.add_child(handle)
