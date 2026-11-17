class_name NpcFace
extends RefCounted
## Dán ảnh mặt người thật 2D lên mặt trước và mặt sau áo NPC (tắt khối đầu vuông `BoxHead`).
## Dùng hai tấm phẳng `QuadMesh` (`ShirtFrontFace`, `ShirtBackFace`) gắn vào xương ngực `"Bip01 Spine2"`
## qua `BoneAttachment3D` + `RemoteTransform3D` ➔ ảnh mặt giữ nguyên pixel gốc, phẳng tuyệt đối, không bị
## ve áo / nếp gấp / đường may / normal map của mesh 3D cắt qua gây nhiễu CNN (YuNet / OSNet).
## Nếu không truyền `back_tex`, mặc định dùng chính ảnh mặt trước `tex` cho cả mặt trước và mặt sau áo.

const FACES_DIR: String = "res://assets/textures/npc_faces"
const PLATE_CLEARANCE_M: float = 0.018   # cách bề mặt ngoài cùng của ngực / lưng 1.8 cm để không bị ve áo / bụng cắt lúc cúi ngồi
const MIN_QUAD_SIZE_M: float = 0.19
const MAX_QUAD_SIZE_M: float = 0.24
const TORSO_BONES: Array[String] = [
	"Bip01 Pelvis", "Bip01 Spine", "Bip01 Spine1", "Bip01 Spine2",
	"Bip01 Neck", "Bip01 L Clavicle", "Bip01 R Clavicle",
]


## Đường dẫn texture mặt cho một NPC (theo tên ngoại hình, vd. "patient_f02"); không có file ➔ "".
static func face_path(look: String) -> String:
	var p: String = "%s/%s.png" % [FACES_DIR, look]
	return p if FileAccess.file_exists(ProjectSettings.globalize_path(p)) else ""


static func load_face(look: String) -> Texture2D:
	var p: String = NpcFace.face_path(look)
	if p == "":
		return null
	var img := Image.load_from_file(ProjectSettings.globalize_path(p))
	if img == null:
		return null
	img.generate_mipmaps()   # không có mipmap ➔ mặt nhỏ trên màn hình (xa camera) bị răng cưa / nhấp nháy
	return ImageTexture.create_from_image(img)


## Gắn hai tấm ảnh 2D phẳng `tex` (trước áo) và `back_tex` (sau áo; null ➔ dùng `tex`) bám xương ngực NPC.
## Tắt đầu vuông (`BoxHead`/`FaceMask`) để giữ nguyên đầu 3D gốc của NPC. Trả `ShirtFrontFace` hoặc null nếu lỗi.
static func attach(npc: Node3D, tex: Texture2D, back_tex: Texture2D = null) -> MeshInstance3D:
	if npc == null or tex == null:
		return null
	var mesh_node := _find_body_mesh(npc)
	var sk := _find_skeleton(npc, mesh_node)
	if sk == null:
		return null

	var sp1_i: int = sk.find_bone("Bip01 Spine1")
	var sp2_i: int = sk.find_bone("Bip01 Spine2")
	var neck_i: int = sk.find_bone("Bip01 Neck")
	if sp1_i < 0 or sp2_i < 0 or neck_i < 0:
		return null

	_remove_existing(npc, sk)

	var sk_rest_xf: Transform3D = _local_xform(npc, sk)
	var sp1_rest: Vector3 = (sk_rest_xf * sk.get_bone_global_rest(sp1_i)).origin
	var sp2_rest_xf: Transform3D = sk_rest_xf * sk.get_bone_global_rest(sp2_i)
	var neck_rest: Vector3 = (sk_rest_xf * sk.get_bone_global_rest(neck_i)).origin

	var frame: Dictionary = _torso_rest_axes(sk, sk_rest_xf, sp2_rest_xf.origin)
	var side: Vector3 = frame["side"]
	var up: Vector3 = frame["up"]
	var fwd: Vector3 = frame["fwd"]

	var torso_h: float = (neck_rest - sp1_rest).dot(up)
	var quad_sz: float = clampf(torso_h * 0.64, MIN_QUAD_SIZE_M, MAX_QUAD_SIZE_M)
	var y_top_rel: float = (neck_rest - sp2_rest_xf.origin).dot(up) * 0.36
	var y_mid_rel: float = y_top_rel - 0.5 * quad_sz

	var rest_depth: Vector2 = _measure_chest_depth(
		npc, mesh_node, sk, sp2_rest_xf, side, up, fwd,
		0.5 * quad_sz, y_top_rel - quad_sz, y_top_rel, true
	)
	var sp2_pose_local: Transform3D = sk_rest_xf * sk.get_bone_global_pose(sp2_i)
	var cur_basis: Basis = sp2_pose_local.basis * (sp2_rest_xf.basis.inverse() * Basis(side, up, fwd))
	var pose_depth: Vector2 = _measure_chest_depth(
		npc, mesh_node, sk, sp2_pose_local,
		cur_basis.x.normalized(), cur_basis.y.normalized(), cur_basis.z.normalized(),
		0.5 * quad_sz, y_top_rel - quad_sz, y_top_rel, false
	)
	var depth := Vector2(maxf(rest_depth.x, pose_depth.x), minf(rest_depth.y, pose_depth.y))

	var front_pos_rest: Vector3 = sp2_rest_xf.origin + up * y_mid_rel + fwd * (depth.x + PLATE_CLEARANCE_M)
	var back_pos_rest: Vector3 = sp2_rest_xf.origin + up * y_mid_rel + fwd * (depth.y - PLATE_CLEARANCE_M)

	var sp2_rest_inv: Transform3D = sp2_rest_xf.affine_inverse()
	var front_rel_sp2: Transform3D = sp2_rest_inv * Transform3D(Basis(side, up, fwd), front_pos_rest)
	var back_rel_sp2: Transform3D = sp2_rest_inv * Transform3D(Basis(-side, up, -fwd), back_pos_rest)

	var sp2_now: Transform3D = sk.global_transform * sk.get_bone_global_pose(sp2_i)
	var front_xf: Transform3D = sp2_now * front_rel_sp2
	var back_xf: Transform3D = sp2_now * back_rel_sp2

	var front_mi := _make_plate(&"ShirtFrontFace", tex, quad_sz)
	var back_src: Texture2D = back_tex if back_tex != null else tex
	var back_mi := _make_plate(&"ShirtBackFace", back_src, quad_sz)
	npc.add_child(front_mi)
	npc.add_child(back_mi)

	var att := BoneAttachment3D.new()
	att.name = &"ShirtFaceAttachment"
	att.bone_name = &"Bip01 Spine2"
	sk.add_child(att)
	att.global_transform = sp2_now

	for pair: Array in [
		[front_mi, front_xf, &"ShirtFrontFollow"],
		[back_mi, back_xf, &"ShirtBackFollow"],
	]:
		var target := pair[0] as MeshInstance3D
		var xf: Transform3D = pair[1]
		var rt := RemoteTransform3D.new()
		rt.name = pair[2]
		rt.update_scale = false
		att.add_child(rt)
		rt.global_transform = xf
		rt.remote_path = rt.get_path_to(target)
		target.global_transform = xf

	return front_mi


static func _make_plate(node_name: StringName, tex: Texture2D, size_m: float) -> MeshInstance3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_texture = tex
	mat.transparency = BaseMaterial3D.TRANSPARENCY_DISABLED
	mat.roughness = 0.9
	mat.specular_mode = BaseMaterial3D.SPECULAR_DISABLED
	mat.cull_mode = BaseMaterial3D.CULL_BACK
	mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	var quad := QuadMesh.new()
	quad.size = Vector2(size_m, size_m)
	quad.material = mat
	var mi := MeshInstance3D.new()
	mi.name = node_name
	mi.mesh = quad
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	return mi


static func _remove_existing(npc: Node3D, sk: Skeleton3D) -> void:
	for node_name: StringName in [&"BoxHead", &"FaceMask", &"ShirtFrontFace", &"ShirtBackFace"]:
		var node: Node = npc.get_node_or_null(NodePath(node_name))
		if node != null:
			npc.remove_child(node)
			node.queue_free()
	if sk != null:
		for att_name: StringName in [&"FaceAttachment", &"ShirtFaceAttachment"]:
			var att: Node = sk.get_node_or_null(NodePath(att_name))
			if att != null:
				sk.remove_child(att)
				att.queue_free()


static func _find_body_mesh(npc: Node3D) -> MeshInstance3D:
	var model: Node = npc.get_node_or_null(^"Model")
	var search_root: Node = model if model != null else npc
	for n: Node in search_root.find_children("*", "MeshInstance3D", true, false):
		var mi := n as MeshInstance3D
		if mi != null and mi.mesh != null and not (mi.name in [&"BoxHead", &"FaceMask", &"ShirtFrontFace", &"ShirtBackFace"]):
			return mi
	return null


static func _find_skeleton(npc: Node3D, mesh_node: MeshInstance3D) -> Skeleton3D:
	if mesh_node != null and mesh_node.skeleton != NodePath():
		var sk := mesh_node.get_node_or_null(mesh_node.skeleton) as Skeleton3D
		if sk != null:
			return sk
	var found: Array[Node] = npc.find_children("*", "Skeleton3D", true, false)
	return found[0] as Skeleton3D if not found.is_empty() else null


static func _local_xform(root: Node3D, node: Node3D) -> Transform3D:
	if root.is_inside_tree() and node.is_inside_tree():
		return root.global_transform.affine_inverse() * node.global_transform
	var xf := Transform3D.IDENTITY
	var cur: Node = node
	while cur != null and cur != root:
		if cur is Node3D:
			xf = (cur as Node3D).transform * xf
		cur = cur.get_parent()
	return xf


static func _torso_rest_axes(sk: Skeleton3D, sk_rest_xf: Transform3D, sp2_pos: Vector3) -> Dictionary:
	var side := Vector3.RIGHT
	var le_i: int = sk.find_bone("Bip01 LEye")
	var re_i: int = sk.find_bone("Bip01 REye")
	if le_i >= 0 and re_i >= 0:
		var pl: Vector3 = (sk_rest_xf * sk.get_bone_global_rest(le_i)).origin
		var pr: Vector3 = (sk_rest_xf * sk.get_bone_global_rest(re_i)).origin
		var d: Vector3 = Vector3(pl.x - pr.x, 0.0, pl.z - pr.z)
		if d.length() > 1e-4:
			side = d.normalized()
	else:
		var lc_i: int = sk.find_bone("Bip01 L UpperArm")
		var rc_i: int = sk.find_bone("Bip01 R UpperArm")
		if lc_i >= 0 and rc_i >= 0:
			var pl: Vector3 = (sk_rest_xf * sk.get_bone_global_rest(lc_i)).origin
			var pr: Vector3 = (sk_rest_xf * sk.get_bone_global_rest(rc_i)).origin
			var d: Vector3 = Vector3(pl.x - pr.x, 0.0, pl.z - pr.z)
			if d.length() > 1e-4:
				side = d.normalized()
	var fwd: Vector3 = side.cross(Vector3.UP).normalized()
	if fwd.dot(Vector3.BACK) < 0.0:   # Rocketbox rest pose hướng mặt về +Z (Vector3.BACK trong Godot)
		side = -side
		fwd = -fwd
	var up: Vector3 = fwd.cross(side).normalized()
	var _unused := sp2_pos
	return {"side": side, "up": up, "fwd": fwd}


## Đo độ dày phía trước (+fwd) và phía sau (-fwd) của vùng ngực/lưng quanh `ref_xf.origin` (rest hoặc tư thế hiện tại).
static func _measure_chest_depth(
	npc: Node3D, mesh_node: MeshInstance3D, sk: Skeleton3D, ref_xf: Transform3D,
	side: Vector3, up: Vector3, fwd: Vector3, half_w: float, y_bot: float, y_top: float, use_rest: bool
) -> Vector2:
	var fallback := Vector2(0.15, -0.14)
	if mesh_node == null or mesh_node.mesh == null or mesh_node.skin == null:
		return fallback
	var surf: int = 0
	for s: int in mesh_node.mesh.get_surface_count():
		var m: Material = mesh_node.mesh.surface_get_material(s)
		if m != null and "body" in m.resource_name.to_lower():
			surf = s
			break
	var arr: Array = mesh_node.mesh.surface_get_arrays(surf)
	if arr.is_empty():
		return fallback
	var verts: PackedVector3Array = arr[Mesh.ARRAY_VERTEX]
	var bones: PackedInt32Array = arr[Mesh.ARRAY_BONES] if arr[Mesh.ARRAY_BONES] != null else PackedInt32Array()
	var weights: PackedFloat32Array = arr[Mesh.ARRAY_WEIGHTS] if arr[Mesh.ARRAY_WEIGHTS] != null else PackedFloat32Array()
	var skin: Skin = mesh_node.skin
	var b_per_v: int = bones.size() / maxi(verts.size(), 1)
	if b_per_v <= 0 or weights.size() != bones.size():
		return fallback

	var sk_xf: Transform3D = _local_xform(npc, sk)
	var bind_count: int = skin.get_bind_count()
	var torso_bind := PackedByteArray()
	torso_bind.resize(bind_count)
	var bone_xforms: Array[Transform3D] = []
	bone_xforms.resize(bind_count)
	var any_torso := false
	for bi: int in bind_count:
		var bname: StringName = skin.get_bind_name(bi)
		var sk_idx: int = sk.find_bone(bname) if bname != &"" else skin.get_bind_bone(bi)
		var bone_pose: Transform3D = (sk.get_bone_global_rest(sk_idx) if use_rest else sk.get_bone_global_pose(sk_idx)) if sk_idx >= 0 else Transform3D.IDENTITY
		bone_xforms[bi] = (sk_xf * bone_pose * skin.get_bind_pose(bi)) if sk_idx >= 0 else Transform3D.IDENTITY
		var is_t := TORSO_BONES.has(String(bname))
		torso_bind[bi] = 1 if is_t else 0
		any_torso = any_torso or is_t

	var ref_pos: Vector3 = ref_xf.origin
	var z_max: float = -INF
	var z_min: float = INF
	for i: int in verts.size():
		var tw: float = 0.0
		for k: int in b_per_v:
			var w: float = weights[i * b_per_v + k]
			if w > 0.0 and torso_bind[bones[i * b_per_v + k]] == 1:
				tw += w
		if any_torso and tw < 0.5:
			continue
		var p := Vector3.ZERO
		var v_raw: Vector3 = verts[i]
		for k: int in b_per_v:
			var w: float = weights[i * b_per_v + k]
			if w > 0.0:
				p += (bone_xforms[bones[i * b_per_v + k]] * v_raw) * w
		var rel: Vector3 = p - ref_pos
		var lx: float = rel.dot(side)
		var ly: float = rel.dot(up)
		if absf(lx) <= half_w + 0.02 and ly >= y_bot - 0.02 and ly <= y_top + 0.02:
			var lz: float = rel.dot(fwd)
			z_max = maxf(z_max, lz)
			z_min = minf(z_min, lz)

	if z_max == -INF or z_min == INF:
		return fallback
	return Vector2(z_max, z_min)
