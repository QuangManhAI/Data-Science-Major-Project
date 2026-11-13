class_name NpcFace
extends RefCounted
## Dán ảnh mặt người thật lên NPC (thử nghiệm "pixel thật trong sim", docs/UPGRADE_HRI_PLAN.md):
## đầu NPC được bọc bằng một KHỐI HỘP VUÔNG (che kín đầu gốc), mặt trước dán nguyên ảnh mặt vuông — không cắt oval.
## Hộp bám xương "Bip01 Head" qua RemoteTransform3D ➔ đi theo đầu khi đi / quay / nói. Hướng hộp lấy theo hai xương mắt
## Rocketbox (nhìn ngang, bỏ cúi / ngửa). Texture: sim_vision/python/identity/make_npc_faces.py (RGB 256x256, hai mắt
## ở 34% / 66% bề ngang, 42% chiều cao).

const FACES_DIR: String = "res://assets/textures/npc_faces"
const HEAD_SIZE_M: Vector3 = Vector3(0.26, 0.28, 0.27)   # rộng, cao, sâu: đủ che đầu Rocketbox (kể cả tóc ngắn, mũi)
const EYE_BEHIND_FRONT_M: float = 0.075  # mặt trước hộp nằm trước hai mắt chừng này (chóp mũi ~0.05 m)
const TEX_EYE_Y: float = 0.42            # mắt trong ảnh cách mép trên / chiều cao ảnh
const FRONT_EPS_M: float = 0.002         # ảnh nằm ngay trước mặt hộp (tránh z-fighting)


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


## Màu mặt bên của hộp: trung bình viền ảnh (tóc / da / nền quanh mặt) ➔ hộp không lộ khác màu với ảnh.
static func side_color(tex: Texture2D) -> Color:
	var img: Image = tex.get_image() if tex != null else null
	if img == null or img.get_width() < 4:
		return Color(0.35, 0.27, 0.22)
	var acc := Color(0, 0, 0, 0)
	var n: int = 0
	var w: int = img.get_width()
	var h: int = img.get_height()
	for i: int in 32:
		var t: float = (i + 0.5) / 32.0
		for p: Vector2i in [Vector2i(int(t * (w - 1)), 1), Vector2i(1, int(t * (h - 1))), Vector2i(w - 2, int(t * (h - 1)))]:
			acc += img.get_pixelv(p)
			n += 1
	return Color(acc.r / n, acc.g / n, acc.b / n, 1.0)


## Gắn đầu hộp + mặt; NPC phải đang ở trong cây scene (cần tư thế xương). Trả MeshInstance3D của tấm mặt hoặc null.
static func attach(npc: Node3D, tex: Texture2D) -> MeshInstance3D:
	if tex == null:
		return null
	var found: Array[Node] = npc.find_children("*", "Skeleton3D", true, false)
	if found.is_empty():
		return null
	var sk := found[0] as Skeleton3D
	var head: int = sk.find_bone("Bip01 Head")
	var le: int = sk.find_bone("Bip01 LEye")
	var re: int = sk.find_bone("Bip01 REye")
	if head < 0 or le < 0 or re < 0:
		return null
	var pl: Vector3 = (sk.global_transform * sk.get_bone_global_pose(le)).origin
	var pr: Vector3 = (sk.global_transform * sk.get_bone_global_pose(re)).origin
	var eye_mid: Vector3 = (pl + pr) * 0.5
	# Hướng mặt theo đầu (không theo thân): trục ngang = mắt phải ➔ mắt trái; nhìn ngang (bỏ cúi / ngửa).
	var side: Vector3 = (pl - pr).normalized()
	var fwd: Vector3 = side.cross(Vector3.UP).normalized()
	if fwd.dot(npc.global_basis.z) < 0.0:                     # phòng khi tên xương trái / phải ngược quy ước
		side = -side
		fwd = -fwd
	var up: Vector3 = fwd.cross(side).normalized()
	var sz: Vector3 = HEAD_SIZE_M
	# Mắt trong ảnh trùng độ cao mắt NPC; mặt trước hộp trước mắt EYE_BEHIND_FRONT_M.
	var front_c: Vector3 = eye_mid + fwd * EYE_BEHIND_FRONT_M - up * ((0.5 - TEX_EYE_Y) * sz.y)
	var box_c: Vector3 = front_c - fwd * (sz.z * 0.5)

	var box_mat := StandardMaterial3D.new()
	box_mat.albedo_color = NpcFace.side_color(tex)
	box_mat.roughness = 0.85
	var box := BoxMesh.new()
	box.size = sz
	box.material = box_mat
	var head_mi := MeshInstance3D.new()
	head_mi.name = "BoxHead"
	head_mi.mesh = box
	npc.add_child(head_mi)

	var mat := StandardMaterial3D.new()
	mat.albedo_texture = tex
	mat.transparency = BaseMaterial3D.TRANSPARENCY_DISABLED   # ảnh vuông nguyên vẹn, không cắt viền
	mat.roughness = 0.75
	mat.cull_mode = BaseMaterial3D.CULL_BACK
	mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	var quad := QuadMesh.new()                                 # mặt trước quay +Z cục bộ
	quad.size = Vector2(sz.x, sz.y)
	quad.material = mat
	var mi := MeshInstance3D.new()
	mi.name = "FaceMask"
	mi.mesh = quad
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	npc.add_child(mi)

	var att := BoneAttachment3D.new()
	att.name = "FaceAttachment"
	att.bone_name = "Bip01 Head"
	sk.add_child(att)
	# Attachment chỉ bám xương ở frame sau ➔ đặt ngay vào tư thế xương hiện tại rồi mới đặt hộp / tấm mặt.
	att.global_transform = sk.global_transform * sk.get_bone_global_pose(head)
	var basis := Basis(side, up, fwd)        # QuadMesh: +X cục bộ = bên phải ảnh khi nhìn từ phía trước
	for pair: Array in [[head_mi, Transform3D(basis, box_c), "BoxFollow"],
			[mi, Transform3D(basis, front_c + fwd * FRONT_EPS_M), "FaceFollow"]]:
		var target := pair[0] as MeshInstance3D
		var xf: Transform3D = pair[1]
		var rt := RemoteTransform3D.new()
		rt.name = pair[2]
		att.add_child(rt)
		rt.global_transform = xf
		rt.remote_path = rt.get_path_to(target)
		target.global_transform = xf
	return mi
