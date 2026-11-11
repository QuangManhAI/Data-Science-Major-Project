class_name NpcFace
extends RefCounted
## Dán ảnh mặt người thật lên NPC (thử nghiệm "pixel thật trong sim", docs/UPGRADE_HRI_PLAN.md):
## một tấm oval PHẲNG, viền cắt sắc, đặt ngay trước mặt NPC (vừa qua chóp mũi), bám xương "Bip01 Head" qua
## RemoteTransform3D ➔ đi theo đầu khi đi / quay / nói. Cỡ thật lấy từ khoảng cách hai xương mắt Rocketbox; texture
## (sim_vision/python/identity/make_npc_faces.py) đặt hai mắt ở 32% / 68% bề ngang, 40% chiều cao.

const FACES_DIR: String = "res://assets/textures/npc_faces"
const TEX_EYE_FRAC: float = 0.36       # khoảng cách hai mắt / bề ngang texture
const TEX_EYE_Y: float = 0.40          # mắt cách mép trên / chiều cao texture
const TEX_ASPECT: float = 320.0 / 256.0
const FRONT_OFFSET_M: float = 0.06     # trước tâm nhãn cầu chừng này (chóp mũi ~ 0.05 m)
const ALPHA_CUT: float = 0.5           # alpha scissor: viền oval sắc


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


## Gắn mặt; NPC phải đang ở trong cây scene (cần tư thế xương). Trả MeshInstance3D hoặc null.
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
	var w: float = pl.distance_to(pr) / TEX_EYE_FRAC          # cỡ thật: hai mắt ảnh trùng hai mắt NPC
	var h: float = w * TEX_ASPECT
	var mat := StandardMaterial3D.new()
	mat.albedo_texture = tex
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	mat.alpha_scissor_threshold = ALPHA_CUT
	mat.roughness = 0.75
	mat.cull_mode = BaseMaterial3D.CULL_BACK
	mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	var quad := QuadMesh.new()                                 # mặt trước quay +Z cục bộ
	quad.size = Vector2(w, h)
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
	# Attachment chỉ bám xương ở frame sau ➔ đặt ngay vào tư thế xương hiện tại rồi mới đặt tấm mặt.
	att.global_transform = sk.global_transform * sk.get_bone_global_pose(head)
	# QuadMesh: +X cục bộ là bên phải ảnh khi nhìn từ phía trước = bên trái NPC (side); tâm ảnh thấp hơn mắt.
	var center: Vector3 = eye_mid + fwd * FRONT_OFFSET_M - up * ((0.5 - TEX_EYE_Y) * h)
	var xf := Transform3D(Basis(side, up, fwd), center)
	var rt := RemoteTransform3D.new()
	rt.name = "FaceFollow"
	att.add_child(rt)
	rt.global_transform = xf
	rt.remote_path = rt.get_path_to(mi)
	mi.global_transform = xf
	return mi
