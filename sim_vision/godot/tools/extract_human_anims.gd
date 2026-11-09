extends SceneTree
## Tách animation Rocketbox (FBX) thành file .res dùng cho human.tscn và các NPC bệnh viện.
## Chạy: godot --headless -s res://tools/extract_human_anims.gd
## Clip trong `IN_PLACE` bị gỡ root motion (hông Bip01 tiến 1.5 m/vòng rồi giật về đầu mỗi lần
## lặp ➔ NPC "giật"); script NPC tự di chuyển thân và đồng bộ nhịp bước (`speed_scale`).

const AVATAR_M: String = "res://assets/models/human_m06/Male_Adult_06.fbx"
const AVATAR_F: String = "res://assets/models/medical_f01/Medical_Female_01.fbx"
const SOURCES: Dictionary = {
	"idle": ["res://assets/models/human_m06/anims/m_idle_neutral_01.fbx", AVATAR_M],
	"walk": ["res://assets/models/human_m06/anims/m_walk_neutral.fbx", AVATAR_M],
	"m_idle_waiting": ["res://assets/models/human_m06/anims/m_idle_waiting_01.fbx", AVATAR_M],
	"m_sit_waiting": ["res://assets/models/human_m06/anims/m_sit_chair_idle_waiting_01.fbx", AVATAR_M],
	"m_sit_cough": ["res://assets/models/human_m06/anims/m_sit_chair_idle_cough.fbx", AVATAR_M],
	"f_idle": ["res://assets/models/human_m06/anims/f_idle_neutral_01.fbx", AVATAR_F],
	"f_documents_check": ["res://assets/models/human_m06/anims/f_documents_check.fbx", AVATAR_F],
	"f_sit_waiting": ["res://assets/models/human_m06/anims/f_sit_chair_idle_waiting_01.fbx", AVATAR_F],
	"f_sit_idle": ["res://assets/models/human_m06/anims/f_sit_chair_idle_neutral_01.fbx", AVATAR_F],
}
const IN_PLACE: Array[String] = ["walk"]
const ROOT_TRACK: NodePath = ^"Skeleton3D:Bip01"


## Trừ phần tiến tuyến tính (XZ) của xương gốc, giữ lắc ngang/nhún dọc trong vòng bước.
## Trả về tốc độ tiến tự nhiên của clip (m/s) để NPC chỉnh nhịp chân khớp vận tốc thật.
static func strip_root_motion(anim: Animation) -> float:
	var i: int = anim.find_track(ROOT_TRACK, Animation.TYPE_POSITION_3D)
	if i == -1 or anim.track_get_key_count(i) < 2 or anim.length <= 0.0:
		return 0.0
	var n: int = anim.track_get_key_count(i)
	var p0: Vector3 = anim.track_get_key_value(i, 0)
	var drift: Vector3 = anim.track_get_key_value(i, n - 1) - p0
	drift.y = 0.0
	for k: int in n:
		var t: float = anim.track_get_key_time(i, k) / anim.length
		anim.track_set_key_value(i, k, (anim.track_get_key_value(i, k) as Vector3) - drift * t)
	return drift.length() / anim.length


func _init() -> void:
	for anim_name: String in SOURCES:
		var entry: Array = SOURCES[anim_name]
		var fbx_path: String = entry[0]
		var avatar_path: String = entry[1]
		var avatar: Node = (load(avatar_path) as PackedScene).instantiate()
		var skeleton := avatar.get_node("Skeleton3D") as Skeleton3D
		var src: Node = (load(fbx_path) as PackedScene).instantiate()
		var player := src.get_node("AnimationPlayer") as AnimationPlayer
		var anim: Animation = player.get_animation("Take 001").duplicate(true)
		# Bỏ track của xương không có trong avatar (file animation có 122 xương, avatar 81).
		for i: int in range(anim.get_track_count() - 1, -1, -1):
			var path: NodePath = anim.track_get_path(i)
			if str(path.get_concatenated_names()) != "Skeleton3D" or path.get_subname_count() == 0 \
					or skeleton.find_bone(str(path.get_subname(0))) == -1:
				anim.remove_track(i)
		anim.loop_mode = Animation.LOOP_LINEAR
		if anim_name in IN_PLACE:
			print("%s: root motion %.3f m/s ➔ in-place" % [anim_name, strip_root_motion(anim)])
		var out: String = "res://assets/models/human_m06/anims/%s.res" % anim_name
		var err: Error = ResourceSaver.save(anim, out)
		print("%s: %d tracks, %.2f s ➔ %s (%s)" % [anim_name, anim.get_track_count(), anim.length, out, error_string(err)])
		src.free()
		avatar.free()
	quit()
