extends SceneTree
## Tách animation Rocketbox (FBX) thành file .res dùng cho human.tscn.
## Chạy: godot --headless -s res://tools/extract_human_anims.gd

const AVATAR: String = "res://assets/models/human_m06/Male_Adult_06.fbx"
const SOURCES: Dictionary = {
	"idle": "res://assets/models/human_m06/anims/m_idle_neutral_01.fbx",
	"walk": "res://assets/models/human_m06/anims/m_walk_neutral.fbx",
}


func _init() -> void:
	var avatar: Node = (load(AVATAR) as PackedScene).instantiate()
	var skeleton := avatar.get_node("Skeleton3D") as Skeleton3D
	for anim_name: String in SOURCES:
		var src: Node = (load(SOURCES[anim_name]) as PackedScene).instantiate()
		var player := src.get_node("AnimationPlayer") as AnimationPlayer
		var anim: Animation = player.get_animation("Take 001").duplicate(true)
		# Bỏ track của xương không có trong avatar (file animation có 122 xương, avatar 81).
		for i: int in range(anim.get_track_count() - 1, -1, -1):
			var path: NodePath = anim.track_get_path(i)
			if str(path.get_concatenated_names()) != "Skeleton3D" or path.get_subname_count() == 0 \
					or skeleton.find_bone(str(path.get_subname(0))) == -1:
				anim.remove_track(i)
		anim.loop_mode = Animation.LOOP_LINEAR
		var out: String = "res://assets/models/human_m06/anims/%s.res" % anim_name
		var err: Error = ResourceSaver.save(anim, out)
		print("%s: %d tracks, %.2f s ➔ %s (%s)" % [anim_name, anim.get_track_count(), anim.length, out, error_string(err)])
		src.free()
	avatar.free()
	quit()
