extends SceneTree

func _init() -> void:
	print("--- BẮT ĐẦU TEST RAYCAST CHE KHUẤT TRONG GODOT 4 ---")
	var scene_res = load("res://scenes/hospital_vision_bench_extended.tscn")
	var main_node = scene_res.instantiate()
	root.add_child(main_node)
	
	# Wait one frame for physics engine to initialize
	await process_frame
	await process_frame
	
	var space_state = main_node.get_world_3d().direct_space_state
	var camera: Camera3D = main_node.get_node("RobotCamera")
	var nurse = main_node.get_node("ActorsPool/nurse_female")
	var monitor = main_node.get_node("ObstacleProps/vital_monitor")
	
	print("Camera pos:", camera.global_position)
	print("Nurse pos:", nurse.global_position)
	
	# Position nurse at (0, 0, 0)
	nurse.visible = true
	nurse.global_position = Vector3(0, 0, 0)
	
	# Position monitor directly between camera and nurse
	monitor.visible = true
	monitor.global_position = Vector3(0, 0, 0.5)
	
	# Position camera at (0, 1.15, 2.0) looking at nurse
	camera.global_position = Vector3(0, 1.15, 2.0)
	camera.look_at(Vector3(0, 1.0, 0), Vector3.UP)
	
	# Wait for physics transforms to update
	await physics_frame
	await physics_frame
	
	# Test ray to nurse head (y = 1.6m) -> should NOT be blocked by monitor (h=1.2m)
	var head_pos = Vector3(0, 1.65, 0)
	var q_head = PhysicsRayQueryParameters3D.create(camera.global_position, head_pos)
	var hit_head = space_state.intersect_ray(q_head)
	print("Ray to Head (1.65m): Hit =", hit_head.get("collider"))
	
	# Test ray to nurse waist/knees (y = 0.6m) -> SHOULD BE BLOCKED BY MONITOR!
	var knee_pos = Vector3(0, 0.6, 0)
	var q_knee = PhysicsRayQueryParameters3D.create(camera.global_position, knee_pos)
	var hit_knee = space_state.intersect_ray(q_knee)
	print("Ray to Knees (0.60m): Hit =", hit_knee.get("collider"))
	
	print("--- TEST HOÀN TẤT ---")
	quit(0)
