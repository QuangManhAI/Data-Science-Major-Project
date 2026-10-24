extends SceneTree

func _init() -> void:
	print("--- TEST PHYSICS ISOLATION ---")
	var scene_res = load("res://scenes/hospital_vision_bench_extended.tscn")
	var main_node = scene_res.instantiate()
	root.add_child(main_node)
	
	await process_frame
	await process_frame
	
	var space_state = main_node.get_world_3d().direct_space_state
	var camera: Camera3D = main_node.get_node("RobotCamera")
	var actors = main_node.get_node("ActorsPool")
	var props = main_node.get_node("ObstacleProps")
	var hneg = main_node.get_node("HardNegativesPool")
	
	# Move ALL inactive actors and props far away underground
	for c in actors.get_children():
		if c is Node3D:
			c.global_position = Vector3(0, -500, 0)
			c.visible = false
	for c in props.get_children():
		if c is Node3D:
			c.global_position = Vector3(0, -500, 0)
			c.visible = false
	for c in hneg.get_children():
		if c is Node3D:
			c.global_position = Vector3(0, -500, 0)
			c.visible = false
			
	var nurse = main_node.get_node("ActorsPool/nurse_female")
	var monitor = main_node.get_node("ObstacleProps/vital_monitor")
	
	# Position ONLY nurse and monitor
	nurse.visible = true
	nurse.global_position = Vector3(0, 0, 0)
	
	monitor.visible = true
	# Monitor is 1.2m high, 0.48m wide. Position it at 0.5m in front of nurse
	monitor.global_position = Vector3(0, 0, 0.5)
	
	camera.global_position = Vector3(0, 1.15, 2.0)
	camera.look_at(Vector3(0, 1.0, 0), Vector3.UP)
	
	await physics_frame
	await physics_frame
	
	# Head (1.65m) -> should NOT hit monitor!
	var q_head = PhysicsRayQueryParameters3D.create(camera.global_position, Vector3(0, 1.65, 0))
	var hit_head = space_state.intersect_ray(q_head)
	print("Head ray hit:", hit_head.get("collider"))
	
	# Knee (0.50m) -> MUST HIT VITAL MONITOR!
	var q_knee = PhysicsRayQueryParameters3D.create(camera.global_position, Vector3(0, 0.50, 0))
	var hit_knee = space_state.intersect_ray(q_knee)
	print("Knee ray hit:", hit_knee.get("collider"))
	
	quit(0)
