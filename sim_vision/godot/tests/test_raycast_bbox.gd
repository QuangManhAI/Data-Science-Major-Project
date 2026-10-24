extends SceneTree

func _init() -> void:
	print("--- TEST BBOX RAYCAST WITH NURSE & VITAL MONITOR ---")
	var scene_res = load("res://scenes/hospital_vision_bench_unified.tscn")
	var main_node = scene_res.instantiate()
	root.add_child(main_node)
	
	await process_frame
	await process_frame
	
	var camera: Camera3D = main_node.get_node("RobotCamera")
	var actors = main_node.get_node("ActorsPool")
	var props = main_node.get_node("ObstacleProps")
	var hneg = main_node.get_node("HardNegativesPool")
	
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
	
	# Case A: Nurse standing alone (No Obstacle)
	nurse.visible = true
	nurse.global_position = Vector3(0, 0, 0)
	
	camera.global_position = Vector3(0, 1.15, 2.0)
	camera.look_at(Vector3(0, 1.0, 0), Vector3.UP)
	
	await physics_frame
	await physics_frame
	
	var res_alone = _test_compute_bbox(main_node, camera, nurse, "standing", Vector2(640, 480))
	print("CASE A (No obstacle):")
	print("  Vis ratio:", res_alone.get("visibility_ratio"))
	print("  Visible BBox:", res_alone.get("bbox_xyxy_px"))
	print("  Full BBox:", res_alone.get("bbox_full_xyxy"))
	
	# Case B: Nurse with Monitor Cart in front (h=1.2m, z=0.5m)
	monitor.visible = true
	monitor.global_position = Vector3(0, 0, 0.5)
	
	await physics_frame
	await physics_frame
	
	var res_occluded = _test_compute_bbox(main_node, camera, nurse, "standing", Vector2(640, 480))
	print("CASE B (Nurse behind Vital Monitor Cart):")
	print("  Vis ratio:", res_occluded.get("visibility_ratio"))
	print("  Visible BBox:", res_occluded.get("bbox_xyxy_px"))
	print("  Full BBox:", res_occluded.get("bbox_full_xyxy"))
	
	quit(0)

func _test_compute_bbox(tree_root: Node3D, cam: Camera3D, actor_root: Node3D, subcategory: String, vp_size: Vector2) -> Dictionary:
	var space_state = tree_root.get_world_3d().direct_space_state
	var h: float = 1.75
	var r: float = 0.28
	
	var local_pts: Array[Vector3] = []
	var y_layers = [0.10, 0.30, 0.55, 0.85, 1.15, 1.40, 1.60, 1.72]
	for y_val in y_layers:
		var cur_y = y_val if y_val < h else h * 0.95
		var cur_r = r if cur_y < h * 0.8 else r * 0.65
		for theta_deg in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
			var rad = deg_to_rad(theta_deg)
			local_pts.append(Vector3(cur_r * cos(rad), cur_y, cur_r * sin(rad)))
	local_pts.append(Vector3(0.0, h, 0.0))
	local_pts.append(Vector3(0.0, h * 0.9, 0.12))
	
	var visible_2d_pts: Array[Vector2] = []
	var all_2d_pts: Array[Vector2] = []
	var front_pts_count: int = 0
	var unoccluded_count: int = 0
	
	for lp in local_pts:
		var wp: Vector3 = actor_root.global_transform * lp
		if cam.is_position_behind(wp):
			continue
		front_pts_count += 1
		var uv: Vector2 = cam.unproject_position(wp)
		all_2d_pts.append(uv)
		
		var q = PhysicsRayQueryParameters3D.create(cam.global_position, wp)
		var hit = space_state.intersect_ray(q)
		
		if hit.is_empty():
			unoccluded_count += 1
			visible_2d_pts.append(uv)
		else:
			var collider = hit.get("collider")
			if collider == actor_root or actor_root.is_ancestor_of(collider):
				unoccluded_count += 1
				visible_2d_pts.append(uv)
				
	var vis_ratio: float = float(unoccluded_count) / float(max(front_pts_count, 1))
	
	var all_x: Array = []
	var all_y: Array = []
	for p in all_2d_pts:
		all_x.append(p.x)
		all_y.append(p.y)
	all_x.sort()
	all_y.sort()
	
	var vis_x: Array = []
	var vis_y: Array = []
	for p in visible_2d_pts:
		vis_x.append(p.x)
		vis_y.append(p.y)
	vis_x.sort()
	vis_y.sort()
	
	var full_min_x = clampf(all_x[0], 0.0, vp_size.x)
	var full_min_y = clampf(all_y[0], 0.0, vp_size.y)
	var full_max_x = clampf(all_x[all_x.size() - 1], 0.0, vp_size.x)
	var full_max_y = clampf(all_y[all_y.size() - 1], 0.0, vp_size.y)
	
	if vis_x.is_empty():
		return {"valid": false, "visibility_ratio": 0.0, "bbox_full_xyxy": [full_min_x, full_min_y, full_max_x, full_max_y]}
		
	var vis_min_x = clampf(vis_x[0], 0.0, vp_size.x)
	var vis_min_y = clampf(vis_y[0], 0.0, vp_size.y)
	var vis_max_x = clampf(vis_x[vis_x.size() - 1], 0.0, vp_size.x)
	var vis_max_y = clampf(vis_y[vis_y.size() - 1], 0.0, vp_size.y)
	
	return {
		"valid": true,
		"visibility_ratio": snappedf(vis_ratio, 0.01),
		"bbox_xyxy_px": [snappedf(vis_min_x, 0.1), snappedf(vis_min_y, 0.1), snappedf(vis_max_x, 0.1), snappedf(vis_max_y, 0.1)],
		"bbox_full_xyxy": [snappedf(full_min_x, 0.1), snappedf(full_min_y, 0.1), snappedf(full_max_x, 0.1), snappedf(full_max_y, 0.1)]
	}
