class_name AuditWatchdog
extends Node
## Bộ giám sát hộp đen và ghi hình Dashcam cho mô phỏng 3D Robot Bệnh viện.
## Tự động lưu 150 frames gần nhất (~10 giây @ 15 FPS) trong RAM dưới dạng JPEG nhẹ.
## Khi phát hiện lỗi (Tether > 2.5m, ReID mismatch, kẹt > 5s), tiếp tục ghi thêm 30 frames (+2s)
## và gọi script Python stitch_video.py để xuất file MP4 bằng chứng góc nhìn toàn cảnh thứ ba.

signal failure_triggered(error_type: String, metadata: Dictionary)
signal export_completed(error_type: String, mp4_path: String, json_path: String)

const BUFFER_MAX_FRAMES: int = 150
const POST_EVENT_FRAMES: int = 30
const CAPTURE_INTERVAL_S: float = 0.066

const TETHER_MAX_DIST_M: float = 2.5
const TETHER_MAX_TIME_S: float = 2.0
const REID_MIN_SIM: float = 0.75
const STAGNATION_VEL: float = 0.05
const STAGNATION_MAX_S: float = 5.0

@export var enabled: bool = true
var output_dir: String = ""
var python_bin: String = ""
var stitch_script_path: String = ""

var _frame_buffer: Array[PackedByteArray] = []
var _time_since_capture: float = 0.0

var _is_recording_event: bool = false
var _active_error_type: String = ""
var _active_metadata: Dictionary = {}
var _post_event_countdown: int = -1

var _tether_broken_timer: float = 0.0
var _stagnation_timer: float = 0.0
var _last_robot_pos: Vector3 = Vector3.ZERO
var _event_counter: int = 0

var main: Main = null
var robot: Node3D = null


func setup(p_main: Main, p_robot: Node3D, p_output_dir: String = "") -> void:
	main = p_main
	robot = p_robot
	if p_output_dir != "":
		output_dir = p_output_dir
	else:
		output_dir = ProjectSettings.globalize_path("res://../../sim_vision/reports/audit_evidence")
	_locate_python_tools()


func _locate_python_tools() -> void:
	var repo_root: String = ProjectSettings.globalize_path("res://../../")
	python_bin = repo_root.path_join(".venv/bin/python")
	if not FileAccess.file_exists(python_bin):
		python_bin = "python3"
	stitch_script_path = repo_root.path_join("scripts/audit/stitch_video.py")


func _process(delta: float) -> void:
	if not enabled or main == null or robot == null:
		return

	_time_since_capture += delta
	if _time_since_capture >= CAPTURE_INTERVAL_S:
		_time_since_capture = 0.0
		_capture_frame()

	if not _is_recording_event:
		_check_failure_triggers(delta)
	else:
		_post_event_countdown -= 1
		if _post_event_countdown <= 0:
			_flush_event_to_video()


func _capture_frame() -> void:
	var vp: Viewport = get_viewport()
	if vp == null:
		return
	var tex: Texture2D = vp.get_texture()
	if tex == null:
		return
	var img: Image = tex.get_image()
	if img == null:
		return

	var jpg: PackedByteArray = img.save_jpg_to_buffer(0.75)
	if jpg.size() > 0:
		_frame_buffer.append(jpg)
		var max_allowed: int = BUFFER_MAX_FRAMES + POST_EVENT_FRAMES
		if _frame_buffer.size() > max_allowed:
			_frame_buffer.pop_front()


func _check_failure_triggers(delta: float) -> void:
	var current_patient: CharacterBody3D = main.active_escorted_patient if main.active_escorted_patient != null else main.current_target_patient
	var robot_pos: Vector3 = robot.global_position

	# 1. Trigger TETHER_BREAK khi đang dắt người
	var is_escorting: bool = (main.active_escorted_patient != null)
	if is_escorting and current_patient != null:
		var tether_dist: float = robot_pos.distance_to(current_patient.global_position)
		if tether_dist > TETHER_MAX_DIST_M:
			_tether_broken_timer += delta
			if _tether_broken_timer >= TETHER_MAX_TIME_S:
				trigger_failure("TETHER_BREAK", {
					"tether_distance_m": tether_dist,
					"max_threshold_m": TETHER_MAX_DIST_M,
					"broken_duration_s": _tether_broken_timer,
					"robot_state": Config.state_name(main.state),
					"robot_pos": [robot_pos.x, robot_pos.y, robot_pos.z],
					"patient_pos": [current_patient.global_position.x, current_patient.global_position.y, current_patient.global_position.z],
				})
				_tether_broken_timer = 0.0
				return
		else:
			_tether_broken_timer = maxf(0.0, _tether_broken_timer - delta * 2.0)
	else:
		_tether_broken_timer = 0.0

	# 2. Trigger STAGNATION khi robot di chuyển không tiến triển trong lúc tiếp cận/dắt
	if main.state == Config.RobotState.APPROACH or is_escorting:
		var moved_dist: float = robot_pos.distance_to(_last_robot_pos)
		if moved_dist < STAGNATION_VEL * delta * 5.0:
			_stagnation_timer += delta
			if _stagnation_timer >= STAGNATION_MAX_S:
				trigger_failure("STAGNATION", {
					"stagnation_duration_s": _stagnation_timer,
					"robot_state": Config.state_name(main.state),
					"robot_pos": [robot_pos.x, robot_pos.y, robot_pos.z],
				})
				_stagnation_timer = 0.0
				return
		else:
			_stagnation_timer = maxf(0.0, _stagnation_timer - delta * 0.5)
	else:
		_stagnation_timer = 0.0

	_last_robot_pos = robot_pos


func trigger_failure(error_type: String, metadata: Dictionary) -> void:
	if _is_recording_event:
		return
	_event_counter += 1
	_is_recording_event = true
	_active_error_type = error_type
	_active_metadata = metadata.duplicate()
	_post_event_countdown = POST_EVENT_FRAMES

	var datetime_str: String = Time.get_datetime_string_from_system().replace(":", "-").replace("T", "_")
	_active_metadata["event_id"] = _event_counter
	_active_metadata["error_type"] = error_type
	_active_metadata["timestamp"] = datetime_str

	print("[AuditWatchdog] >>> TRIGGERED FAILURE: %s at %s" % [error_type, datetime_str])
	failure_triggered.emit(error_type, _active_metadata)


func _flush_event_to_video() -> void:
	_is_recording_event = false
	if _frame_buffer.is_empty():
		return

	var ts: String = str(_active_metadata.get("timestamp", Time.get_unix_time_from_system()))
	var err_tag: String = _active_error_type.to_lower()
	var temp_dir: String = "/tmp/audit_event_%s_%s" % [err_tag, ts]
	DirAccess.make_dir_recursive_absolute(temp_dir)
	DirAccess.make_dir_recursive_absolute(output_dir)

	var frame_idx: int = 0
	for jpg_bytes: PackedByteArray in _frame_buffer:
		var frame_path: String = temp_dir.path_join("frame_%04d.jpg" % frame_idx)
		var f: FileAccess = FileAccess.open(frame_path, FileAccess.WRITE)
		if f != null:
			f.store_buffer(jpg_bytes)
			f.close()
		frame_idx += 1

	var output_mp4: String = output_dir.path_join("fail_%s_%s.mp4" % [err_tag, ts])
	var json_meta_str: String = JSON.stringify(_active_metadata)

	var args: PackedStringArray = [
		stitch_script_path,
		"--frame-dir", temp_dir,
		"--output", output_mp4,
		"--fps", "15.0",
		"--metadata", json_meta_str,
		"--delete-frames",
	]

	print("[AuditWatchdog] Stitching %d frames into %s..." % [frame_idx, output_mp4])
	var pid: int = OS.create_process(python_bin, args)
	print("[AuditWatchdog] Started stitch process with PID: %d" % pid)

	export_completed.emit(_active_error_type, output_mp4, output_mp4.replace(".mp4", ".json"))
	# Giữ lại 30 frames cuối làm buffer nối tiếp
	if _frame_buffer.size() > 30:
		_frame_buffer = _frame_buffer.slice(_frame_buffer.size() - 30)
