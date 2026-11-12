class_name RenderLayers
extends RefCounted
## Tách thứ người xem thấy (ViewerCamera) và thứ camera robot thấy (ảnh gửi YOLO / YuNet).
## Tường trong suốt, ẩn trần chỉ để người xem dễ nhìn: camera robot phải luôn thấy tường đặc và trần như robot thật.
## Layer 10 (bit 9) đã dùng cho tay robot (RobotArms.VISUAL_LAYER).

const VIEWER_ONLY_BIT: int = 10   # layer 11: chỉ ViewerCamera render (tường "ma" trong suốt)
const ROBOT_ONLY_BIT: int = 11    # layer 12: chỉ camera robot render (tường đặc / trần đang ẩn với người xem)
const VIEWER_ONLY: int = 1 << VIEWER_ONLY_BIT
const ROBOT_ONLY: int = 1 << ROBOT_ONLY_BIT


static func setup_viewer_camera(cam: Camera3D) -> void:
	cam.cull_mask &= ~ROBOT_ONLY


static func setup_robot_camera(cam: Camera3D) -> void:
	cam.cull_mask &= ~VIEWER_ONLY
	cam.cull_mask |= ROBOT_ONLY


## Ẩn / hiện một mesh với người xem; camera robot vẫn luôn thấy.
static func set_viewer_visible(mi: VisualInstance3D, v: bool) -> void:
	if not mi.has_meta("orig_layers"):
		mi.set_meta("orig_layers", mi.layers)
	mi.layers = int(mi.get_meta("orig_layers")) if v else ROBOT_ONLY


static func is_viewer_visible(mi: VisualInstance3D) -> bool:
	return mi.visible and (mi.layers & ~ROBOT_ONLY) != 0
