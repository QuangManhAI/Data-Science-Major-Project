class_name Config
extends RefCounted
## Hằng số dùng chung (Mục 2.1) và trạng thái robot (Mục 2.3).
## Tên và giá trị phải khớp đúng file kế hoạch; không thêm hằng số mới ở đây.

const WS_URL: String = "ws://127.0.0.1:8765"
const CAPTURE_HZ: int = 5
const IMG_W: int = 640
const IMG_H: int = 480
const JPEG_QUALITY: float = 0.8
const YOLO_CONF: float = 0.4
const V_MAX: float = 0.8            # m/s
const OMEGA_MAX: float = 1.5        # rad/s
const K_ANG: float = 2.0
const K_LIN: float = 1.5
const H_TARGET: float = 0.85
const H_HYST: float = 0.10
const ALIGN_TOL: float = 0.25
const LOST_TIMEOUT_S: float = 1.5
const SEARCH_OMEGA: float = 0.5     # rad/s
const STALE_MS: int = 500
const SAFE_DIST_HUMAN: float = 0.8  # m
const SAFE_DIST_OBSTACLE: float = 0.5  # m
const RECONNECT_S: float = 2.0

enum RobotState { SEARCH, APPROACH, ARRIVED }


## Trả tên trạng thái đúng chuỗi ở Mục 2.3; giá trị lạ ➔ push_error và trả "".
static func state_name(s: int) -> String:
	match s:
		RobotState.SEARCH:
			return "SEARCH"
		RobotState.APPROACH:
			return "APPROACH"
		RobotState.ARRIVED:
			return "ARRIVED"
	push_error("Config.state_name: unknown state %d" % s)
	return ""
