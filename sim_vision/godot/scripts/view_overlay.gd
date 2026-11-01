class_name ViewOverlay
extends Control
## Ô góc nhìn robot ở góc trên bên phải (G5): ảnh camera + box đối tượng + 5-sector visual HUD.
## Bấm phím TAB để bật / tắt Sector HUD lớp phủ.

const BOX_COLOR_PATIENT := Color(0.2, 0.85, 1.0)      # Cyan sáng cho bệnh nhân
const BOX_COLOR_DOCTOR := Color(1.0, 0.35, 0.15)      # Cam đỏ cảnh báo cho bác sĩ
const BOX_COLOR_OBSTACLE := Color(0.95, 0.8, 0.2)     # Vàng cho vật cản / thiết bị
const TEXT_COLOR := Color(1, 1, 1)
const SHADOW_COLOR := Color(0, 0, 0, 0.85)
const FONT_SIZE: int = 13

var _dets: Array = []
var _state: String = ""
var _connected: bool = false
var _stt_shown: bool = false
var _stt_connected: bool = false
var _tts_shown: bool = false
var _tts_connected: bool = false
var _alert: String = ""
var _status: String = ""

# 5-Sector Visual Depth Overlay (Tab toggled)
var show_sector_hud: bool = true
var _sectors: Array = [0.0, 0.0, 0.0, 0.0, 0.0]

const ALERT_COLOR := Color(1, 0.15, 0.1)
const ALERT_FONT_SIZE: int = 22
const EMERGENCY_TEXT: String = "GỌI NHÂN VIÊN Y TẾ"

const SECTOR_LABELS: Array[String] = ["FL", "L", "C", "R", "FR"]

@onready var _view: TextureRect = $View
@onready var _boxes: Control = $Boxes


func _ready() -> void:
	_boxes.draw.connect(_on_boxes_draw)


func _unhandled_input(event: InputEvent) -> void:
	var key := event as InputEventKey
	if key == null or not key.pressed or key.echo:
		return
	if key.keycode == KEY_TAB:
		show_sector_hud = not show_sector_hud
		_boxes.queue_redraw()
		get_viewport().set_input_as_handled()


func set_viewport(viewport: SubViewport) -> void:
	_view.texture = viewport.get_texture()


func set_visual_sectors(sectors: Array) -> void:
	if sectors.size() == 5:
		_sectors = sectors
		_boxes.queue_redraw()


func show_detections(dets: Array, state: String, connected: bool) -> void:
	_dets = dets
	_state = state
	_connected = connected
	_boxes.queue_redraw()


static func norm_to_pixel(det: Dictionary, size: Vector2) -> Rect2:
	var p1 := Vector2(float(det["x1"]) * size.x, float(det["y1"]) * size.y)
	var p2 := Vector2(float(det["x2"]) * size.x, float(det["y2"]) * size.y)
	return Rect2(p1, p2 - p1)


func set_stt_connected(connected: bool) -> void:
	_stt_shown = true
	_stt_connected = connected
	_boxes.queue_redraw()


func set_tts_connected(connected: bool) -> void:
	_tts_shown = true
	_tts_connected = connected
	_boxes.queue_redraw()


func set_alert(text: String) -> void:
	_alert = text
	_boxes.queue_redraw()


func set_status(text: String) -> void:
	if text != _status and not text.is_empty():
		print("[status %.1fs] %s" % [Time.get_ticks_msec() / 1000.0, text])
	_status = text
	_boxes.queue_redraw()


func get_alert() -> String:
	return _alert


func get_status() -> String:
	return _status


static func guide_text(slots: Dictionary) -> String:
	if slots.has("department"):
		return "Đang dẫn đến: khoa %s" % str(slots["department"])
	if slots.has("room"):
		return "Đang dẫn đến: phòng %s" % str(slots["room"])
	return ""


static func stt_connection_text(connected: bool) -> String:
	return "STT: đã kết nối" if connected else "STT: mất kết nối"


static func tts_connection_text(connected: bool) -> String:
	return "TTS: đã kết nối" if connected else "TTS: mất kết nối"


static func label_text(det: Dictionary) -> String:
	var cname: String = str(det.get("class_name", "person"))
	var conf: float = float(det.get("conf", 0.0))
	return "%s %.2f" % [cname, conf]


static func get_box_color(det: Dictionary) -> Color:
	var cid: int = int(det.get("class_id", -1))
	var cname: String = str(det.get("class_name", ""))
	if cid == 0 or cname == "patient":
		return BOX_COLOR_PATIENT
	elif cid == 1 or cname == "doctor":
		return BOX_COLOR_DOCTOR
	return BOX_COLOR_OBSTACLE


static func connection_text(connected: bool) -> String:
	return "YOLO: đã kết nối" if connected else "YOLO: mất kết nối"


func _on_boxes_draw() -> void:
	var font: Font = ThemeDB.fallback_font
	var size: Vector2 = _boxes.size

	# 1. Vẽ Bounding Boxes đa nhãn (Bệnh nhân, Bác sĩ, Vật cản)
	for det: Variant in _dets:
		var d: Dictionary = det as Dictionary
		var r: Rect2 = ViewOverlay.norm_to_pixel(d, size)
		var col: Color = ViewOverlay.get_box_color(d)
		_boxes.draw_rect(r, col, false, 2.0)
		var label_pos: Vector2 = r.position + Vector2(2, -4) if r.position.y > FONT_SIZE + 2 else r.position + Vector2(2, FONT_SIZE + 2)
		_text(font, label_pos, ViewOverlay.label_text(d), col)

	# 2. Vẽ 5-Sector Visual Depth Overlay (Nếu Tab HUD được bật)
	if show_sector_hud:
		_draw_sector_hud(font, size)

	# 3. Trạng thái robot & WebSocket
	var state_y: float = (30.0 + FONT_SIZE) if show_sector_hud else (FONT_SIZE + 4)
	_text(font, Vector2(6, state_y), _state, TEXT_COLOR)
	_text(font, Vector2(6, size.y - 6), ViewOverlay.connection_text(_connected),
			TEXT_COLOR if _connected else Color(1, 0.35, 0.3))

	var line: int = 1
	if _stt_shown:
		_text(font, Vector2(6, size.y - 6 - (FONT_SIZE + 2) * line), "STT: đã kết nối" if _stt_connected else "STT: mất kết nối",
				TEXT_COLOR if _stt_connected else Color(1, 0.35, 0.3))
		line += 1
	if _tts_shown:
		_text(font, Vector2(6, size.y - 6 - (FONT_SIZE + 2) * line), "TTS: đã kết nối" if _tts_connected else "TTS: mất kết nối",
				TEXT_COLOR if _tts_connected else Color(1, 0.35, 0.3))

	if _status != "":
		_text(font, Vector2(6, 2 * FONT_SIZE + 8), _status, Color(1, 0.85, 0.3))

	if _alert != "":
		var w: float = font.get_string_size(_alert, HORIZONTAL_ALIGNMENT_LEFT, -1, ALERT_FONT_SIZE).x
		var pos := Vector2((size.x - w) / 2.0, size.y / 2.0)
		_boxes.draw_rect(Rect2(pos - Vector2(8, ALERT_FONT_SIZE + 4), Vector2(w + 16, ALERT_FONT_SIZE + 14)),
				Color(0, 0, 0, 0.7))
		_boxes.draw_string(font, pos, _alert, HORIZONTAL_ALIGNMENT_LEFT, -1, ALERT_FONT_SIZE, ALERT_COLOR)


func _draw_sector_hud(font: Font, size: Vector2) -> void:
	var hud_w: float = size.x - 12.0
	var bar_h: float = 6.0
	var sec_w: float = hud_w / 5.0
	var top_y: float = 3.0

	# Nền đen bán trong suốt cho HUD
	_boxes.draw_rect(Rect2(6.0, top_y, hud_w, bar_h + 16.0), Color(0.05, 0.08, 0.12, 0.75))

	for i in range(5):
		var x: float = 6.0 + i * sec_w
		var s_val: float = float(_sectors[i]) if i < _sectors.size() else 0.0
		# Màu sắc theo mức độ vật cản: Xanh (<0.35) -> Vàng (0.35-0.6) -> Đỏ (>=0.6)
		var col: Color = Color(0.2, 0.85, 0.3)
		if s_val >= 0.60:
			col = Color(0.95, 0.2, 0.2)
		elif s_val >= 0.35:
			col = Color(0.95, 0.8, 0.15)

		# Thanh thanh đo mức độ vật cản
		_boxes.draw_rect(Rect2(x + 1.0, top_y + 12.0, sec_w - 2.0, bar_h), Color(0.2, 0.2, 0.25, 0.6))
		var fill_w: float = (sec_w - 2.0) * clampf(s_val, 0.0, 1.0)
		if fill_w > 0.5:
			_boxes.draw_rect(Rect2(x + 1.0, top_y + 12.0, fill_w, bar_h), col)

		# Nhãn viết tắt góc nhìn (FL, L, C, R, FR)
		var lbl: String = SECTOR_LABELS[i]
		_boxes.draw_string(font, Vector2(x + (sec_w - 14.0) / 2.0, top_y + 10.0), lbl, HORIZONTAL_ALIGNMENT_CENTER, -1, 10, Color(0.85, 0.9, 0.95))

	# Chỉ dẫn Tab
	_boxes.draw_string(font, Vector2(size.x - 46.0, top_y + 28.0), "[TAB]", HORIZONTAL_ALIGNMENT_RIGHT, -1, 9, Color(0.6, 0.7, 0.8, 0.7))


func _text(font: Font, pos: Vector2, text: String, color: Color) -> void:
	_boxes.draw_string(font, pos + Vector2(1, 1), text, HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE, SHADOW_COLOR)
	_boxes.draw_string(font, pos, text, HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE, color)
