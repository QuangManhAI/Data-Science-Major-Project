class_name ViewOverlay
extends Control
## Ô camera robot ở góc trên bên phải: CAMERA TRƯỚC (ảnh gửi YOLO + policy) và CAMERA SAU (chỉ khi dẫn đường,
## để thấy bệnh nhân đi sau / tụt lại). Chữ không đè lên ảnh: tiêu đề nằm trên mỗi ảnh, trạng thái + kết nối ở
## thanh dưới cùng; nhãn box có nền, không chồng nhau, không tràn khung. Bệnh nhân đang dẫn: khung xanh lá đậm.

const PANEL_W: float = 320.0
## Ô "MẶT BẮT ĐƯỢC" bên trái camera trước: mặt đã ghi nhớ + mặt đang thấy (cắt từ ảnh camera theo box YuNet).
const FACE_W: float = 120.0
const CAM_X: float = FACE_W + 6.0
const TOTAL_W: float = CAM_X + PANEL_W
const LIVE_FACES_MAX: int = 2
const IMG_H: float = 240.0
const HEADER_H: float = 22.0
const SECTOR_H: float = 18.0
const FOOTER_H: float = 44.0
const GAP: float = 6.0
const PAD: float = 6.0

const BOX_COLOR_PATIENT := Color(0.2, 0.85, 1.0)
const BOX_COLOR_TRACKED := Color(0.3, 1.0, 0.45)
const BOX_COLOR_OTHER := Color(0.55, 0.7, 0.8, 0.7)
const BOX_COLOR_DOCTOR := Color(1.0, 0.45, 0.2)
const BOX_COLOR_OBSTACLE := Color(0.95, 0.8, 0.2, 0.7)
const FACE_COLOR := Color(1.0, 0.45, 0.85)              # box mặt (YuNet): hồng, nét đứt ➔ khác box YOLO (nét liền)
const FACE_DASH: float = 4.0
const TEXT_COLOR := Color(0.93, 0.95, 0.97)
const MUTED_COLOR := Color(0.6, 0.66, 0.72)
const PANEL_BG := Color(0.07, 0.09, 0.12, 0.92)
const HEADER_BG := Color(0.12, 0.15, 0.2, 0.95)
const OK_COLOR := Color(0.35, 0.9, 0.45)
const BAD_COLOR := Color(1.0, 0.38, 0.32)
const STATUS_COLOR := Color(1.0, 0.85, 0.35)
const FONT_SIZE: int = 13
const LABEL_FONT_SIZE: int = 11
const ALERT_COLOR := Color(1, 0.15, 0.1)
const ALERT_FONT_SIZE: int = 22
const EMERGENCY_TEXT: String = "GỌI NHÂN VIÊN Y TẾ"
const SECTOR_LABELS: Array[String] = ["FL", "L", "C", "R", "FR"]

var _dets: Array = []
var _rear_dets: Array = []
var _faces: Array = []
var _rear_faces: Array = []
var _remembered_face: Texture2D = null
var _live_faces: Array = []          # [{tex: Texture2D, conf: float, mine: int}] mine: 1 đang dắt, 0 người khác, -1 chưa rõ
var _rear_live_faces: Array = []
var _tracked: Dictionary = {}
var _rear_tracked: Dictionary = {}
var _rear_active: bool = false
var _has_rear: bool = false
var _state: String = ""
var _connected: bool = false
var _stt_shown: bool = false
var _stt_connected: bool = false
var _tts_shown: bool = false
var _tts_connected: bool = false
var _alert: String = ""
var _status: String = ""
var show_sector_hud: bool = true
var _sectors: Array = [0.0, 0.0, 0.0, 0.0, 0.0]

var _front_view: TextureRect
var _rear_view: TextureRect
var _canvas: Control


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	_front_view = _make_view("FrontView")
	_rear_view = _make_view("RearView")
	_canvas = Control.new()
	_canvas.name = "Canvas"
	_canvas.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_canvas)
	_canvas.draw.connect(_on_canvas_draw)
	_layout()


func _make_view(n: String) -> TextureRect:
	var t := TextureRect.new()
	t.name = n
	t.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	t.stretch_mode = TextureRect.STRETCH_SCALE
	t.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(t)
	return t


# ---------- vị trí các khối ----------

func face_panel_rect() -> Rect2:
	return Rect2(0, 0, FACE_W, sector_rect().end.y)


## Ô mặt camera sau (cạnh ô camera sau, chỉ khi đang dẫn): mặt đang thấy ở sau lưng, có phải người đang dắt không.
func rear_face_panel_rect() -> Rect2:
	return Rect2(0, rear_header_rect().position.y, FACE_W, HEADER_H + rear_image_rect().size.y)


func front_header_rect() -> Rect2:
	return Rect2(CAM_X, 0, PANEL_W, HEADER_H)


func front_image_rect() -> Rect2:
	return Rect2(CAM_X, HEADER_H, PANEL_W, IMG_H)


func sector_rect() -> Rect2:
	return Rect2(CAM_X, HEADER_H + IMG_H, PANEL_W, SECTOR_H)


func rear_header_rect() -> Rect2:
	return Rect2(CAM_X, sector_rect().end.y + GAP, PANEL_W, HEADER_H)


func rear_image_rect() -> Rect2:
	return Rect2(CAM_X, rear_header_rect().end.y, PANEL_W, IMG_H if _rear_active else 0.0)


func footer_rect() -> Rect2:
	var top: float = (rear_image_rect().end.y if _has_rear else sector_rect().end.y) + GAP
	return Rect2(CAM_X, top, PANEL_W, FOOTER_H)


func total_height() -> float:
	return footer_rect().end.y


func _layout() -> void:
	if _canvas == null:
		return
	custom_minimum_size = Vector2(TOTAL_W, total_height())
	offset_left = offset_right - TOTAL_W
	offset_bottom = offset_top + total_height()
	size = Vector2(TOTAL_W, total_height())
	_front_view.position = front_image_rect().position
	_front_view.size = front_image_rect().size
	_rear_view.visible = _has_rear and _rear_active
	_rear_view.position = rear_image_rect().position
	_rear_view.size = Vector2(PANEL_W, IMG_H)
	_canvas.position = Vector2.ZERO
	_canvas.size = size
	_canvas.queue_redraw()


# ---------- API ----------

func set_viewport(viewport: SubViewport) -> void:
	_front_view.texture = viewport.get_texture()


func set_rear_viewport(viewport: SubViewport) -> void:
	_rear_view.texture = viewport.get_texture()
	_has_rear = true
	_layout()


func set_rear_active(active: bool) -> void:
	if active != _rear_active:
		_rear_active = active
		if not active:
			_rear_dets = []
			_rear_faces = []
			_rear_live_faces = []
			_rear_tracked = {}
		_layout()


func set_visual_sectors(sectors: Array) -> void:
	if sectors.size() == 5:
		_sectors = sectors
		_redraw()


func show_detections(dets: Array, state: String, connected: bool) -> void:
	_dets = dets
	_state = state
	_connected = connected
	_redraw()


func show_rear_detections(dets: Array) -> void:
	_rear_dets = dets
	_redraw()


## Mặt của bệnh nhân robot đã ghi nhớ lúc nói chuyện (null = chưa ghi nhớ / đã xoá).
func set_remembered_face(tex: Texture2D) -> void:
	_remembered_face = tex
	_redraw()


## Mặt đang thấy ở camera trước: [{tex, conf}], to nhất trước.
func set_live_faces(items: Array, rear_items: Array = []) -> void:
	_live_faces = items.slice(0, LIVE_FACES_MAX)
	_rear_live_faces = rear_items.slice(0, LIVE_FACES_MAX)
	_redraw()


## Box mặt (YuNet) của từng camera, toạ độ chuẩn hoá; vẽ nét đứt.
func show_faces(front: Array, rear: Array = []) -> void:
	_faces = front
	_rear_faces = rear
	_redraw()


## Box của bệnh nhân đang dẫn trên từng camera ({} = không thấy).
func set_tracked(front_det: Dictionary, rear_det: Dictionary) -> void:
	_tracked = front_det
	_rear_tracked = rear_det
	_redraw()


func set_stt_connected(connected: bool) -> void:
	_stt_shown = true
	_stt_connected = connected
	_redraw()


func set_tts_connected(connected: bool) -> void:
	_tts_shown = true
	_tts_connected = connected
	_redraw()


func set_alert(text: String) -> void:
	_alert = text
	_redraw()


func set_status(text: String) -> void:
	if text != _status and not text.is_empty():
		print("[status %.1fs] %s" % [Time.get_ticks_msec() / 1000.0, text])
	_status = text
	_redraw()


func get_alert() -> String:
	return _alert


func get_status() -> String:
	return _status


func _redraw() -> void:
	if _canvas != null:
		_canvas.queue_redraw()


# ---------- chữ / màu (dùng cả trong test) ----------

static func norm_to_pixel(det: Dictionary, size: Vector2) -> Rect2:
	var p1 := Vector2(float(det["x1"]) * size.x, float(det["y1"]) * size.y)
	var p2 := Vector2(float(det["x2"]) * size.x, float(det["y2"]) * size.y)
	return Rect2(p1, p2 - p1)


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


static func connection_text(connected: bool) -> String:
	return "YOLO: đã kết nối" if connected else "YOLO: mất kết nối"


static func label_text(det: Dictionary) -> String:
	var cname: String = str(det.get("class_name", ""))
	if cname.is_empty():
		var cid: int = int(det.get("class_id", -1))
		cname = "doctor" if cid == 1 else ("obstacle_%d" % cid if cid >= 2 else "patient")
	return "%s %.2f" % [cname, float(det.get("conf", 0.0))]


static func is_person(det: Dictionary) -> bool:
	var cid: int = int(det.get("class_id", -1))
	var cname: String = str(det.get("class_name", ""))
	return cid in [0, 1] or cname in ["patient", "doctor"]


static func get_box_color(det: Dictionary) -> Color:
	var cid: int = int(det.get("class_id", -1))
	var cname: String = str(det.get("class_name", ""))
	if cid == 0 or cname == "patient":
		return BOX_COLOR_PATIENT
	elif cid == 1 or cname == "doctor":
		return BOX_COLOR_DOCTOR
	return BOX_COLOR_OBSTACLE


## Nhãn ngắn tiếng Việt trên ảnh; vật cản không ghi nhãn (đỡ rối, màu vàng đủ hiểu).
static func short_label(det: Dictionary, tracked: bool, tracking: bool) -> String:
	if tracked:
		return "đang dẫn"
	var cid: int = int(det.get("class_id", -1))
	var cname: String = str(det.get("class_name", ""))
	if cid == 0 or cname == "patient":
		return "bệnh nhân khác" if tracking else "bệnh nhân %.0f%%" % (100.0 * float(det.get("conf", 0.0)))
	if cid == 1 or cname == "doctor":
		return "bác sĩ"
	return ""


## Cắt chuỗi cho vừa `max_w` px (thêm "…").
static func ellipsize(font: Font, text: String, font_size: int, max_w: float) -> String:
	if font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x <= max_w:
		return text
	var t: String = text
	while t.length() > 1 and font.get_string_size(t + "…", HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x > max_w:
		t = t.left(t.length() - 1)
	return t + "…"


## Đặt nhãn: ưu tiên ngay trên box, trong khung ảnh; trùng nhãn khác ➔ dời xuống. Trả rect nhãn.
static func place_label(box: Rect2, label_size: Vector2, area: Rect2, placed: Array[Rect2]) -> Rect2:
	var r := Rect2(Vector2(box.position.x, box.position.y - label_size.y - 1.0), label_size)
	if r.position.y < area.position.y:
		r.position.y = box.position.y + 1.0
	for i: int in 6:
		r.position.x = clampf(r.position.x, area.position.x, area.end.x - label_size.x)
		r.position.y = clampf(r.position.y, area.position.y, area.end.y - label_size.y)
		var hit: bool = false
		for p: Rect2 in placed:
			if p.intersects(r):
				hit = true
				break
		if not hit:
			break
		r.position.y += label_size.y + 1.0
	return r


# ---------- vẽ ----------

func _on_canvas_draw() -> void:
	var font: Font = ThemeDB.fallback_font
	var tracking: bool = not _tracked.is_empty() or not _rear_tracked.is_empty() or _rear_active
	_draw_face_panel(font)
	# Camera trước
	_draw_header(font, front_header_rect(), "CAMERA TRƯỚC", _state, MUTED_COLOR)
	_draw_dets(font, _dets, front_image_rect(), _tracked, tracking)
	_draw_faces(_faces, front_image_rect())
	_canvas.draw_rect(front_image_rect(), Color(1, 1, 1, 0.15), false, 1.0)
	_draw_sectors(font, sector_rect())
	if _alert != "":
		var w: float = font.get_string_size(_alert, HORIZONTAL_ALIGNMENT_LEFT, -1, ALERT_FONT_SIZE).x
		var c: Vector2 = front_image_rect().get_center()
		_canvas.draw_rect(Rect2(c.x - w / 2.0 - 8.0, c.y - ALERT_FONT_SIZE, w + 16.0, ALERT_FONT_SIZE + 12.0), Color(0, 0, 0, 0.75))
		_canvas.draw_string(font, Vector2(c.x - w / 2.0, c.y), _alert, HORIZONTAL_ALIGNMENT_LEFT, -1, ALERT_FONT_SIZE, ALERT_COLOR)
	# Camera sau
	if _has_rear:
		var hr: Rect2 = rear_header_rect()
		_draw_header(font, hr, "CAMERA SAU", "đang dẫn" if _rear_active else "tắt · chỉ bật khi dẫn đường",
				OK_COLOR if _rear_active else MUTED_COLOR)
		if _rear_active:
			_draw_dets(font, _rear_dets, rear_image_rect(), _rear_tracked, true)
			_draw_faces(_rear_faces, rear_image_rect())
			_draw_rear_face_panel(font)
			_canvas.draw_rect(rear_image_rect(), Color(1, 1, 1, 0.15), false, 1.0)
	# Thanh dưới: trạng thái + kết nối
	var fr: Rect2 = footer_rect()
	_canvas.draw_rect(fr, PANEL_BG)
	var status: String = _status if _status != "" else "—"
	_canvas.draw_string(font, fr.position + Vector2(PAD, 17), ViewOverlay.ellipsize(font, status, FONT_SIZE, PANEL_W - 2.0 * PAD),
			HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE, STATUS_COLOR if _status != "" else MUTED_COLOR)
	var x: float = PAD
	var items: Array = [["YOLO", true, _connected]]
	if _stt_shown:
		items.append(["STT", true, _stt_connected])
	if _tts_shown:
		items.append(["TTS", true, _tts_connected])
	for it: Array in items:
		var col: Color = OK_COLOR if it[2] else BAD_COLOR
		_canvas.draw_circle(fr.position + Vector2(x + 4.0, 33.0), 4.0, col)
		_canvas.draw_string(font, fr.position + Vector2(x + 12.0, 37.0), it[0], HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_COLOR)
		x += 14.0 + font.get_string_size(it[0], HORIZONTAL_ALIGNMENT_LEFT, -1, 12).x + 14.0
	# Chú thích: box YOLO nét liền, box mặt (YuNet) nét đứt.
	var lx: float = fr.end.x - PAD - 118.0
	var ly: float = fr.position.y + 28.0
	_canvas.draw_rect(Rect2(lx, ly, 12.0, 10.0), BOX_COLOR_PATIENT, false, 1.5)
	_canvas.draw_string(font, Vector2(lx + 16.0, ly + 9.0), "vật thể", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED_COLOR)
	for seg: Array in ViewOverlay.rect_edges(Rect2(lx + 64.0, ly, 12.0, 10.0)):
		_canvas.draw_dashed_line(seg[0], seg[1], FACE_COLOR, 1.5, 3.0)
	_canvas.draw_string(font, Vector2(lx + 80.0, ly + 9.0), "mặt", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED_COLOR)


func _draw_header(font: Font, r: Rect2, title: String, right_text: String, right_color: Color) -> void:
	_canvas.draw_rect(r, HEADER_BG)
	_canvas.draw_string(font, r.position + Vector2(PAD, 16), title, HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE, TEXT_COLOR)
	if right_text == "":
		return
	var title_w: float = font.get_string_size(title, HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE).x
	var max_w: float = r.size.x - title_w - 3.0 * PAD
	var t: String = ViewOverlay.ellipsize(font, right_text, 12, max_w)
	var w: float = font.get_string_size(t, HORIZONTAL_ALIGNMENT_LEFT, -1, 12).x
	_canvas.draw_string(font, Vector2(r.end.x - PAD - w, r.position.y + 15), t, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, right_color)


func _draw_dets(font: Font, dets: Array, area: Rect2, tracked: Dictionary, tracking: bool) -> void:
	var placed: Array[Rect2] = []
	# Vật cản trước, người sau (người nằm trên cùng); bệnh nhân đang dẫn vẽ cuối.
	var order: Array = []
	for dv: Variant in dets:
		if not ViewOverlay.is_person(dv):
			order.append(dv)
	for dv: Variant in dets:
		if ViewOverlay.is_person(dv) and not is_same(dv, tracked):
			order.append(dv)
	if not tracked.is_empty():
		order.append(tracked)
	for dv: Variant in order:
		var d: Dictionary = dv
		var is_tracked: bool = is_same(d, tracked)
		var r: Rect2 = ViewOverlay.norm_to_pixel(d, area.size)
		r.position += area.position
		r = r.intersection(area)
		var col: Color = ViewOverlay.get_box_color(d)
		var width: float = 1.0
		if is_tracked:
			col = BOX_COLOR_TRACKED
			width = 3.0
		elif ViewOverlay.is_person(d):
			width = 2.0
			if tracking and int(d.get("class_id", -1)) == 0:
				col = BOX_COLOR_OTHER
		_canvas.draw_rect(r, col, false, width)
		var text: String = ViewOverlay.short_label(d, is_tracked, tracking)
		if text == "":
			continue
		var ts: Vector2 = font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, LABEL_FONT_SIZE)
		var lr: Rect2 = ViewOverlay.place_label(r, Vector2(ts.x + 8.0, LABEL_FONT_SIZE + 5.0), area, placed)
		placed.append(lr)
		_canvas.draw_rect(lr, Color(0, 0, 0, 0.7))
		_canvas.draw_rect(Rect2(lr.position, Vector2(3.0, lr.size.y)), col)
		_canvas.draw_string(font, lr.position + Vector2(5.0, LABEL_FONT_SIZE + 1.0), text, HORIZONTAL_ALIGNMENT_LEFT, -1,
				LABEL_FONT_SIZE, TEXT_COLOR)


func _draw_face_panel(font: Font) -> void:
	var r: Rect2 = face_panel_rect()
	_canvas.draw_rect(Rect2(r.position + Vector2(0, HEADER_H), Vector2(r.size.x, r.size.y - HEADER_H)), PANEL_BG)
	_draw_header(font, Rect2(r.position, Vector2(r.size.x, HEADER_H)), "MẶT", "", MUTED_COLOR)
	var y: float = r.position.y + HEADER_H + 4.0
	_canvas.draw_string(font, Vector2(r.position.x + PAD, y + 11.0), "ghi nhớ", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED_COLOR)
	var big := Rect2(r.position.x + PAD, y + 16.0, r.size.x - 2.0 * PAD, 116.0)
	_draw_thumb(font, big, _remembered_face, BOX_COLOR_TRACKED, "")
	y = big.end.y + 6.0
	_canvas.draw_string(font, Vector2(r.position.x + PAD, y + 11.0), "đang thấy (%d)" % _faces.size(), HORIZONTAL_ALIGNMENT_LEFT, -1,
			11, MUTED_COLOR)
	var tw: float = (r.size.x - 3.0 * PAD) / 2.0
	for i: int in LIVE_FACES_MAX:
		var tr := Rect2(r.position.x + PAD + i * (tw + PAD), y + 16.0, tw, minf(tw * 1.25, r.end.y - y - 20.0))
		var it: Dictionary = _live_faces[i] if i < _live_faces.size() else {}
		_draw_live_thumb(font, tr, it)


## Mặt đang thấy ở camera sau: viền xanh + "đang dắt" nếu mặt nằm trong box ReID khớp bệnh nhân, "khác" nếu không.
func _draw_rear_face_panel(font: Font) -> void:
	var r: Rect2 = rear_face_panel_rect()
	_canvas.draw_rect(Rect2(r.position + Vector2(0, HEADER_H), Vector2(r.size.x, r.size.y - HEADER_H)), PANEL_BG)
	_draw_header(font, Rect2(r.position, Vector2(r.size.x, HEADER_H)), "MẶT · SAU", "", MUTED_COLOR)
	var y: float = r.position.y + HEADER_H + 4.0
	_canvas.draw_string(font, Vector2(r.position.x + PAD, y + 11.0), "đang thấy (%d)" % _rear_faces.size(),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED_COLOR)
	var th: float = (r.end.y - (y + 16.0) - 3.0 * PAD) / 2.0
	for i: int in LIVE_FACES_MAX:
		var tr := Rect2(r.position.x + PAD, y + 16.0 + i * (th + PAD), r.size.x - 2.0 * PAD, th)
		_draw_live_thumb(font, tr, _rear_live_faces[i] if i < _rear_live_faces.size() else {})


func _draw_live_thumb(font: Font, r: Rect2, it: Dictionary) -> void:
	if it.is_empty():
		_draw_thumb(font, r, null, FACE_COLOR, "")
		return
	var mine: int = int(it.get("mine", -1))
	var col: Color = BOX_COLOR_TRACKED if mine == 1 else FACE_COLOR
	var tag: String = " · đang dắt" if mine == 1 else (" · khác" if mine == 0 else "")
	_draw_thumb(font, r, it.get("tex"), col, "%.0f%%%s" % [100.0 * float(it.get("conf", 0.0)), tag])


func _draw_thumb(font: Font, r: Rect2, tex: Texture2D, col: Color, caption: String) -> void:
	if tex == null:
		for seg: Array in ViewOverlay.rect_edges(r):
			_canvas.draw_dashed_line(seg[0], seg[1], Color(1, 1, 1, 0.25), 1.0, 3.0)
		_canvas.draw_string(font, r.get_center() + Vector2(-4, 4), "—", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, MUTED_COLOR)
		return
	# Giữ tỉ lệ ảnh: vừa khít trong khung, căn giữa.
	var ts: Vector2 = tex.get_size()
	var s: float = minf(r.size.x / ts.x, r.size.y / ts.y)
	var dst := Rect2(r.get_center() - ts * s * 0.5, ts * s)
	_canvas.draw_texture_rect(tex, dst, false)
	_canvas.draw_rect(dst, col, false, 1.5)
	if caption != "":
		_canvas.draw_string(font, dst.position + Vector2(3, dst.size.y - 3), caption, HORIZONTAL_ALIGNMENT_LEFT, -1, 10, TEXT_COLOR)


func _draw_faces(faces: Array, area: Rect2) -> void:
	for fv: Variant in faces:
		var r: Rect2 = ViewOverlay.norm_to_pixel(fv, area.size)
		r.position += area.position
		r = r.intersection(area)
		if r.size.x < 1.0 or r.size.y < 1.0:
			continue
		for seg: Array in ViewOverlay.rect_edges(r):
			_canvas.draw_dashed_line(seg[0], seg[1], FACE_COLOR, 1.5, FACE_DASH)


static func rect_edges(r: Rect2) -> Array:
	var a := r.position
	var b := Vector2(r.end.x, r.position.y)
	var c := r.end
	var d := Vector2(r.position.x, r.end.y)
	return [[a, b], [b, c], [c, d], [d, a]]


func _draw_sectors(font: Font, r: Rect2) -> void:
	_canvas.draw_rect(r, HEADER_BG)
	if not show_sector_hud:
		return
	var label_w: float = 46.0
	_canvas.draw_string(font, r.position + Vector2(PAD, 13), "vật cản", HORIZONTAL_ALIGNMENT_LEFT, -1, 10, MUTED_COLOR)
	var sec_w: float = (r.size.x - label_w - PAD) / 5.0
	for i: int in 5:
		var x: float = r.position.x + label_w + i * sec_w
		var v: float = clampf(float(_sectors[i]) if i < _sectors.size() else 0.0, 0.0, 1.0)
		var col := Color(0.25, 0.85, 0.35) if v < 0.35 else (Color(0.95, 0.8, 0.15) if v < 0.6 else Color(0.95, 0.25, 0.2))
		var bar := Rect2(x + 1.0, r.position.y + 4.0, sec_w - 2.0, r.size.y - 8.0)
		_canvas.draw_rect(bar, Color(0.22, 0.25, 0.3))
		if v > 0.01:
			_canvas.draw_rect(Rect2(bar.position, Vector2(bar.size.x * v, bar.size.y)), col)
		_canvas.draw_string(font, Vector2(x + 3.0, r.position.y + 13.0), SECTOR_LABELS[i], HORIZONTAL_ALIGNMENT_LEFT, -1, 9,
				Color(1, 1, 1, 0.85))
