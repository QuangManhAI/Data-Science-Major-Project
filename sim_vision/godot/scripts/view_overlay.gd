class_name ViewOverlay
extends Control
## Ô góc nhìn robot ở góc trên bên phải (G5): ảnh camera + box người + trạng thái.
## Box vẽ bằng `_draw` của node `Boxes` đặt trên `View`; ảnh gốc không bị sửa.

const BOX_COLOR := Color(0.1, 1.0, 0.2)
const TEXT_COLOR := Color(1, 1, 1)
const SHADOW_COLOR := Color(0, 0, 0, 0.8)
const FONT_SIZE: int = 13

var _dets: Array = []
var _state: String = ""
var _connected: bool = false

@onready var _view: TextureRect = $View
@onready var _boxes: Control = $Boxes


func _ready() -> void:
	_boxes.draw.connect(_on_boxes_draw)


func set_viewport(viewport: SubViewport) -> void:
	_view.texture = viewport.get_texture()


func show_detections(dets: Array, state: String, connected: bool) -> void:
	_dets = dets
	_state = state
	_connected = connected
	_boxes.queue_redraw()


## Box chuẩn hóa [0, 1] ➔ hình chữ nhật pixel trong ô kích thước `size`.
static func norm_to_pixel(det: Dictionary, size: Vector2) -> Rect2:
	var p1 := Vector2(float(det["x1"]) * size.x, float(det["y1"]) * size.y)
	var p2 := Vector2(float(det["x2"]) * size.x, float(det["y2"]) * size.y)
	return Rect2(p1, p2 - p1)


static func label_text(det: Dictionary) -> String:
	return "person %.2f" % float(det["conf"])


static func connection_text(connected: bool) -> String:
	return "YOLO: đã kết nối" if connected else "YOLO: mất kết nối"


func _on_boxes_draw() -> void:
	var font: Font = ThemeDB.fallback_font
	var size: Vector2 = _boxes.size
	for det: Variant in _dets:
		var r: Rect2 = ViewOverlay.norm_to_pixel(det as Dictionary, size)
		_boxes.draw_rect(r, BOX_COLOR, false, 2.0)
		_text(font, r.position + Vector2(2, -4) if r.position.y > FONT_SIZE + 2 else r.position + Vector2(2, FONT_SIZE + 2),
				ViewOverlay.label_text(det as Dictionary), BOX_COLOR)
	_text(font, Vector2(6, FONT_SIZE + 4), _state, TEXT_COLOR)
	_text(font, Vector2(6, size.y - 6), ViewOverlay.connection_text(_connected),
			TEXT_COLOR if _connected else Color(1, 0.35, 0.3))


func _text(font: Font, pos: Vector2, text: String, color: Color) -> void:
	_boxes.draw_string(font, pos + Vector2(1, 1), text, HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE, SHADOW_COLOR)
	_boxes.draw_string(font, pos, text, HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE, color)
