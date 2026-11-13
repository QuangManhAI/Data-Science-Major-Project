class_name VisionClient
extends Node
## Gửi ảnh camera robot tới YOLO server và nhận detection (G6, hợp đồng Mục 2.2).
## Chụp CAPTURE_HZ lần/giây khi đã kết nối; bỏ kết quả cũ (frame_id ≤ frame gần nhất)
## hoặc trễ hơn STALE_MS; mất kết nối thì thử lại mỗi RECONNECT_S.

signal detections_received(frame_id: int, dets: Array, infer_ms: float)
signal connection_changed(connected: bool)

const OUTBOUND_BUFFER: int = 1 << 22  # 4 MB; mặc định 64 KB nhỏ hơn 1 ảnh base64
## Nhận diện bệnh nhân (docs/UPGRADE_HRI_PLAN.md D3–D4): frame nào cũng có `reid` ➔ server gắn embedding OSNet vào box
## patient (MPS ~9 ms cho 1–3 người) ➔ ReID cùng tần số camera (30 Hz). Muốn nhẹ hơn thì tăng lên 2–3.
const REID_EVERY_N: int = 1
const EMB_DIM: int = 512

## Camera sau dùng client thứ hai: tắt khi không dẫn đường, chụp thưa hơn, frame nào cũng có `reid`.
var enabled: bool = true
## Xin server box mặt (YuNet, ~4 ms/frame) để hiển thị nét đứt trên ô camera. Kết quả frame mới nhất: last_faces.
var want_faces: bool = true
var last_faces: Array = []
var capture_hz: float = Config.CAPTURE_HZ
var reid_every_n: int = REID_EVERY_N

## "" = tắt, "enroll" = đang nói chuyện (ghi nhớ, có kiểm tra mặt), "track" = đang dẫn đường. Main đặt.
var reid_mode: String = ""

var url: String = Config.WS_URL

var _ws: WebSocketPeer = null
var _viewport: SubViewport = null
var _connected: bool = false
var _last_attempt_ms: int = -1
var _capture_accum: float = 0.0
var _next_id: int = 0
var _last_id: int = -1
var _sent_at: Dictionary = {}  # frame_id ➔ thời điểm gửi (ms)
var _sent_img: Dictionary = {}  # frame_id ➔ ảnh đã gửi (cắt mặt / box phải dùng đúng ảnh server đã thấy)
## Ảnh của frame có kết quả mới nhất (cùng frame với last_faces / dets). Robot đang quay ➔ ảnh viewport lúc kết quả về
## đã lệch hàng chục px so với frame này, cắt theo box sẽ ra tường.
var last_image: Image = null


func setup(viewport: SubViewport) -> void:
	_viewport = viewport


func is_server_connected() -> bool:
	return _connected


func _process(delta: float) -> void:
	var now: int = Time.get_ticks_msec()
	var state: int = WebSocketPeer.STATE_CLOSED
	if _ws != null:
		_ws.poll()
		state = _ws.get_ready_state()
	_set_connected(state == WebSocketPeer.STATE_OPEN)
	if state == WebSocketPeer.STATE_CLOSED \
			and (_last_attempt_ms < 0 or now - _last_attempt_ms >= int(Config.RECONNECT_S * 1000.0)):
		_try_connect(now)
	if _connected and _viewport != null and enabled:
		_capture_accum += delta
		var period: float = 1.0 / capture_hz
		if _capture_accum >= period:
			_capture_accum = fmod(_capture_accum, period)
			_send_frame(now)
	while _ws != null and _ws.get_available_packet_count() > 0:
		handle_message(_ws.get_packet().get_string_from_utf8(), Time.get_ticks_msec())
	_drop_expired(Time.get_ticks_msec())


func _set_connected(value: bool) -> void:
	if value != _connected:
		_connected = value
		_capture_accum = 0.0
		connection_changed.emit(value)


func _try_connect(now: int) -> void:
	_last_attempt_ms = now
	_ws = WebSocketPeer.new()
	_ws.outbound_buffer_size = OUTBOUND_BUFFER
	var err: Error = _ws.connect_to_url(url)
	if err != OK:
		push_warning("VisionClient: connect_to_url failed: %s" % error_string(err))


func _send_frame(now: int) -> void:
	if not _connected or _ws == null or _ws.get_ready_state() != WebSocketPeer.STATE_OPEN:
		_set_connected(false)
		return
	var img: Image = _viewport.get_texture().get_image()
	if img == null or img.is_empty():
		return
	var msg: Dictionary = {
		"type": "frame", "frame_id": _next_id, "width": img.get_width(), "height": img.get_height(),
		"jpeg_b64": Marshalls.raw_to_base64(img.save_jpg_to_buffer(Config.JPEG_QUALITY)),
	}
	if reid_mode != "" and _next_id % maxi(reid_every_n, 1) == 0:
		msg["reid"] = reid_mode
	if want_faces:
		msg["faces"] = true
	var err: Error = _ws.send_text(JSON.stringify(msg))
	if err != OK:
		_set_connected(false)
		return
	record_sent(_next_id, now)
	_sent_img[_next_id] = img
	_next_id += 1


func record_sent(frame_id: int, now_ms: int) -> void:
	_sent_at[frame_id] = now_ms


func pending_count() -> int:
	return _sent_at.size()


## Xử lý một message từ server; phát `detections_received` nếu hợp lệ, mới và không trễ.
func handle_message(text: String, now_ms: int) -> void:
	var r: Dictionary = VisionClient.parse_message(text)
	if not r["ok"]:
		push_warning("VisionClient: %s" % r["error"])
		return
	var frame_id: int = r["frame_id"]
	var sent: int = _sent_at.get(frame_id, -1)
	if not VisionClient.is_fresh(frame_id, _last_id, sent, now_ms):
		return
	_last_id = frame_id
	last_image = _sent_img.get(frame_id, null)
	_drop_expired(now_ms)
	last_faces = r["faces"]
	detections_received.emit(frame_id, r["dets"], r["infer_ms"])


## Dọn `_sent_at`: bỏ frame đã cũ hơn frame gần nhất, hoặc đã quá STALE_MS (không thể nhận nữa).
func _drop_expired(now_ms: int) -> void:
	for id: int in _sent_at.keys():
		if id <= _last_id or now_ms - int(_sent_at[id]) > Config.STALE_MS:
			_sent_at.erase(id)
	for id: int in _sent_img.keys():
		if not _sent_at.has(id):          # đã có kết quả (ảnh nằm ở last_image) / bị bỏ / quá hạn
			_sent_img.erase(id)


## Kết quả được nhận khi: frame_id mới hơn frame gần nhất, đã ghi thời điểm gửi, và trễ ≤ STALE_MS.
static func is_fresh(frame_id: int, last_id: int, sent_at_ms: int, now_ms: int) -> bool:
	if frame_id <= last_id or sent_at_ms < 0:
		return false
	return now_ms - sent_at_ms <= Config.STALE_MS


## Parse message server theo Mục 2.2.
## Trả {ok, frame_id, dets, infer_ms, error}; detection sai bị bỏ riêng từng cái.
static func parse_message(text: String) -> Dictionary:
	var out: Dictionary = {"ok": false, "frame_id": -1, "dets": [], "infer_ms": 0.0, "error": "", "faces": []}
	# JSON.parse_string in lỗi engine khi chuỗi hỏng; JSON.parse chỉ trả mã lỗi.
	var json := JSON.new()
	var msg: Variant = json.data if json.parse(text) == OK else null
	if not msg is Dictionary:
		out["error"] = "invalid json"
		return out
	var m := msg as Dictionary
	if _is_index(m.get("frame_id")):
		out["frame_id"] = int(m["frame_id"])
	if not m.has("type"):
		out["error"] = "missing field: type"
		return out
	if m["type"] == "error":
		out["error"] = "server error (frame %s): %s" % [m.get("frame_id"), m.get("message")]
		return out
	if m["type"] != "detections":
		out["error"] = "invalid field: type"
		return out
	for field: String in ["frame_id", "infer_ms", "detections"]:
		if not m.has(field):
			out["error"] = "missing field: %s" % field
			return out
	if not _is_index(m["frame_id"]):
		out["error"] = "invalid field: frame_id"
		return out
	if not _is_number(m["infer_ms"]) or float(m["infer_ms"]) < 0.0:
		out["error"] = "invalid field: infer_ms"
		return out
	if not m["detections"] is Array:
		out["error"] = "invalid field: detections"
		return out
	var dets: Array = []
	for d: Variant in m["detections"]:
		if VisionClient.is_valid_detection(d):
			var dd := d as Dictionary
			var item: Dictionary = {
				"x1": float(dd["x1"]),
				"y1": float(dd["y1"]),
				"x2": float(dd["x2"]),
				"y2": float(dd["y2"]),
				"conf": float(dd["conf"]),
			}
			if dd.has("class_id"):
				item["class_id"] = int(dd["class_id"])
			if dd.has("class_name"):
				item["class_name"] = str(dd["class_name"])
			if dd.get("emb_b64") is String:
				var emb: PackedFloat32Array = Marshalls.base64_to_raw(dd["emb_b64"]).to_float32_array()
				if emb.size() == EMB_DIM:
					item["emb"] = emb
			if dd.get("face") is bool:
				item["face"] = dd["face"]
			dets.append(item)
	var faces: Array = []
	if m.get("faces") is Array:
		for f: Variant in m["faces"]:
			if VisionClient.is_valid_detection(f):
				var ff := f as Dictionary
				faces.append({"x1": float(ff["x1"]), "y1": float(ff["y1"]), "x2": float(ff["x2"]), "y2": float(ff["y2"]),
						"conf": float(ff["conf"])})
	out["faces"] = faces
	out["ok"] = true
	out["dets"] = dets
	out["infer_ms"] = float(m["infer_ms"])
	return out


## Đủ 5 trường, là số, nằm trong [0, 1], x1 < x2, y1 < y2.
static func is_valid_detection(d: Variant) -> bool:
	if not d is Dictionary:
		return false
	var dd := d as Dictionary
	for k: String in ["x1", "y1", "x2", "y2", "conf"]:
		if not dd.has(k) or not _is_number(dd[k]):
			return false
		var v: float = float(dd[k])
		if v < 0.0 or v > 1.0:
			return false
	return float(dd["x1"]) < float(dd["x2"]) and float(dd["y1"]) < float(dd["y2"])


static func _is_number(v: Variant) -> bool:
	return (v is float or v is int) and is_finite(float(v))


## JSON của Godot đọc mọi số thành float ➔ số nguyên hợp lệ là số không có phần lẻ và ≥ 0.
static func _is_index(v: Variant) -> bool:
	return _is_number(v) and float(v) >= 0.0 and float(v) == floorf(float(v))
