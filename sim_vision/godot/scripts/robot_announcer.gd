class_name RobotAnnouncer
extends Node
## Robot tự nói khi dẫn đường (docs/UPGRADE_HRI_PLAN.md mục B), tách khỏi lượt hội thoại của DialogueManager:
## - utt_id riêng (từ UTT_ID_BASE) ➔ DialogueManager bỏ qua audio của announcer và ngược lại;
## - không nói đè: đang hội thoại / robot đang nói thì xếp hàng (tối đa 1 câu chờ, câu mới thay câu cũ);
## - không có TTS ➔ chỉ phụ đề LINE_SUBTITLE_S giây.
## Câu thoại đọc từ res://data/escort_lines.json (cùng file với pregen TTS bên Python).

signal line_started(key: String)
signal line_finished(key: String)

const LINES_PATH: String = "res://data/escort_lines.json"
const UTT_ID_BASE: int = 1_000_000
const LINE_SUBTITLE_S: float = 3.5
const SPEAK_GRACE_S: float = 1.0
const SPEAKER: String = "Robot"

var tts: TtsClient = null
var voice: RobotVoice = null
var subtitle: Subtitle = null
## Trả true khi đang có lượt hội thoại (không được nói chen). Main gán.
var is_busy: Callable = func() -> bool: return false

var _lines: Dictionary = {}
var _places: Dictionary = {}
var _next_utt: int = UTT_ID_BASE
var _waiting_utt: int = -1
var _wait_deadline_ms: int = -1
var _current_key: String = ""
var _current_text: String = ""
var _speaking: bool = false
var _done_at_ms: int = -1          # chỉ phụ đề: hết giờ thì xong
var _queued: Dictionary = {}       # {key, text}


func _ready() -> void:
	load_lines(LINES_PATH)
	if tts != null:
		tts.audio_received.connect(_on_tts_audio)
		tts.error_received.connect(_on_tts_error)
	if voice != null:
		voice.finished_speaking.connect(_on_voice_finished)


func load_lines(path: String) -> void:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_warning("RobotAnnouncer: không đọc được %s" % path)
		return
	var d: Variant = JSON.parse_string(f.get_as_text())
	if d is Dictionary:
		_lines = (d as Dictionary).get("lines", {})
		_places = (d as Dictionary).get("places", {})


## Tên nơi đọc ra cho người nghe: số phòng ➔ "Phòng 205"; mã khác ➔ bảng places; không có ➔ "".
func place_name(key: String) -> String:
	return RobotAnnouncer.place_name_of(key, _places)


static func place_name_of(key: String, places: Dictionary) -> String:
	if key.is_valid_int():
		return "Phòng %s" % key
	return str(places.get(key, ""))


## Câu đã điền {place} / {floor}; key không có hoặc thiếu tên nơi ➔ "".
func line_text(key: String, args: Dictionary = {}) -> String:
	return RobotAnnouncer.format_line(_lines, key, args)


static func format_line(lines: Dictionary, key: String, args: Dictionary) -> String:
	if not lines.has(key):
		return ""
	var t: String = str(lines[key])
	for k: String in args:
		t = t.replace("{%s}" % k, str(args[k]))
	return "" if t.contains("{") else t


func is_speaking() -> bool:
	return _current_key != ""


func current_key() -> String:
	return _current_key


## Nói câu `key`. Đang bận ➔ xếp hàng (thay câu đang chờ). Trả false nếu không có câu.
func say(key: String, args: Dictionary = {}) -> bool:
	var text: String = line_text(key, args)
	if text == "":
		return false
	if is_speaking() or is_busy.call():
		_queued = {"key": key, "text": text}
		return true
	_start(key, text)
	return true


## Bỏ câu đang nói và câu đang chờ (vd. bắt đầu lượt hội thoại mới).
func cancel() -> void:
	_queued = {}
	if _current_key == "":
		return
	_waiting_utt = -1
	if _speaking and voice != null:
		_speaking = false
		voice.speaking = false
		voice.stop()
	_current_key = ""
	_done_at_ms = -1


func _start(key: String, text: String) -> void:
	_current_key = key
	_current_text = text
	_next_utt += 1
	line_started.emit(key)
	if tts != null and voice != null and tts.is_server_connected() and tts.send_speak(_next_utt, text, Config.ROBOT_VOICE):
		_waiting_utt = _next_utt
		_wait_deadline_ms = Time.get_ticks_msec() + int(Config.TTS_TIMEOUT_S * 1000.0)
		return
	_subtitle_only()


func _subtitle_only() -> void:
	_waiting_utt = -1
	_show(LINE_SUBTITLE_S)
	_done_at_ms = Time.get_ticks_msec() + int(LINE_SUBTITLE_S * 1000.0)


func _show(duration: float) -> void:
	if subtitle != null:
		subtitle.show_reply("", "%s: %s" % [SPEAKER, _current_text], duration)


func _on_tts_audio(a: Dictionary) -> void:
	if _waiting_utt < 0 or int(a["utt_id"]) != _waiting_utt:
		return
	_waiting_utt = -1
	if is_busy.call():                 # hội thoại vừa bắt đầu trong lúc chờ TTS ➔ bỏ câu
		_finish()
		return
	var duration: float = float(a["duration_s"])
	_speaking = true
	_done_at_ms = Time.get_ticks_msec() + int((duration + SPEAK_GRACE_S + 1.0) * 1000.0)   # chống treo
	voice.say(TtsClient.pcm16_to_stream(a["pcm"], a["sample_rate"]))
	_show(duration + SPEAK_GRACE_S)


func _on_tts_error(err_utt_id: int, _message: String) -> void:
	if err_utt_id == _waiting_utt:
		_subtitle_only()


func _on_voice_finished() -> void:
	if _speaking:
		_finish()


func _process(_delta: float) -> void:
	var now: int = Time.get_ticks_msec()
	if _waiting_utt >= 0 and now > _wait_deadline_ms:
		_subtitle_only()
	elif _current_key != "" and _done_at_ms >= 0 and now > _done_at_ms and _waiting_utt < 0:
		if _speaking and voice != null:
			voice.stop_speaking()      # gọi lại _on_voice_finished
		else:
			_finish()
	if _current_key == "" and not _queued.is_empty() and not is_busy.call():
		var q: Dictionary = _queued
		_queued = {}
		_start(q["key"], q["text"])


func _finish() -> void:
	var key: String = _current_key
	_speaking = false
	_current_key = ""
	_done_at_ms = -1
	if key != "":
		line_finished.emit(key)
