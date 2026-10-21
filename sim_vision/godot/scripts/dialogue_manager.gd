class_name DialogueManager
extends Node
## Điều phối một lượt hội thoại (G16), con trực tiếp của Main:
## NPC TALK ➔ robot LISTEN; NPC nói xong ➔ gửi utterance, robot THINK; nhận transcript ➔ RESPOND + phụ đề;
## hết phụ đề ➔ ARRIVED, npc.on_reply_shown(). Timeout / lỗi / mất kết nối ➔ câu xin lỗi, về ARRIVED ngay.

## Phát sau mỗi lượt (kể cả lỗi), cùng nội dung dòng log; G17 dùng.
signal turn_completed(row: Dictionary)

@export var snr_db: float = Config.DEFAULT_SNR_DB
@export var noise: String = "babble"
@export var npc_path: NodePath = ^"../Human/Npc"
@export var stt_path: NodePath = ^"../SttClient"
@export var subtitle_path: NodePath = ^"../UI/Subtitle"

var utt_id: int = 0                 # utt_id đang chờ (tăng mỗi lượt, không reset)
var deadline_ms: int = -1           # −1 = không chờ
var pending_clip: String = ""
var sent_distance_m: float = 0.0

var _start_ms: int = 0

@onready var main: Main = get_parent() as Main
@onready var npc: Npc = get_node(npc_path)
@onready var stt: SttClient = get_node(stt_path)
@onready var subtitle: Subtitle = get_node(subtitle_path)
@onready var logger: DialogueLogger = get_node_or_null(^"../DialogueLogger")


func _ready() -> void:
	_start_ms = Time.get_ticks_msec()
	# Main chưa _ready (con _ready trước cha) ➔ lấy Robot bằng get_node, đọc main.state lúc gọi.
	npc.setup(main.get_node(^"Robot"), func() -> int: return main.state)
	npc.talk_started.connect(_on_talk_started)
	npc.talk_aborted.connect(_on_talk_aborted)
	npc.utterance_finished.connect(_on_utterance_finished)
	stt.transcript_received.connect(_on_transcript)
	stt.error_received.connect(_on_stt_error)
	stt.connection_changed.connect(_on_stt_connection_changed)
	subtitle.finished.connect(_on_subtitle_finished)
	# Hiện dòng STT ngay từ đầu (chưa nối được lần nào thì cũng không có sự kiện connection_changed).
	(main.get_node(^"UI/ViewOverlay") as ViewOverlay).set_stt_connected(stt.is_server_connected())
	if logger != null:
		logger.open()


func _process(_delta: float) -> void:
	if main.state == Config.RobotState.THINK and DialogueManager.is_timed_out(deadline_ms, Time.get_ticks_msec()):
		_fail("timeout")


## Đặt lại khi Main đặt lại vị trí (phím R). utt_id giữ nguyên để transcript muộn của lượt cũ bị bỏ.
func reset() -> void:
	deadline_ms = -1
	pending_clip = ""
	if subtitle != null:
		subtitle.hide_now()


static func is_timed_out(deadline: int, now_ms: int) -> bool:
	return deadline >= 0 and now_ms > deadline


## Nhận transcript khi đúng utt_id đang chờ và robot đang THINK.
static func accepts(r: Dictionary, waiting_utt_id: int, robot_state: int) -> bool:
	return robot_state == Config.RobotState.THINK and int(r["utt_id"]) == waiting_utt_id


static func clip_speaker(clip_id: String) -> String:
	return clip_id.get_slice("_", 0)


static func clip_take(clip_id: String) -> String:
	var parts: PackedStringArray = clip_id.split("_")
	return parts[parts.size() - 1] if parts.size() >= 3 else ""


func build_log_row(r: Dictionary, error: String) -> Dictionary:
	var row: Dictionary = {
		"time_s": (Time.get_ticks_msec() - _start_ms) / 1000.0, "utt_id": utt_id, "clip_id": pending_clip,
		"speaker": DialogueManager.clip_speaker(pending_clip), "take": DialogueManager.clip_take(pending_clip),
		"distance_m": sent_distance_m, "snr_db": snr_db, "noise": noise, "error": error,
	}
	if not r.is_empty():
		row["effective_snr_db"] = r["effective_snr_db"]
		row["hyp_text"] = r["text"]
		row["intent_pred"] = r["intent"]
		row["intent_conf"] = r["intent_conf"]
		row["stt_ms"] = r["stt_ms"]
	return row


func _on_talk_started(_clip_id: String) -> void:
	if main.state == Config.RobotState.ARRIVED:
		main.set_dialogue_state(Config.RobotState.LISTEN)


func _on_talk_aborted(_clip_id: String) -> void:
	if main.state == Config.RobotState.LISTEN:
		main.set_dialogue_state(Config.RobotState.ARRIVED)


func _on_utterance_finished(clip_id: String) -> void:
	utt_id += 1
	pending_clip = clip_id
	sent_distance_m = Human.flat_distance(main.get_node(^"Robot").global_position, npc.human.global_position)
	main.set_dialogue_state(Config.RobotState.THINK)
	deadline_ms = Time.get_ticks_msec() + int(Config.STT_TIMEOUT_S * 1000.0)
	if not stt.send_utterance(utt_id, clip_id, sent_distance_m, snr_db, noise):
		_fail("stt disconnected")


func _on_transcript(r: Dictionary) -> void:
	if not DialogueManager.accepts(r, utt_id, main.state):
		return
	deadline_ms = -1
	main.set_dialogue_state(Config.RobotState.RESPOND)
	var response: String = r["response"] if r["response"] != "" else Subtitle.SORRY
	subtitle.show_reply(r["text"], response, Config.SUBTITLE_S)
	_log(build_log_row(r, ""))


func _on_stt_error(err_utt_id: int, message: String) -> void:
	if err_utt_id == utt_id and main.state == Config.RobotState.THINK:
		_fail("server: " + message)


func _on_stt_connection_changed(connected: bool) -> void:
	main.overlay.set_stt_connected(connected)
	if not connected and main.state == Config.RobotState.THINK:
		_fail("stt disconnected")


func _on_subtitle_finished() -> void:
	if main.state == Config.RobotState.RESPOND:
		_finish()


## Câu xin lỗi, ghi lỗi, về ARRIVED ngay (phụ đề vẫn hiện hết SUBTITLE_S).
func _fail(error: String) -> void:
	deadline_ms = -1
	subtitle.show_sorry(Config.SUBTITLE_S)
	_log(build_log_row({}, error))
	_finish()


func _finish() -> void:
	main.set_dialogue_state(Config.RobotState.ARRIVED)
	npc.on_reply_shown()


func _log(row: Dictionary) -> void:
	if logger != null:
		logger.log_row(row)
	turn_completed.emit(row)
