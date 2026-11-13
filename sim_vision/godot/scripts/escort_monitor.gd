class_name EscortMonitor
extends RefCounted
## Giám sát khoảng cách robot – bệnh nhân khi dẫn đường (docs/UPGRADE_HRI_PLAN.md mục C). Thuần logic, không đụng
## scene: Main gọi step() mỗi frame, đọc `hold_robot` và lấy danh sách sự kiện để nói / chỉ tay.
##   OK ──(xa > FAR_M quá FAR_DWELL_S)──➔ FAR: nhắc, robot đứng chờ; nhắc lại mỗi REMIND_EVERY_S
##   FAR ──(gần lại ≤ OK_M)──➔ OK ("resumed")
##   FAR ──(xa > LOST_M, mất dấu quá LOST_UNSEEN_S, hoặc chờ quá ASK_AFTER_S)──➔ ASK: hỏi có dẫn tiếp không
##   ASK ──answer(intent)──➔ tiếp tục (OK) / chờ nghỉ (REST) / kết thúc (ENDED)
## Khoảng cách: hiện dùng vị trí thật trong sim; sau mục D sẽ thay bằng khoảng cách ước lượng từ camera.

enum State { OK, FAR, ASK, REST, ENDED }

const OK_M: float = 2.0
const FAR_M: float = 2.8
const FAR_DWELL_S: float = 1.5
const LOST_M: float = 5.0
const LOST_UNSEEN_S: float = 8.0
const REMIND_EVERY_S: float = 8.0
const ASK_AFTER_S: float = 20.0           # đứng chờ lâu vậy mà vẫn chưa theo kịp ➔ hỏi
const REST_WAIT_S: float = 20.0           # bệnh nhân xin dừng nghỉ ➔ chờ rồi hỏi lại
const REST_MIN_S: float = 4.0             # xin nghỉ ➔ đứng chờ ít nhất chừng này rồi mới mời đi tiếp (kể cả đang ở gần)
const ANSWER_WAIT_S: float = 15.0         # hỏi rồi mà quá lâu không có câu trả lời ➔ "no_answer" (coi như không nghe rõ)
const MAX_ASKS: int = 3                   # tổng số lần hỏi trong một ca; hết ➔ kết thúc
const CONTINUE_INTENTS: Array[String] = ["tiep_tuc", "dan_duong", "dieu_chinh_toc_do"]
const REST_INTENTS: Array[String] = ["dung_lai"]
const END_INTENTS: Array[String] = ["cam_on", "khong_ho_tro"]

var state: int = State.OK
var far_s: float = 0.0          # thời gian đã xa (đếm cả lúc chưa vào FAR)
var wait_s: float = 0.0         # thời gian đứng chờ trong FAR / REST
var unseen_s: float = 0.0
var since_remind_s: float = 0.0
var reminds: int = 0
var asks: int = 0
var unclear: int = 0
var total_wait_s: float = 0.0


func reset() -> void:
	state = State.OK
	far_s = 0.0
	wait_s = 0.0
	unseen_s = 0.0
	since_remind_s = 0.0
	reminds = 0
	asks = 0
	unclear = 0
	total_wait_s = 0.0


## Robot phải đứng chờ (không đi tiếp theo policy).
func hold_robot() -> bool:
	return state in [State.FAR, State.ASK, State.REST]


## Cập nhật; trả danh sách sự kiện: "remind", "remind_again", "resumed", "ask_continue", "no_answer", "give_up".
func step(dist_m: float, seen: bool, dt: float) -> Array[String]:
	var ev: Array[String] = []
	# Mất dấu chỉ tính khi đang xa: ở gần mà tia nhìn bị che (góc tường, người khác) vẫn là đang theo kịp.
	unseen_s = 0.0 if seen or dist_m <= FAR_M else unseen_s + dt
	if hold_robot():
		total_wait_s += dt
	match state:
		State.OK:
			far_s = far_s + dt if dist_m > FAR_M else 0.0
			if far_s >= FAR_DWELL_S:
				state = State.FAR
				wait_s = 0.0
				since_remind_s = 0.0
				reminds += 1
				ev.append("remind")
		State.FAR:
			wait_s += dt
			since_remind_s += dt
			if dist_m <= OK_M:
				_to_ok()
				ev.append("resumed")
			elif dist_m > LOST_M or unseen_s >= LOST_UNSEEN_S or wait_s >= ASK_AFTER_S:
				ev.append(_ask())
			elif since_remind_s >= REMIND_EVERY_S:
				since_remind_s = 0.0
				reminds += 1
				ev.append("remind_again")
		State.REST:
			wait_s += dt
			if dist_m <= OK_M and wait_s >= REST_MIN_S:
				_to_ok()
				ev.append("resumed")
			elif wait_s >= REST_WAIT_S:
				ev.append(_ask())
		State.ASK:
			# Chờ answer(). Không có câu trả lời (lượt nói hỏng, người không đáp) thì không được chờ mãi: báo
			# "no_answer" ➔ Main xử như câu không rõ (hỏi lại, lần hai thì kết thúc ca). Log run_1932: chờ 190 s.
			wait_s += dt
			if wait_s >= ANSWER_WAIT_S:
				wait_s = 0.0
				ev.append("no_answer")
	return ev


## Kết quả lượt hỏi: intent của câu trả lời ("" = không nghe được / lỗi STT).
## Trả "continue" / "rest" / "ask_again" / "end".
func answer(intent: String) -> String:
	if state != State.ASK:
		return ""
	if intent in CONTINUE_INTENTS:
		_to_ok()
		return "continue"
	if intent in END_INTENTS:
		state = State.ENDED
		return "end"
	if intent in REST_INTENTS and asks < MAX_ASKS:
		state = State.REST
		wait_s = 0.0
		return "rest"
	unclear += 1
	if unclear >= 2 or asks >= MAX_ASKS:
		state = State.ENDED
		return "end"
	asks += 1
	wait_s = 0.0
	return "ask_again"


func _ask() -> String:
	if asks >= MAX_ASKS:
		state = State.ENDED
		return "give_up"
	asks += 1
	state = State.ASK
	wait_s = 0.0
	return "ask_continue"


func _to_ok() -> void:
	state = State.OK
	far_s = 0.0
	wait_s = 0.0
	since_remind_s = 0.0
	unclear = 0
