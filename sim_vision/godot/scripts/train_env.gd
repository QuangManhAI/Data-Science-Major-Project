class_name TrainEnv
extends Node
## Train policy NGAY TRONG GODOT (`-- --controller=rl --train-env`, xem src/training/train_godot_v3.py).
## Godot là môi trường: mỗi bước policy (0.2 s giờ sim) gửi kèm `train` = sự kiện từ bước trước + trạng thái thật
## (vị trí người cần giúp, đích, khoảng cách bệnh nhân...) để trainer tính phần thưởng. Policy KHÔNG thấy phần này
## (quan sát vẫn dựng y như lúc chạy thật trong rl_policy_server.PolicyRunner).
## Hết ca (SimClock.shift_s giờ sim) ➔ bước cuối có done = true, nhận trả lời xong thì nạp lại scene (ca mới, seed mới).

const HARD_CAP_EXTRA_S: float = 90.0    # policy không chạy (kẹt ở luật giữ / thang máy) ➔ vẫn sang ca mới

var main: Node = null
var _events: Dictionary = {}
var _t0_s: float = 0.0
var _done_sent: bool = false
var _reloading: bool = false
var bump_frames: int = 0               # số physics frame robot cọ tường / đồ đạc kể từ bước trước


func setup(m: Node, policy: Node) -> void:
	main = m
	_t0_s = SimClock.now_s()
	policy.connect("action_received", _on_action)


func event(name: String, n: float = 1.0) -> void:
	_events[name] = float(_events.get(name, 0.0)) + n


func elapsed_s() -> float:
	return SimClock.now_s() - _t0_s


## Phần `train` của bước này (gọi đúng lúc gửi trạng thái cho policy); xoá sự kiện đã gửi.
func payload(gt: Dictionary) -> Dictionary:
	var out: Dictionary = {"ep": SimClock.episode, "t": elapsed_s(), "ev": _events, "gt": gt, "bump": bump_frames}
	_events = {}
	bump_frames = 0
	if elapsed_s() >= SimClock.shift_s and not _done_sent:
		_done_sent = true
		out["done"] = true
	return out


func _on_action(_v: float, _omega: float, _phase: int) -> void:
	if _done_sent:
		_next_episode()


func _process(_delta: float) -> void:
	if not _reloading and elapsed_s() > SimClock.shift_s + HARD_CAP_EXTRA_S:
		push_warning("[TrainEnv] ca %d quá giờ mà policy không chạy ➔ sang ca mới" % SimClock.episode)
		_next_episode()


func _next_episode() -> void:
	if _reloading:
		return
	_reloading = true
	print("[TrainEnv] hết ca %d (%.0fs sim, chờ server %.1fs thật)" % [SimClock.episode, elapsed_s(), SimClock.waited_ms_total / 1000.0])
	SimClock.episode += 1
	SimClock.new_episode()
	get_tree().call_deferred("reload_current_scene")
