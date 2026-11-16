extends Node
## Đồng hồ mô phỏng dùng chung (autoload "SimClock") — để train / đánh giá policy NGAY TRONG GODOT (docs/RL_V3_DESIGN.md,
## "Train trong Godot").
##
## - now_ms() / now_s(): thời gian mô phỏng (cộng dồn delta của frame). Chạy thường (thời gian thực) thì gần như bằng
##   Time.get_ticks_msec(); chạy `--fixed-fps N` không vsync (train) thì sim nhanh hơn thời gian thực mà mọi hẹn giờ
##   (hết lượt nghe STT, chờ TTS, nhắc bệnh nhân...) vẫn đúng theo giờ trong sim.
## - Lockstep (`--train-env` hoặc `--lockstep`): client nào gửi yêu cầu tới server (YOLO, STT, TTS, policy) thì hold();
##   đầu frame sau, SimClock đứng chờ (poll các client) tới khi mọi yêu cầu có trả lời ➔ sim không chạy tiếp trong lúc
##   server đang tính, giống như server trả lời ngay trong một frame.
## - Âm thanh: lockstep ➔ không phát ra loa; play() hẹn `finished` sau đúng độ dài clip theo giờ sim (giọng NPC, robot).

var lockstep: bool = false
## Train trong Godot: Main đọc để bật chế độ env (reset ca, gửi phần thưởng).
var train_env: bool = false
var episode: int = 0
var base_seed: int = 0
var shift_s: float = 600.0
var capture_hz: float = -1.0       # <0 = mặc định của VisionClient
var env_url: String = ""
var yolo_url: String = ""          # train nhiều bản Godot song song: mỗi bản một YOLO server riêng

const HOLD_TIMEOUT_MS: int = 30000   # server treo ➔ bỏ chờ (in cảnh báo) chứ không đứng hình mãi

var _t_us: int = 0
var _holds: Dictionary = {}          # key ➔ số yêu cầu đang chờ
var _pollers: Array[Callable] = []
var _audio: Dictionary = {}          # instance id ➔ [player, giờ sim kết thúc (us)]
var waited_ms_total: int = 0


func _init() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	process_priority = -100000       # chạy trước mọi node khác mỗi frame
	for arg: String in OS.get_cmdline_user_args():
		if arg == "--train-env":
			train_env = true
			lockstep = true
		elif arg == "--lockstep":
			lockstep = true
		elif arg.begins_with("--seed="):
			base_seed = int(arg.get_slice("=", 1))
		elif arg.begins_with("--shift-s="):
			shift_s = float(arg.get_slice("=", 1))
		elif arg.begins_with("--capture-hz="):
			capture_hz = float(arg.get_slice("=", 1))
		elif arg.begins_with("--env-url="):
			env_url = arg.get_slice("=", 1)
		elif arg.begins_with("--yolo-url="):
			yolo_url = arg.get_slice("=", 1)


func _ready() -> void:
	if lockstep:
		AudioServer.set_bus_mute(0, true)
		print("[SimClock] lockstep (train_env=%s, seed=%d, shift=%.0fs)" % [train_env, base_seed, shift_s])


func now_ms() -> int:
	return _t_us / 1000


func now_s() -> float:
	return _t_us / 1e6


func _process(delta: float) -> void:
	_t_us += int(round(delta * 1e6))
	if lockstep and not _holds.is_empty():
		_wait_replies()
	if not _audio.is_empty():
		_tick_audio()


## Yêu cầu tới server `key` đang chờ trả lời (chỉ có tác dụng khi lockstep).
func hold(key: String) -> void:
	if lockstep:
		_holds[key] = int(_holds.get(key, 0)) + 1


func release(key: String, all: bool = false) -> void:
	if not _holds.has(key):
		return
	var n: int = 0 if all else int(_holds[key]) - 1
	if n <= 0:
		_holds.erase(key)
	else:
		_holds[key] = n


func holding(key: String) -> bool:
	return _holds.has(key)


## Hàm poll (đọc gói tin WebSocket) của một client; gọi lặp trong lúc chờ.
func register_poller(fn: Callable) -> void:
	_pollers.append(fn)


func _wait_replies() -> void:
	var t0: int = Time.get_ticks_msec()
	while not _holds.is_empty():
		for i: int in range(_pollers.size() - 1, -1, -1):
			var fn: Callable = _pollers[i]
			if not fn.is_valid():
				_pollers.remove_at(i)
			else:
				fn.call()
		if _holds.is_empty():
			break
		if Time.get_ticks_msec() - t0 > HOLD_TIMEOUT_MS:
			push_warning("[SimClock] chờ server quá %d ms, bỏ chờ: %s" % [HOLD_TIMEOUT_MS, str(_holds)])
			_holds.clear()
			break
		OS.delay_usec(300)
	waited_ms_total += Time.get_ticks_msec() - t0


## Phát clip trên `player` (AudioStreamPlayer / 3D). Lockstep ➔ im lặng, `finished` phát sau đúng độ dài clip (giờ sim).
func play(player: Node) -> void:
	if not lockstep:
		player.call("play")
		return
	var stream: AudioStream = player.get("stream")
	var len_s: float = stream.get_length() if stream != null else 0.0
	_audio[player.get_instance_id()] = [player, _t_us + int(maxf(len_s, 0.05) * 1e6)]


func stop(player: Node) -> void:
	if not lockstep:
		player.call("stop")
		return
	_audio.erase(player.get_instance_id())


func _tick_audio() -> void:
	for id: int in _audio.keys():
		var e: Array = _audio[id]
		if not is_instance_valid(e[0]):
			_audio.erase(id)
		elif _t_us >= int(e[1]):
			_audio.erase(id)
			(e[0] as Node).emit_signal("finished")


## Reset cho ca mới (Main gọi khi nạp lại scene): bỏ hẹn giờ âm thanh / yêu cầu cũ.
func new_episode() -> void:
	_audio.clear()
	_holds.clear()
	_pollers.clear()
