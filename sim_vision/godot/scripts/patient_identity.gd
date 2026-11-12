class_name PatientIdentity
extends RefCounted
## Nhớ đúng một bệnh nhân robot đang nói chuyện / đang dẫn (docs/UPGRADE_HRI_PLAN.md D3–D4).
## Không có cơ sở dữ liệu người: mẫu (embedding OSNet 512-D từ yolo_server) chỉ nằm trong RAM của ca này,
## reset() khi bắt đầu hội thoại mới / kết thúc ca.
##   enroll_from(): lúc nói chuyện ➔ box patient to nhất, gần giữa khung, YuNet thấy mặt ➔ mẫu "trước".
##   match():       mỗi frame ➔ box giống nhất (≥ ACCEPT_SIM, hơn người thứ hai ≥ MARGIN), hoặc bám theo vị trí
##                  box frame trước (frame không có embedding / đang xoay người, CONT_*).
##   Tự thêm mẫu: khớp chắc (≥ ADD_SIM) hoặc đang bám liên tục mà chỉ có một bệnh nhân trong khung ➔ học thêm
##   góc nhìn mới (lưng, nghiêng). Đo ở sim_vision/reports/identity_eval.md: chỉ mẫu trước ➔ lúc dẫn (thấy lưng)
##   chỉ 62% frame đủ ngưỡng; thêm mẫu lưng ➔ 88%.

const ACCEPT_SIM: float = 0.75     # ~ phân vị 99% điểm người lạ giống nhất (0.78–0.79) trừ biên, cộng điều kiện MARGIN
const MARGIN: float = 0.05
## Ngưỡng theo phân bố đo trên ảnh Godot (OSNet x0.25): cùng người (cả trước ↔ sau) 95% cặp ≥ 0.67;
## người lạ: trung vị 0.54, phân vị 90% 0.66, 99% 0.77. Ngưỡng thấp hơn ➔ bám nhầm người lạ đứng gần rồi học luôn mẫu của họ.
const KEEP_SIM: float = 0.65       # đang bám liên tục (vị trí khớp frame trước) ➔ chỉ cần chừng này
const CONT_DIST: float = 0.12      # tâm box lệch ≤ 12% khung so với frame trước
const CONT_MAX_S: float = 0.6
const CONT_H_RATIO: Vector2 = Vector2(0.6, 1.6)
const ADD_SIM: float = 0.86        # người lạ: phân vị 99.9% = 0.84 ➔ học mẫu bằng ngoại hình phải cao hơn
const DUP_SIM: float = 0.92        # giống mẫu đã có hơn chừng này ➔ không thêm (không học thêm gì, chỉ đẩy mẫu cũ ra)
## Cự ly gần (< ~1.3 m): camera chỉ thấy nửa người, box bị cắt ➔ ReID điểm thấp. Box rất cao, ở giữa khung, là box
## bệnh nhân to duy nhất, và vừa thấy người mình ở gần ➔ vẫn nhận, ngưỡng thấp hơn.
const CLOSE_H: float = 0.8
const CLOSE_MAX_CX: float = 0.3
const CLOSE_SIM: float = 0.5
const CLOSE_RECENT_S: float = 6.0
const ADD_CONT_SIM: float = 0.72
const ADD_EVERY_S: float = 1.0
const MAX_TEMPLATES: int = 8
const MAX_ENROLL: int = 3
const ENROLL_EVERY_S: float = 0.3
const SPEAKER_CX_TOL: float = 0.15
const ENROLL_MAX_CX: float = 0.42  # |tâm box − giữa khung|: robot không luôn quay thẳng mặt về người đang nói

var templates: Array[PackedFloat32Array] = []
## Theo từng camera ("front" / "rear"): box khớp gần nhất {cx, cy, h} và thời điểm thấy. Mẫu dùng chung.
var last_box: Dictionary = {}
var last_seen_s: Dictionary = {}
var _last_add_s: float = -INF
var n_enroll: int = 0
## Lý do frame ghi nhớ gần nhất bị loại (chẩn đoán): no_patient / no_emb / no_face / off_center / "".
var last_reject: String = ""
## true ngay sau khi match() thay / thêm mẫu (kể cả khi đủ MAX_TEMPLATES nên số mẫu không đổi). Đọc xong tự tắt ở lần match sau.
var templates_changed: bool = false
var _last_enrolled: Dictionary = {}
var _last_any_seen_s: float = -INF
var _last_any_h: float = 0.0


func reset() -> void:
	templates.clear()
	last_box = {}
	last_seen_s = {}
	_last_add_s = -INF
	n_enroll = 0


func is_enrolled() -> bool:
	return not templates.is_empty()


static func is_patient(d: Dictionary) -> bool:
	return int(d.get("class_id", -1)) == 0 or str(d.get("class_name", "")) == "patient"


static func box_of(d: Dictionary) -> Dictionary:
	return {"cx": (float(d["x1"]) + float(d["x2"])) * 0.5, "cy": (float(d["y1"]) + float(d["y2"])) * 0.5,
			"h": float(d["y2"]) - float(d["y1"])}


static func cosine(a: PackedFloat32Array, b: PackedFloat32Array) -> float:
	var s: float = 0.0
	for i: int in mini(a.size(), b.size()):
		s += a[i] * b[i]
	return s


func score(emb: PackedFloat32Array) -> float:
	var best: float = -1.0
	for t: PackedFloat32Array in templates:
		best = maxf(best, PatientIdentity.cosine(emb, t))
	return best


## Lúc nói chuyện: ghi nhớ người đang nhìn robot. Trả true nếu thêm được một mẫu.
## expect_cx ≥ 0: hướng giọng nói chiếu vào ảnh (micro mảng / DOA) ➔ chỉ nhận box có tâm cách đó ≤ SPEAKER_CX_TOL
## (không nhầm người đứng cạnh); < 0: không biết hướng ➔ box gần giữa khung (ENROLL_MAX_CX).
## require_face = false chỉ để test; robot luôn yêu cầu thấy mặt.
func enroll_from(dets: Array, now_s: float, require_face: bool = true, expect_cx: float = -1.0) -> bool:
	if n_enroll >= MAX_ENROLL or now_s - _last_add_s < ENROLL_EVERY_S:
		return false
	var pick: Dictionary = {}
	last_reject = "no_patient"
	for dv: Variant in dets:
		var d: Dictionary = dv
		if not PatientIdentity.is_patient(d):
			continue
		if not d.has("emb"):
			last_reject = "no_emb" if last_reject == "no_patient" else last_reject
			continue
		if require_face and not bool(d.get("face", false)):
			last_reject = "no_face" if last_reject in ["no_patient", "no_emb"] else last_reject
			continue
		var b: Dictionary = PatientIdentity.box_of(d)
		var off: bool = absf(b["cx"] - expect_cx) > SPEAKER_CX_TOL if expect_cx >= 0.0 else absf(b["cx"] - 0.5) > ENROLL_MAX_CX
		if off:
			last_reject = "not_speaker_dir" if expect_cx >= 0.0 else "off_center"
			continue
		if pick.is_empty() or b["h"] > PatientIdentity.box_of(pick)["h"]:
			pick = d
	if pick.is_empty():
		return false
	last_reject = ""
	_last_enrolled = pick
	templates.append(pick["emb"])
	n_enroll += 1
	_last_add_s = now_s
	last_box["front"] = PatientIdentity.box_of(pick)
	last_seen_s["front"] = now_s
	return true


## Chọn box của người đã ghi nhớ trong frame này. Trả {det, score, how ("appearance" | "continuity")} hoặc {}.
## Box vừa được ghi nhớ ở lần enroll_from() thành công gần nhất (chẩn đoán).
static func box_of_last(p: PatientIdentity, _dets: Array) -> Dictionary:
	return p._last_enrolled


func match(dets: Array, now_s: float, cam: String = "front") -> Dictionary:
	templates_changed = false
	if templates.is_empty():
		return {}
	var cands: Array[Dictionary] = []
	for dv: Variant in dets:
		if PatientIdentity.is_patient(dv):
			cands.append(dv)
	var best: Dictionary = {}
	var best_s: float = -1.0
	var second_s: float = -1.0
	for d: Dictionary in cands:
		if not d.has("emb"):
			continue
		var s: float = score(d["emb"])
		if s > best_s:
			second_s = best_s
			best_s = s
			best = d
		elif s > second_s:
			second_s = s
	var res: Dictionary = {}
	if not best.is_empty() and best_s >= ACCEPT_SIM and best_s - second_s >= MARGIN:
		res = {"det": best, "score": best_s, "how": "appearance"}
	if res.is_empty():
		res = _match_continuity(cands, cam, now_s)
	if res.is_empty():
		res = _match_close(cands, now_s)
	if res.is_empty():
		return {}
	var det: Dictionary = res["det"]
	last_box[cam] = PatientIdentity.box_of(det)
	last_seen_s[cam] = now_s
	_last_any_seen_s = now_s
	_last_any_h = last_box[cam]["h"]
	_maybe_add(det, float(res["score"]), str(res["how"]), cands.size(), best_s - second_s, now_s)
	return res


## Box gần chỗ box của người mình ở frame trước (cùng camera, ≤ CONT_MAX_S trước), điểm ≥ KEEP_SIM nếu có embedding.
func _match_continuity(cands: Array[Dictionary], cam: String, now_s: float) -> Dictionary:
	if not last_box.has(cam) or now_s - float(last_seen_s.get(cam, -INF)) > CONT_MAX_S:
		return {}
	var prev: Dictionary = last_box[cam]
	var near: Dictionary = {}
	var near_d: float = INF
	for d: Dictionary in cands:
		var b: Dictionary = PatientIdentity.box_of(d)
		var dist: float = Vector2(b["cx"] - prev["cx"], b["cy"] - prev["cy"]).length()
		var ratio: float = b["h"] / maxf(prev["h"], 1e-3)
		if dist <= CONT_DIST and ratio >= CONT_H_RATIO.x and ratio <= CONT_H_RATIO.y and dist < near_d:
			near = d
			near_d = dist
	if near.is_empty():
		return {}
	var s2: float = score(near["emb"]) if near.has("emb") else -1.0
	if near.has("emb") and s2 < KEEP_SIM:
		return {}
	return {"det": near, "score": s2, "how": "continuity"}


## Người mình vừa ở gần (box cao ≥ 0.45 trong CLOSE_RECENT_S) và giờ có đúng một box rất cao giữa khung ➔ là họ, sát robot.
func _match_close(cands: Array[Dictionary], now_s: float) -> Dictionary:
	if now_s - _last_any_seen_s > CLOSE_RECENT_S or _last_any_h < 0.45:
		return {}
	var close: Array[Dictionary] = []
	for d: Dictionary in cands:
		if PatientIdentity.box_of(d)["h"] >= CLOSE_H:
			close.append(d)
	if close.size() != 1 or absf(PatientIdentity.box_of(close[0])["cx"] - 0.5) > CLOSE_MAX_CX:
		return {}
	var sc: float = score(close[0]["emb"]) if close[0].has("emb") else CLOSE_SIM
	return {"det": close[0], "score": sc, "how": "close"} if sc >= CLOSE_SIM else {}


func _maybe_add(det: Dictionary, s: float, how: String, n_cands: int, margin: float, now_s: float) -> void:
	if not det.has("emb") or now_s - _last_add_s < ADD_EVERY_S:
		return
	# Chỉ học khi trong khung không có bệnh nhân nào khác (tránh học nhầm mẫu người đứng cạnh).
	var ok: bool = n_cands == 1 and ((how == "appearance" and s >= ADD_SIM) or (how == "continuity" and s >= ADD_CONT_SIM))
	if not ok or s >= DUP_SIM:
		return
	if templates.size() >= MAX_TEMPLATES:
		templates.remove_at(n_enroll if n_enroll < templates.size() else 1)   # giữ mẫu lúc nói chuyện
	templates.append(det["emb"])
	templates_changed = true
	_last_add_s = now_s
