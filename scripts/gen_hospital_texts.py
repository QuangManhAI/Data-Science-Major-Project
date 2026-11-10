"""Generate a hospital-domain Vietnamese sentence bank for synthetic TTS training data.

Sentences are composed from slot lists x templates per intent, and each template is
tagged with the speaker *register* it fits (how a child / young adult / adult / elder
addresses the robot). TTS voices are matched to registers later, so an "elderly"
voice says "Cháu ơi, ..." and a child voice says "Robot ơi, con ...".

Leakage control: the real recordings are reserved for testing, so every sentence
whose normalized form equals one of the 110+ real prompt texts (configs/voice_scripts.json)
is dropped. Near-paraphrases are allowed on purpose: the robot will hear the same
intents, the point of the real test set is unseen *voices + recording conditions*.

Output: data/processed/synth/texts.jsonl  {text_id, text, tts_text, intent, register, split}
  text      label as the ASR should output it (digits kept, like the real transcripts)
  tts_text  same sentence with numbers spelled out so the TTS reads them correctly
  split     ~8% of sentences go to `val` only, so val measures unseen text too
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_asr_testset import normalize, num_to_words  # noqa: E402

OUT = ROOT / "data" / "processed" / "synth" / "texts.jsonl"
rng = random.Random(1234)

# ----------------------------------------------------------------------------- slots
KHOA = ["tim mạch", "nhi", "sản", "tiêu hóa", "tai mũi họng", "răng hàm mặt", "mắt", "da liễu",
        "thần kinh", "hô hấp", "nội tiết", "thận", "tiết niệu", "cơ xương khớp", "chấn thương chỉnh hình",
        "ung bướu", "truyền nhiễm", "phục hồi chức năng", "y học cổ truyền", "dinh dưỡng", "lão khoa",
        "tâm thần", "huyết học", "nội tổng quát", "ngoại tổng quát", "gan mật", "phụ khoa", "nam khoa",
        "hồi sức tích cực", "dị ứng", "khám bệnh", "chẩn đoán hình ảnh", "xét nghiệm", "phẫu thuật thẩm mỹ"]
PHONG = ["phòng siêu âm", "phòng xét nghiệm máu", "phòng chụp X-quang", "phòng chụp CT",
         "phòng chụp cộng hưởng từ", "phòng nội soi", "phòng điện tim", "phòng điện não", "phòng tiêm",
         "phòng tiêm chủng", "phòng lấy máu", "phòng đo huyết áp", "phòng vật lý trị liệu", "phòng khám mắt",
         "phòng đo thị lực", "phòng đo loãng xương", "phòng siêu âm tim", "phòng siêu âm thai",
         "phòng nội soi dạ dày", "phòng nội soi đại tràng", "phòng thay băng", "phòng mổ", "phòng hành chính",
         "phòng công tác xã hội", "phòng trả kết quả", "phòng lưu bệnh", "phòng đo chức năng hô hấp",
         "phòng xét nghiệm nước tiểu", "phòng khám tiền mê", "phòng tư vấn dinh dưỡng"]
TIEN_ICH = ["nhà vệ sinh", "nhà vệ sinh nam", "nhà vệ sinh nữ", "thang máy", "cầu thang bộ", "căng tin",
            "chỗ bán đồ ăn", "chỗ uống nước", "máy bán nước", "chỗ ngồi chờ", "ghế chờ", "bãi gửi xe máy",
            "chỗ gửi ô tô", "cây ATM", "phòng cho con bú", "khu vui chơi trẻ em", "lối ra", "cổng chính",
            "cổng sau", "chỗ hút thuốc", "phòng chờ người nhà", "quầy thông tin", "chỗ sạc điện thoại",
            "quầy photocopy", "chỗ lấy nước nóng", "khu nhà ăn"]
QUAY = ["quầy đăng ký khám", "quầy tiếp đón", "quầy thu ngân", "quầy bảo hiểm", "quầy lấy số",
        "quầy thanh toán viện phí", "quầy hướng dẫn", "quầy ưu tiên", "quầy trả kết quả", "quầy nhập viện",
        "quầy xuất viện", "quầy tạm ứng"]
NHA_THUOC = ["nhà thuốc", "nhà thuốc bệnh viện", "quầy thuốc bảo hiểm", "quầy phát thuốc", "nhà thuốc số 2"]
TANG = ["tầng một", "tầng hai", "tầng ba", "tầng bốn", "tầng năm", "tầng trệt", "tầng hầm"]
KHU = ["khu A", "khu B", "khu C", "nhà A", "nhà B", "tòa nhà mới", "khu khám dịch vụ", "khu nội trú"]
TRIEU_CHUNG = ["đau bụng", "đau đầu", "sốt cao", "ho nhiều", "đau ngực", "khó thở", "đau lưng", "đau họng",
               "chóng mặt", "mờ mắt", "đau răng", "nổi mẩn ngứa", "đau đầu gối", "tê tay", "mất ngủ",
               "buồn nôn", "tiêu chảy", "đau tai", "ù tai", "đau dạ dày", "tức ngực", "phù chân", "đau khớp",
               "ho ra máu", "sổ mũi", "đau vai gáy", "tiểu buốt"]
NGUOI_NHA = {"child": ["mẹ", "bố", "bà", "ông", "em", "chị"],
             "young": ["bố mình", "mẹ mình", "bà mình", "ông mình", "bạn mình", "em mình", "bà em", "mẹ em"],
             "adult": ["vợ tôi", "chồng tôi", "con tôi", "mẹ tôi", "bố tôi", "bé nhà tôi", "ông nhà tôi"],
             "elder": ["ông nhà tôi", "bà nhà tôi", "cháu tôi", "con trai tôi", "con gái tôi", "bà ấy"]}

# How each register addresses the robot. `me`/`you` are drawn once per sentence and
# the openers/tails are built from them, so "Cháu ơi, cho bác hỏi ..." never turns
# into "Cháu ơi cho bác hỏi ... cô ...". Question tails only go on questions,
# statement tails only on requests/statements.
VOICE = {
    "child": dict(me=["con", "em", "cháu"], you=["robot", "bạn robot"],
                  call=["Robot ơi, ", "", "", "Ơ robot ơi, "], qtail=[" ạ", "", " ạ"], stail=[" ạ", "", " với ạ", " nhé"]),
    "young": dict(me=["mình", "em", "tớ"], you=["bạn", "robot"],
                  call=["", "", "Chào bạn, ", "Bạn ơi, ", "Robot ơi, "], qtail=["", " vậy", " ạ", " vậy bạn", " nhỉ", " thế"],
                  stail=["", " nha", " nhé", " với", " ạ"]),
    "adult": dict(me=["tôi", "chị", "anh"], you=["em", "robot", "bạn"],
                  call=["", "", "Em ơi, ", "Này em, ", "Robot ơi, "], qtail=["", " vậy em", " thế", " đấy", " hả em", " vậy"],
                  stail=["", " nhé", " nhé em", " với"]),
    "elder": dict(me=["bác", "tôi", "bà", "ông", "cô", "chú"], you=["cháu", "con"],
                  call=["", "", "Cháu ơi, ", "Này cháu, ", "Robot ơi, "], qtail=["", " cháu", " hả cháu", " thế cháu", " cháu nhỉ", " vậy cháu"],
                  stail=["", " cháu nhé", " nhé", " với cháu", " cháu ơi"]),
}
ASK = {"child": ["", "cho {me} hỏi "], "young": ["", "cho {me} hỏi ", "cho hỏi "],
       "adult": ["", "cho {me} hỏi ", "cho hỏi "], "elder": ["", "cho {me} hỏi "]}
WHERE = ["ở đâu", "ở chỗ nào", "đi đường nào", "đi lối nào", "nằm ở đâu", "ở tầng mấy", "ở khu nào", "đi hướng nào"]


def cap(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,?.!])", r"\1", s)
    s = re.sub(r"\b(\w+)( \1\b)+", r"\1", s)  # "với với" -> "với"
    return s[0].upper() + s[1:] if s else s


def end(s: str, q: bool = True) -> str:
    s = s.rstrip(" ,")
    return s + ("?" if q else ".")


def room_no() -> str:
    return str(rng.choice([rng.randint(101, 120), rng.randint(201, 230), rng.randint(301, 320),
                           rng.randint(401, 415), rng.randint(501, 510), rng.randint(1, 30)]))


# ------------------------------------------------------------------- templates per intent
# each template: f(register_dict) -> sentence (without final punctuation choice)
def T(reg: str):
    v = VOICE[reg]
    me, you = rng.choice(v["me"]), rng.choice(v["you"])
    oc = rng.choice(v["call"])                                   # call-out only ("Cháu ơi, ")
    o = oc + rng.choice(ASK[reg]).format(me=me)                  # call + optional "cho bác hỏi "
    tl, ts = rng.choice(v["qtail"]), rng.choice(v["stail"])       # question / statement tails
    nn = rng.choice(NGUOI_NHA[reg])
    k, p, ti, q = rng.choice(KHOA), rng.choice(PHONG), rng.choice(TIEN_ICH), rng.choice(QUAY)
    w, tc, tg, kh = rng.choice(WHERE), rng.choice(TRIEU_CHUNG), rng.choice(TANG), rng.choice(KHU)
    n = room_no()
    return {
        "hoi_khoa": [
            f"{o}khoa {k} {w}{tl}", f"{o}{me} muốn khám {k} thì đi đâu{tl}", f"khoa {k} {w}{tl}",
            f"{o}{me} bị {tc} thì nên khám khoa nào{tl}", f"{nn} bị {tc}, khám ở khoa nào{tl}",
            f"{o}khoa {k} có ở {tg} không{tl}", f"{o}khám {k} ở {kh} hay {rng.choice(KHU)}{tl}",
            f"{me} cần tìm khoa {k}{ts}", f"{o}{me} bị {tc} mấy hôm nay, khám ở đâu{tl}",
            f"{o}khoa {k} ở {tg} đúng không{tl}"],
        "hoi_phong": [
            f"{o}{p} {w}{tl}", f"{o}phòng {n} {w}{tl}", f"{o}phòng số {n} ở {tg} hay {rng.choice(TANG)}{tl}",
            f"{o}bác sĩ bảo {me} đi {p.replace('phòng ', '')}, phòng đó {w}{tl}", f"{p} là phòng mấy{tl}",
            f"phòng {n} là phòng gì{tl}", f"{o}lấy kết quả {rng.choice(['xét nghiệm', 'siêu âm', 'chụp phim', 'nội soi'])} ở đâu{tl}",
            f"{o}{me} cần nộp phiếu chỉ định ở phòng nào{tl}", f"{o}{p} có ở {kh} không{tl}",
            f"{o}{me} phải lên {p} trước hay đi đóng tiền trước{tl}"],
        "dan_duong": [
            f"{o}{you} dẫn {me} đến {p} được không{tl}", f"{oc}{you} đưa {me} lên khoa {k} với{ts}",
            f"dẫn {me} tới {q} giúp {me} với{ts}", f"{o}{you} dẫn {nn} đến {rng.choice(TIEN_ICH)} được không{tl}",
            f"{oc}{you} chỉ đường cho {me} tới khoa {k} với{ts}", f"{you} đi trước dẫn đường cho {me} nhé",
            f"{oc}đưa {me} ra {rng.choice(['cổng chính', 'lối ra', 'bãi gửi xe', 'quầy thu ngân'])} với{ts}",
            f"{oc}{me} đi theo {you} đến {p} nhé"],
        "dang_ky_kham": [
            f"{o}{q if 'đăng ký' in q or 'lấy số' in q else 'quầy đăng ký khám'} {w}{tl}",
            f"{o}{me} khám lần đầu thì làm thủ tục ở đâu{tl}", f"{o}lấy số thứ tự khám bệnh ở đâu{tl}",
            f"{o}{me} đặt lịch {rng.choice(['qua điện thoại', 'trên mạng', 'qua app', 'qua zalo'])} rồi thì đến quầy nào{tl}",
            f"{o}{me} đến tái khám thì đăng ký ở đâu{tl}", f"{o}có quầy ưu tiên cho {rng.choice(['người già', 'người khuyết tật', 'phụ nữ có thai', 'trẻ em'])} không{tl}",
            f"{o}{me} muốn khám dịch vụ thì đăng ký chỗ nào{tl}", f"{oc}{me} muốn đặt lịch khám {k}{ts}",
            f"{o}khám bệnh thì phải lấy số trước à{tl}"],
        "bao_hiem": [
            f"{o}{me} có thẻ bảo hiểm y tế thì nộp ở quầy nào{tl}", f"{o}{me} quên mang thẻ bảo hiểm rồi, làm sao{tl}",
            f"{o}dùng căn cước công dân thay thẻ bảo hiểm được không{tl}",
            f"{o}bảo hiểm có chi trả {rng.choice(['tiền thuốc', 'tiền chụp phim', 'tiền xét nghiệm', 'tiền giường'])} không{tl}",
            f"{o}thẻ bảo hiểm ở tỉnh khác có dùng được không{tl}", f"{oc}{me} muốn gia hạn thẻ bảo hiểm{ts}",
            f"{o}bảo hiểm trên VNeID có dùng được không{tl}", f"{o}khám trái tuyến thì bảo hiểm trả bao nhiêu{tl}"],
        "thanh_toan": [
            f"{o}{rng.choice(QUAY[2:6])} {w}{tl}", f"{o}đóng tiền viện phí ở đâu{tl}",
            f"{o}có thanh toán bằng {rng.choice(['chuyển khoản', 'thẻ ngân hàng', 'mã QR', 'ví điện tử', 'tiền mặt'])} được không{tl}",
            f"{o}{me} đóng tiền tạm ứng ở quầy nào{tl}", f"{o}trả tiền xét nghiệm ở chỗ nào{tl}",
            f"{o}{me} muốn lấy hóa đơn thì ra đâu{tl}", f"{o}đóng tiền xong thì đi đâu tiếp{tl}"],
        "nha_thuoc": [
            f"{o}{rng.choice(NHA_THUOC)} {w}{tl}", f"{o}lấy thuốc bảo hiểm ở đâu{tl}",
            f"{o}nhà thuốc mở cửa đến mấy giờ{tl}", f"{o}nhà thuốc có bán {rng.choice(['khẩu trang', 'sữa cho trẻ sơ sinh', 'nước muối', 'băng gạc', 'thuốc hạ sốt'])} không{tl}",
            f"{o}{me} cầm đơn thuốc thì ra đâu lấy{tl}", f"{o}mua thuốc ở đâu{tl}"],
        "tien_ich": [
            f"{o}{ti} {w}{tl}", f"{o}ở đây có {ti} không{tl}", f"{o}{ti} gần nhất ở đâu{tl}",
            f"{oc}{me} muốn đi vệ sinh{ts}", f"{oc}{me} đói quá, có chỗ ăn không{ts}",
            f"{o}có thang máy không, {nn} không leo cầu thang được{tl}", f"{o}{me} gửi xe máy ở đâu{tl}",
            f"{o}{me} muốn ngồi chờ thì ngồi đâu{tl}", f"{o}chỗ rút tiền ở đâu{tl}"],
        "ho_tro_di_chuyen": [
            f"{o}bệnh viện có cho mượn xe lăn không{tl}", f"{o}{nn} đi lại khó, có xe lăn không{tl}",
            f"{o}có xe đẩy cho em bé không{tl}", f"{oc}{me} cần cáng để đưa {nn} vào{ts}",
            f"{o}mượn nạng ở đâu{tl}", f"{o}có ai đẩy xe lăn giúp {me} không{tl}"],
        "dieu_chinh_toc_do": [
            f"{oc}đi chậm lại giúp {me}{ts}", f"{you} đi chậm thôi, {me} theo không kịp", f"{oc}đi nhanh lên một chút{ts}",
            f"{o}từ từ thôi, {nn} đi không nhanh được", f"chậm chậm thôi {you} ơi", f"{oc}{you} đi nhanh quá{ts}",
            f"{o}đi nhanh hơn được không{tl}", f"{you} đi chậm lại đi, {me} bị đau chân"],
        "dung_lai": [
            f"{oc}dừng lại một chút{ts}", f"khoan, đợi {me} chút", f"{oc}cho {me} nghỉ một lát{ts}",
            f"đứng lại đã, {nn} mệt rồi", f"{oc}chờ {me} một chút{ts}", f"{you} dừng lại đi", f"đợi đã, {me} quên đồ"],
        "tiep_tuc": [
            f"được rồi, đi tiếp đi", f"rồi, mình đi tiếp{ts}", f"ok, đi tiếp {you} ơi", f"đi tiếp đi {you}",
            f"{me} ổn rồi, đi tiếp thôi", f"ừ, đi tiếp", f"đi thôi {you}"],
        "nhac_lai": [
            f"{o}{you} nói lại được không{tl}", f"{me} chưa nghe rõ, nói lại giúp {me}{ts}",
            f"{you} nói chậm lại cho {me} nghe với", f"{o}{you} chỉ lại đường giúp {me} được không{tl}",
            f"nói lại lần nữa đi {you}", f"{me} không hiểu, {you} nói lại đi"],
        "dieu_chinh_am_luong": [
            f"{oc}nói to lên chút{ts}", f"{me} không nghe rõ, {you} nói to hơn được không", f"{you} nói nhỏ thôi",
            f"ở đây ồn quá, {you} nói to lên", f"{oc}giảm âm lượng xuống{ts}", f"{oc}to lên một chút{ts}"],
        "gap_nhan_vien": [
            f"{oc}gọi nhân viên giúp {me} với{ts}", f"{oc}{me} muốn gặp nhân viên hướng dẫn{ts}",
            f"{oc}{me} bị lạc {rng.choice(NGUOI_NHA['child'])} rồi{ts}", f"{oc}{me} không tìm thấy {nn}{ts}",
            f"{o}{me} đợi lâu quá rồi, hỏi ai bây giờ{tl}", f"{oc}cho {me} gặp bác sĩ trực{ts}",
            f"{oc}{me} muốn gặp người phụ trách{ts}", f"{oc}{me} muốn khiếu nại{ts}"],
        "thoi_gian_cho": [
            f"{o}bao lâu thì có kết quả{tl}", f"{o}còn bao nhiêu người nữa thì tới lượt {me}{tl}",
            f"{o}mấy giờ bệnh viện nghỉ trưa{tl}", f"{o}chờ khám {k} khoảng bao lâu{tl}",
            f"{o}buổi chiều mấy giờ thì khám lại{tl}", f"{o}bao giờ đến lượt {me}{tl}",
            f"{o}chủ nhật bệnh viện có làm việc không{tl}"],
        "cap_cuu": [
            f"{oc}{nn} khó thở quá, gọi bác sĩ giúp {me} với", f"{me} {rng.choice(['tức ngực', 'chóng mặt', 'đau bụng dữ dội', 'khó thở'])} quá, cứu {me} với",
            f"khoa cấp cứu ở đâu{tl}", f"{oc}{nn} bị ngất rồi, gọi người giúp với", f"{nn} đang co giật, gọi bác sĩ gấp",
            f"{nn} bị ngã chảy máu rồi, cứu với", f"cứu với, có người bị ngất", f"gọi cấp cứu nhanh lên"],
        "cam_on": [
            f"cảm ơn {you} nhé", f"{me} cảm ơn {you}{ts}", f"cảm ơn {you} nhiều", f"ok, cảm ơn nha",
            f"dạ cảm ơn {you} ạ", f"cảm ơn, {me} biết rồi", f"được rồi, cảm ơn {you}"],
        "khong_ho_tro": [
            f"{o}{me} phải nhịn ăn trước khi xét nghiệm không{tl}", f"{o}thuốc này uống trước hay sau ăn{tl}",
            f"{o}{me} có cần đeo khẩu trang không{tl}", f"{o}{nn} uống thuốc hạ sốt mấy lần một ngày{tl}",
            f"{o}bệnh của {me} có nặng không{tl}", f"{o}hôm nay thời tiết thế nào{tl}", f"{o}mấy giờ rồi{tl}",
            f"{o}{you} tên là gì{tl}", f"{o}{you} có biết hát không{tl}", f"{o}gần đây có quán cà phê nào không{tl}",
            f"{o}{me} có nên tiêm phòng cúm không{tl}"],
    }


# Register weights per intent (children don't ask about insurance or payment much)
REG_WEIGHTS = {"child": 0.18, "young": 0.27, "adult": 0.27, "elder": 0.28}
CHILD_SKIP = {"bao_hiem", "thanh_toan", "dang_ky_kham"}
FILLERS = ["Ờ, ", "À, ", "Ừm, ", "Dạ, "]
QWORDS = re.compile(r"\b(ở đâu|chỗ nào|đường nào|lối nào|hướng nào|khoa nào|quầy nào|phòng nào|khu nào|tầng mấy|phòng mấy|"
                    r"mấy giờ|mấy lần|bao lâu|bao nhiêu|bao giờ|là gì|tên là gì|thế nào|làm sao|không|đúng không|hay |à|hả)\b[^,]*$")
PER_INTENT = 140  # attempts per (intent, register); duplicates are collapsed


def to_tts(text: str) -> str:
    t = re.sub(r"\d+", lambda m: num_to_words(int(m.group())), text)
    return t.replace("X-quang", "ích quang").replace("CT", "xi ti").replace("ATM", "ây ti em").replace("QR", "quy a")


def main() -> None:
    real = json.loads((ROOT / "configs" / "voice_scripts.json").read_text(encoding="utf8"))["scripts"]
    real_norm = {normalize(it["text"]) for v in real.values() for it in v}
    intents = list(T("adult").keys())
    seen, rows, dropped = set(), [], 0
    for intent, reg in itertools.product(intents, VOICE):
        if reg == "child" and intent in CHILD_SKIP:
            continue
        n = int(PER_INTENT * REG_WEIGHTS[reg] / 0.25)
        for _ in range(n):
            s = rng.choice(T(reg)[intent])
            if rng.random() < 0.08:
                s = rng.choice(FILLERS) + s[0].lower() + s[1:]
            s = end(cap(s), q=bool(QWORDS.search(s)))
            key = normalize(s)
            if key in real_norm:
                dropped += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(text=s, tts_text=to_tts(s), intent=intent, register=reg))
    for r in rows:
        h = int(hashlib.md5(normalize(r["text"]).encode()).hexdigest(), 16)
        r["text_id"] = f"t{h % 10**8:08d}"
        r["split"] = "val" if h % 100 < 8 else "train"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(f"{len(rows)} unique sentences, dropped {dropped} identical to real prompts -> {OUT}")
    print(collections.Counter(r["register"] for r in rows))
    print(collections.Counter(r["intent"] for r in rows).most_common())
    print(collections.Counter(r["split"] for r in rows))
    for r in rng.sample(rows, 25):
        print(f"  [{r['register']:5s} {r['intent']:18s}] {r['text']}")


if __name__ == "__main__":
    main()
