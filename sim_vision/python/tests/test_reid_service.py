import json

import numpy as np
import pytest

from conftest import make_frame_msg
from identity.reid_service import ReidAugmenter, decode_emb, encode_emb
from protocol import parse_frame_opts


class FakeReid:
    def __init__(self):
        self.calls = 0

    def embed(self, crops):
        self.calls += 1
        out = np.zeros((len(crops), 512), np.float32)
        for i, c in enumerate(crops):
            out[i, i] = 1.0
            assert c.size > 0
        return out


class FakeFace:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, img):
        return [type("F", (), {"box": b, "score": 0.9})() for b in self.boxes]


def _dets():
    return [
        {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.9, "conf": 0.9, "class_id": 0, "class_name": "patient"},
        {"x1": 0.5, "y1": 0.1, "x2": 0.7, "y2": 0.9, "conf": 0.8, "class_id": 1, "class_name": "doctor"},
        {"x1": 0.8, "y1": 0.5, "x2": 0.9, "y2": 0.55, "conf": 0.7, "class_id": 0, "class_name": "patient"},  # quá nhỏ
    ]


def test_emb_roundtrip():
    v = np.random.default_rng(0).normal(size=512).astype(np.float32)
    assert np.allclose(decode_emb(encode_emb(v)), v)


def test_track_adds_emb_only_to_patients():
    img = np.zeros((480, 640, 3), np.uint8)
    dets = _dets()
    ReidAugmenter(FakeReid(), FakeFace([])).augment(img, dets, "track")
    assert "emb_b64" in dets[0] and decode_emb(dets[0]["emb_b64"]).shape == (512,)
    assert "emb_b64" not in dets[1], "bác sĩ không cần"
    assert "emb_b64" not in dets[2], "box quá nhỏ"
    assert "face" not in dets[0], "track không chạy YuNet"
    json.dumps(dets)


def test_enroll_marks_face_in_upper_half():
    img = np.zeros((480, 640, 3), np.uint8)
    dets = _dets()
    # mặt tâm (128, 80) nằm trong nửa trên box patient (x 64–192, y 48–432)
    ReidAugmenter(FakeReid(), FakeFace([(118.0, 70.0, 20.0, 20.0)])).augment(img, dets, "enroll")
    assert dets[0]["face"] is True
    dets = _dets()
    ReidAugmenter(FakeReid(), FakeFace([(118.0, 400.0, 20.0, 20.0)])).augment(img, dets, "enroll")
    assert dets[0]["face"] is False


def test_no_mode_or_unknown_mode_is_noop():
    reid = FakeReid()
    dets = _dets()
    for mode in ("", "abc"):
        ReidAugmenter(reid, FakeFace([])).augment(np.zeros((480, 640, 3), np.uint8), dets, mode)
    assert reid.calls == 0 and "emb_b64" not in dets[0]


@pytest.mark.parametrize("val,want", [("enroll", "enroll"), ("track", "track"), ("x", ""), (None, "")])
def test_parse_frame_opts_reid(val, want):
    msg = json.loads(make_frame_msg(1))
    if val is not None:
        msg["reid"] = val
    assert parse_frame_opts(json.dumps(msg))[2] == want


def test_detect_faces_normalised():
    img = np.zeros((480, 640, 3), np.uint8)
    faces = ReidAugmenter(FakeReid(), FakeFace([(64.0, 48.0, 32.0, 48.0), (630.0, 10.0, 40.0, 40.0)])).detect_faces(img)
    assert faces[0] == {"x1": 0.1, "y1": 0.1, "x2": 0.15, "y2": 0.2, "conf": 0.9}
    assert faces[1]["x2"] == 1.0, "kẹp vào khung ảnh"


def test_build_detections_faces_optional():
    from protocol import build_detections, wants_faces
    assert "faces" not in json.loads(build_detections(1, [], 3.0))
    assert json.loads(build_detections(1, [], 3.0, []))["faces"] == []
    msg = json.loads(make_frame_msg(1))
    assert not wants_faces(json.dumps(msg))
    msg["faces"] = True
    assert wants_faces(json.dumps(msg))
    assert not wants_faces("abc")
