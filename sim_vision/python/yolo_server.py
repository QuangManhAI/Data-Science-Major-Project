"""YOLO server WebSocket (G2). Nhận `frame`, trả `detections` hoặc `error` (Mục 2.2).

Chạy: python yolo_server.py --model yolov8n.pt [--port 8765]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from detector import DEFAULT_YOLO_PATH, Detector, YoloDetector, resolve_model_path
from protocol import WS_HOST, WS_PORT, ProtocolError, build_detections, build_error, parse_frame_opts, wants_faces

log = logging.getLogger("yolo_server")


# Camera trước và camera sau là hai kết nối riêng nhưng dùng chung một model ➔ chạy lần lượt.
_MODEL_LOCKS: dict[int, asyncio.Lock] = {}


def _model_lock() -> asyncio.Lock:
    """Một khóa cho mỗi event loop (test tạo loop mới mỗi lần)."""
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _MODEL_LOCKS:
        _MODEL_LOCKS.clear()
        _MODEL_LOCKS[loop_id] = asyncio.Lock()
    return _MODEL_LOCKS[loop_id]


async def process_message(text: str | bytes, detector: Detector, reid=None) -> str:
    """Xử lý một message, luôn trả chuỗi JSON để gửi lại (không ném lỗi giao thức).
    Frame có `reid` và server có ReidAugmenter ➔ box patient có thêm emb_b64 (+ face khi enroll)."""
    try:
        frame_id, img, reid_mode = parse_frame_opts(text)
    except ProtocolError as e:
        log.warning("frame_id=%s error=%s", e.frame_id, e.message)
        return build_error(e.frame_id, e.message)
    faces = None
    async with _model_lock():
        dets, infer_ms = await asyncio.to_thread(detector.detect, img)
        if reid is not None and (wants_faces(text) or reid_mode == "enroll"):
            faces = await asyncio.to_thread(reid.detect_faces, img)
        if reid_mode and reid is not None:
            await asyncio.to_thread(reid.augment, img, dets, reid_mode, faces)
    log.info("frame_id=%d n_det=%d infer_ms=%.1f reid=%s faces=%s", frame_id, len(dets), infer_ms, reid_mode or "-",
             "-" if faces is None else len(faces))
    return build_detections(frame_id, dets, infer_ms, faces if wants_faces(text) else None)


async def handle_connection(ws: ServerConnection, detector: Detector, reid=None) -> None:
    """Đọc liên tục; chỉ giữ message mới nhất chưa xử lý, các message chờ cũ hơn bị bỏ."""
    latest: list[str | bytes] = []  # tối đa 1 phần tử
    has_new = asyncio.Event()
    closed = asyncio.Event()

    async def reader() -> None:
        try:
            async for msg in ws:
                if latest:
                    log.debug("dropped stale pending message")
                latest[:] = [msg]
                has_new.set()
        except ConnectionClosed:
            pass
        finally:
            closed.set()
            has_new.set()

    reader_task = asyncio.create_task(reader())
    try:
        while True:
            await has_new.wait()
            has_new.clear()
            if not latest:
                if closed.is_set():
                    break
                continue
            msg = latest.pop()
            reply = await process_message(msg, detector, reid)
            try:
                await ws.send(reply)
            except ConnectionClosed:
                break
    finally:
        reader_task.cancel()


async def start_server(detector: Detector, host: str = WS_HOST, port: int = WS_PORT, reid=None) -> Server:
    return await serve(lambda ws: handle_connection(ws, detector, reid), host, port, max_size=2**24)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO 17-class hospital detection WebSocket server")
    p.add_argument("--model", default=DEFAULT_YOLO_PATH, help="đường dẫn model YOLO (mặc định: YOLOv6 17-class)")
    p.add_argument("--port", type=int, default=WS_PORT)
    p.add_argument("--no-reid", action="store_true", help="tắt nhận diện bệnh nhân (OSNet + YuNet) cho frame có `reid`")
    return p.parse_args(argv)


async def _serve_forever(detector: Detector, port: int, reid=None) -> None:
    server = await start_server(detector, WS_HOST, port, reid)
    log.info("listening on ws://%s:%d", WS_HOST, port)
    await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_path = resolve_model_path(args.model)
    if not os.path.isfile(model_path):
        print(f"ERROR: model file not found: {args.model} (resolved: {model_path})", file=sys.stderr)
        return 1

    detector = YoloDetector(model_path)
    log.info("loaded model %s on %s", model_path, detector.device)
    try:
        reid = None
        if not args.no_reid:
            from identity.reid_service import ReidAugmenter

            reid = ReidAugmenter()
            reid.preload()
        asyncio.run(_serve_forever(detector, args.port, reid))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
