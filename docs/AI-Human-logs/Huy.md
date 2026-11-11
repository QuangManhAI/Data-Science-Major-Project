# Huy Decision Log & Vision Onboarding Audit

- **Motivation/Background**: Ghi nhận toàn bộ kết quả khảo sát hệ thống, phân tích kiến trúc đa tầng và các quyết định kỹ thuật của Huy khi tiếp nhận vai trò phụ trách Module Thị giác (Vision & Perception) trong đồ án Robot trợ lý dẫn đường bệnh viện.
- **Purpose**: Đóng vai trò là tài liệu gốc cho Báo cáo Giai đoạn 1 (Đọc & Khảo sát Full Dự án) và nhật ký quyết định kỹ thuật của nhánh Vision.
- **Overview Pipeline**: Tiến hành rà soát tĩnh và động toàn bộ cấu trúc repo `Data-Science-Major-Project`: mô phỏng 3D Godot, giao thức WebSocket đa server, mạng học tăng cường PPO RL, pipeline giọng nói STT/NLU/TTS và mã nguồn Vision hiện tại.
- **Detailed Plan**: 
 §1 Quyết định kỹ thuật (Decision Index & D-045); 
 §2 Bản sắc & Mục tiêu cốt lõi; 
 §3 Kiến trúc hệ thống đa tầng & Hợp đồng giao tiếp; 
 §4 Bản đồ thư mục & Chức năng từng module; 
 §5 Hiện trạng & Khoảng trống của Module Vision.
- **References**: [agents/rules/MD_CONVENTION.md](../../agents/rules/MD_CONVENTION.md), [docs/AI-Human-logs/Vuong.md](Vuong.md), [docs/Đề cương đồ án Robot trợ lý dẫn đường bệnh viện.md](../%C4%90%E1%BB%81%20c%C6%B0%C6%A1ng%20%C4%91%E1%BB%93%20%C3%A1n%20Robot%20tr%E1%BB%A3%20l%C3%BD%20d%E1%BA%ABn%20%C4%91%C6%B0%E1%BB%9Dng%20b%E1%BB%87nh%20vi%E1%BB%87n.md), [docs/progress/huy/khao_sat_vision/0001.md](../progress/huy/khao_sat_vision/0001.md), [sim_vision/README.md](../../sim_vision/README.md).
- **Created**: 2026-11-09T23:57:00+07:00
- **Last Updated**: 2026-11-10T15:58:00+07:00

---

## 1. Danh mục quyết định kỹ thuật (Decision Index)

| ID | Tiêu đề | Phụ trách | Trạng thái |
| :--- | :--- | :--- | :---: |
| [D-045](#d-045--tiếp-nhận-module-vision-và-hoàn-thành-khảo-sát-toàn-bộ-kiến-trúc-dự-án-giai-đoạn-1) | Tiếp nhận Module Vision và hoàn thành khảo sát toàn bộ kiến trúc dự án (Giai đoạn 1) | Huy (Human) + Antigravity AI | Accepted |

---

### D-045 — Tiếp nhận Module Vision và hoàn thành khảo sát toàn bộ kiến trúc dự án (Giai đoạn 1)
- **Trạng thái:** Accepted
- **Thời gian:** 2026-11-09T23:57:00+07:00
- **Phụ trách:** Huy (Human) + Antigravity AI
- **Bối cảnh:** Huy tiếp nhận phụ trách Module Thị giác (Vision) cho dự án `Data-Science-Major-Project`. Dự án đã có sẵn nhiều module phức tạp từ các thành viên trước (Godot 3D, YOLO server, PhoWhisper STT, PhoBERT NLU, VieNeu-TTS, RL PPO), cần một bản khảo sát toàn diện (Giai đoạn 1) để nắm bắt kiến trúc, hợp đồng dữ liệu và xác định các khoảng trống kỹ thuật của module Vision.
- **Quyết định:**
  1. Thống nhất lộ trình 4 giai đoạn làm chủ module Vision: Giai đoạn 1 (Đọc full dự án), 
  Giai đoạn 2 (Mổ xẻ module Vision), 
  Giai đoạn 3 (Nắm flow code & hợp đồng giao tiếp), 
  Giai đoạn 4 (Chiến lược Data Science, phân lớp ca khó/dễ, tracking & Re-ID).
  2. Xác lập ranh giới hiện tại: Pipeline YOLOv8n hiện tại chỉ đóng vai trò bộ khung kết nối (Scaffolding) phát hiện `person` cơ bản cho FSM lái robot; toàn bộ bài toán dữ liệu bệnh viện (xe lăn, người chống nạng, góc quay thấp, che khuất, Re-ID) chưa được xây dựng và sẽ là trọng tâm phát triển của Huy.
  3. Lưu trữ toàn bộ kết quả khảo sát chi tiết vào `docs/AI-Human-logs/Huy.md` và `docs/progress/huy/khao_sat_vision/0001.md` tuân thủ chuẩn `MD_CONVENTION.md`.
- **Lý do:** Đảm bảo tính minh bạch, khả năng truy vết kiến trúc, tránh hiểu lầm phạm vi và ngăn ngừa sự trôi dạt kỹ thuật (technical drift).
- **File liên quan:** `docs/AI-Human-logs/Huy.md`, `docs/progress/huy/khao_sat_vision/0001.md`, `sim_vision/python/yolo_server.py`, `sim_vision/godot/scripts/vision_client.gd`.
- **Minh chứng:** Báo cáo chi tiết Giai đoạn 1 tại [docs/progress/huy/khao_sat_vision/0001.md](../progress/huy/khao_sat_vision/0001.md) và các phần bên dưới (§2, §3, §4, §5).

---

## 2. Bản sắc & Mục tiêu cốt lõi của Dự án (Core Mission)

- **Tên đồ án:** **Robot trợ lý dẫn đường bệnh viện** (*Hospital Helper Robot*).
- **Mục tiêu thực tế:** Hỗ trợ bệnh nhân tại quầy tiếp đón (đặc biệt là **người cao tuổi, người đi xe lăn, người khiếm thị**) tìm đúng phòng khám, giảm tải cho nhân viên y tế vào giờ cao điểm.
- **Phạm vi triển khai:** Được mô phỏng và kiểm chứng 100% trong môi trường **Godot Engine 4 (3D)** dựa trên thông số khảo sát thực tế (không làm phần cứng robot vật lý).
- **Hành vi nghiệp vụ của Robot trong ca làm việc (180 giây/shift):**
  1. **Tuần tra / Tìm kiếm (`SEARCH`):** Robot xuất phát từ cổng chính (`RobotDock`), quan sát hành lang tìm bệnh nhân đang cần giúp đỡ.
  2. **Tiếp cận (`APPROACH`):** Dùng camera/thị giác tiến lại gần bệnh nhân ($\le 1.6\text{ m}$).
  3. **Hội thoại (`DIALOGUE`):** Lắng nghe câu hỏi qua mic $\to$ Hiểu ý định (hỏi phòng, giờ khám, cấp cứu) $\to$ Trả lời bằng giọng nói tiếng Việt qua loa 3D.
  4. **Dẫn đường (`GUIDE_ESCORT`):** Dẫn bệnh nhân tới các phòng tầng 1 (101, 110, 112, Nhà thuốc, Cấp cứu) hoặc **dẫn tới thang máy** nếu phòng khám ở tầng 2, 3, 4. Trong lúc dẫn, robot phải giữ khoảng cách nhịp độ (tether $\le 2.5\text{ m}$) để người già đi kịp.
  5. **Gọi nhân viên y tế (`CALL_STAFF`):** Trường hợp khẩn cấp, robot tự tìm điều dưỡng/bác sĩ gần nhất để báo động.

---

## 3. Kiến trúc hệ thống đa tầng & Hợp đồng giao tiếp (Microservice Contracts)

Hệ thống hoạt động theo mô hình phân tán thời gian thực (Micro-services cục bộ) giao tiếp hoàn toàn qua **WebSocket nội bộ (localhost)**:

```mermaid
flowchart TD
    subgraph GODOT["1. MÔI TRƯỜNG MÔ PHỎNG (Godot 4.7 3D)"]
        Cam["Robot Camera (640x480)"]
        LiDAR["LiDAR 17 tia (200° FOV)"]
        Mic["Micro tiếp nhận giọng nói NPC"]
        Speaker["Loa 3D robot"]
        Hospital3D["Map bệnh viện: 6 phòng, cửa trượt, thang máy, NPC"]
    end

    subgraph VISION["2. MODULE THỊ GIÁC (Port 8765)"]
        YoloSrv["yolo_server.py (YOLOv8n Offline)"]
    end

    subgraph VOICE["3. MODULE GIỌNG NÓI & NLU (Port 8766 & 8767)"]
        STT["STT: PhoASR-Whisper-Small"]
        NLU["NLU: PhoBERT Fine-tuned / Rule"]
        TTS["TTS: VieNeu-TTS v3 Turbo"]
    end

    subgraph BRAIN["4. BỘ ĐIỀU KHIỂN & HỌC TĂNG CƯỜNG (RL)"]
        FSM["Controller FSM (Luật hình học)"]
        RL["RL Policy: PPOHospital-full_strongest.pt"]
    end

    Cam -->|JPEG 5 Hz (Base64)| YoloSrv
    YoloSrv -->|Detections [x1, y1, x2, y2, conf]| GODOT
    
    Mic -->|Audio WAV 16kHz| STT
    STT -->|Text| NLU
    NLU -->|Intent, Action, Target Slot| GODOT
    
    GODOT -->|Text phản hồi| TTS
    TTS -->|PCM Audio| Speaker

    GODOT -->|31-D Vector: LiDAR + Bounding Box + Goal| RL
    RL -->|Vận tốc liên tục: v (0..0.8 m/s), omega (-1.2..1.2 rad/s)| GODOT
```

### Các hợp đồng WebSocket chính:
1. **Vision Service (Port `8765`):**
   - **Tần số:** 5 Hz (`Config.CAPTURE_HZ = 5`).
   - **Payload In:** `{"type": "frame", "frame_id": int, "jpeg_b64": str, "width": 640, "height": 480}`.
   - **Payload Out:** `{"type": "detections", "frame_id": int, "boxes": [{"x1": float, "y1": float, "x2": float, "y2": float, "conf": float}], "infer_ms": float}`.
   - **Cơ chế chống ứ đọng (Drop stale):** Server chỉ xử lý frame mới nhất nhận được; các frame chờ cũ hơn bị loại bỏ ngay lập tức.
2. **Speech & NLU Service (Port `8766`):**
   - Nhận diện giọng nói tiếng Việt bằng `PhoASR-whisper-small` và phân loại ý định qua `PhoBERT`.
3. **TTS Service (Port `8767`):**
   - Chuyển văn bản câu trả lời sang âm thanh qua `VieNeu-TTS v3 Turbo`.
4. **RL Policy Service (Port `8768`):**
   - Nhận vector quan sát 31 chiều (17 LiDAR + 4 detection features + 4 one-hot phase + 3 goal polar + 3 tether/velocity) và trả về action liên tục $(v, \omega)$.

---

## 4. Bản đồ thư mục & Chức năng từng Module (Repository Breakdown)

```text
Data-Science-Major-Project/
├── agents/                    # Governance & luật AI bất biến (LOGGING, COMMIT, MD)
├── configs/                   # File cấu hình YAML mẫu
├── data/                      # Dữ liệu STT giọng nói (real_voice: 3,668 file) [CHƯA CÓ DATA VISION]
├── docs/                      # Tài liệu nghiên cứu, đề cương, hướng dẫn, nhật ký kỹ thuật
│   ├── AI-Human-logs/         # Nhật ký quyết định của các kỹ sư (Vuong.md, Manh.md, Tuan.md, Huy.md)
│   └── Đề cương đồ án...      # Mục tiêu, phạm vi và tiêu chí đánh giá
├── models/                    # Trọng số offline (yolov8n, PhoASR, PhoBERT, VieNeu-TTS)
├── experiments/               # Kết quả huấn luyện RL và chạy thực nghiệm (checkpoints, logs)
├── notebooks/                 # Phân tích & biểu diễn trực quan (không dùng để train model)
├── scripts/                   # Tiện ích chuyển đổi audio, download model, test ASR
├── sim_vision/                # TRỌNG TÂM: Môi trường Godot 3D & các Backend Server
│   ├── godot/                 # Mã nguồn Godot (Scene, Script, Assets, GUT Tests)
│   └── python/                # Backend Python: yolo_server, detector, voice, nlu, tts
├── src/                       # Thư viện thuật toán học máy lõi
│   ├── models/                # Kiến trúc mạng neural (ppo_agent.py)
│   ├── training/              # Môi trường huấn luyện surrogate (hospital_env.py) & scripts train PPO
│   └── ui/                    # Dashboard giám sát huấn luyện (rl_dashboard.py)
├── tests/                     # Bộ kiểm thử Pytest tích hợp toàn diện
├── run_simulation.py          # Unified Launcher khởi động toàn bộ hệ thống
└── run_dashboard.py           # Web Dashboard Gradio giám sát RL
```

---

## 5. Hiện trạng & Khoảng trống kỹ thuật của Module Vision (Audit Findings)

| Thành phần | Hiện trạng do người trước để lại | Khoảng trống & Nhiệm vụ của Huy |
| :--- | :--- | :--- |
| **Pipeline kết nối** | Đã hoàn thành (Godot Camera $\leftrightarrow$ WebSocket Port 8765 $\leftrightarrow$ `yolo_server.py`). | Duy trì và mở rộng thêm metadata (ID tracking, class nhãn). |
| **Model thị giác** | Dùng nguyên bản `yolov8n.pt` pretrained trên COCO, chỉ lọc `person`. | Chưa tối ưu cho môi trường bệnh viện và góc camera robot tầm thấp. |
| **Theo dõi đối tượng (MOT)** | **Chưa có**: Nhận diện từng frame độc lập, không giữ danh tính bệnh nhân. | Cần bổ sung thuật toán tracking (**ByteTrack** hoặc **DeepSORT/BoT-SORT**). |
| **Tái định danh (Re-ID)** | **Chưa có**: Dễ bám nhầm người lạ khi bệnh nhân đi vào đám đông. | Cần Re-ID feature extraction để cố định đối tượng bệnh nhân mục tiêu. |
| **Dataset thị giác** | **Chưa có thư mục data chuẩn**: Hoàn toàn thiếu dataset có gán nhãn. | **Nhiệm vụ cốt lõi**: Thu thập, phân loại ca Dễ vs Khó, gán nhãn xe lăn, người chống nạng, góc quay thấp. |
| **Đo lường nhiễu (Noise Profile)** | Chưa có số liệu thực tế về tỷ lệ mất dấu, đổi ID. | Đo đạc MOTA, IDF1, tỷ lệ mất dấu để đưa vào mô phỏng RL huấn luyện. |

---

### 👉 Kết luận nghiệm thu Giai đoạn 1:
- Khảo sát hoàn tất: Hiểu rõ toàn bộ cấu trúc dự án, luồng dữ liệu liên module và các hợp đồng kỹ thuật.
- Xác định rõ nhiệm vụ của Huy: **Xây dựng chiến lược dữ liệu, phân loại ca khó/dễ, và nâng cấp module Vision từ nhận diện tĩnh đơn thuần thành hệ thống Tracking & Re-ID chuyên sâu cho bệnh viện.**
- Sẵn sàng chuyển tiếp sang **Giai đoạn 2: Tìm hiểu sâu mã nguồn module Vision**.
