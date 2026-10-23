# 📚 TỔNG QUAN TÀI LIỆU DỰ ÁN ROBOT BỆNH VIỆN (`/docs`)

| Thuộc Tính | Giá Trị |
| :--- | :--- |
| **Loại Tài Liệu** | Master Documentation Index (Bản Đồ Điều Hướng Tài Liệu Trung Tâm) |
| **Dự Án** | Data Science Major Project — Robot Trợ Lý Dẫn Đường Bệnh Viện |
| **Trạng Thái** | Active Living Index |
| **Quy Chuẩn Quản Trị** | [agents/rules/FOLDER_STRUCTURE.md](../agents/rules/FOLDER_STRUCTURE.md) & [agents/rules/MD_CONVENTION.md](../agents/rules/MD_CONVENTION.md) |
| **Cập Nhật Lần Cuối** | 2026-10-23T21:33:00+07:00 |

---

## 🗺️ Cấu Trúc Phân Cấp Thư Mục Tài Liệu Chuẩn Hóa

Toàn bộ tài liệu kỹ thuật, đề cương, báo cáo kiểm định và kế hoạch triển khai của dự án được tổ chức chặt chẽ theo các thư mục chuyên biệt:

```text
docs/
├── README.md                                    # 🗺️ Bản đồ điều hướng tài liệu tổng quát (File này)
├── PURPOSE.md                                   # 🎯 Mục tiêu dự án & Yêu cầu ràng buộc hệ thống
├── OVERVIEW.md                                  # 🧭 Lộ trình phát triển sống & Theo dõi các cột mốc
│
├── reports/                                     # 📊 Báo Cáo Kỹ Thuật & Đánh Giá Benchmark Thị Giác
│   ├── cases_under_70pct_detection_forensic_report.md
│   ├── master_failure_cases_and_low_detection_rates_report.md
│   ├── horizontal_label_detection_visualization_report.md
│   └── README.md
│
├── plans/                                       # 📋 Kế Hoạch, Đề Cương & Bản Quy Hoạch Kỹ Thuật
│   ├── de_cuong_do_an_robot_benh_vien.md       # Đề cương đồ án tốt nghiệp Data Science
│   ├── plan_bad_detection_cases_deep_dive.md   # Quy hoạch kỹ thuật đào sâu 44 ca phát hiện lỗi
│   ├── RL_TRAINING_PLAN.md                     # Kế hoạch huấn luyện mạng PPO / RL Navigation
│   ├── kich_ban_thu_am.md                      # Kịch bản 80 câu thoại tiếng Việt cho Voice/STT
│   └── README.md
│
├── experiments/                                 # 🧪 Báo Cáo Đặc Tả & Thực Nghiệm Độc Lập
│   ├── EXP_VISION_48CASES_1200_BENCHMARK.md    # Benchmark 48 kịch bản baseline (vision_eval_1200)
│   ├── EXP_VISION_EXTENDED_NPC_BENCHMARK.md    # Benchmark 36 ca lâm sàng mở rộng (10,800 frames)
│   ├── EXP_VISION_MODEL_BENCHMARK.md           # Đối kháng 4 mô hình (v8n, v11n, v26n, OpenVINO)
│   └── README.md
│
├── phases/                                      # 🚀 Đặc Tả Các Pha Triển Khai Kỹ Thuật
│   ├── PHASE_VISION_EVAL_BENCHMARK.md          # Pha đánh giá bộ não thị giác robot bệnh viện
│   └── README.md
│
├── progress/                                    # 📈 Báo Cáo Trạng Thái Tiến Độ Phiên Làm Việc
│   ├── VISION_EVAL_STATUS.md                   # Trạng thái hoàn thành bộ benchmark thị giác
│   └── README.md
│
├── references/                                  # 📚 Hướng Dẫn Kỹ Thuật & Tài Liệu Tra Cứu
│   ├── guide.md                                 # Sổ tay vận hành & phát triển dự án
│   ├── GIT_AND_RELEASE_BEST_PRACTICES.md       # Quy chuẩn Git, Commit và Release
│   ├── ML_PIPELINE_REFERENCE_v3.md             # Cẩm nang 18 bước chuẩn hóa kỹ nghệ ML
│   ├── ML_PIPELINE_REFERENCE_v4.md             # Bản mở rộng v4 cho Pipeline ML
│   ├── OPTUNA_DB_GUIDE.md                      # Hướng dẫn lưu trữ và tối ưu Optuna SQLite
│   └── README.md
│
├── PROMPT/                                      # 💬 Kịch Bản Prompt Cho Godot / NLU / STT
│   ├── godot_npc_stt_prompt.md                 # Prompt tích hợp NPC STT trong Godot
│   ├── godot_robot_yolo_prompt.md              # Prompt điều khiển robot qua YOLO
│   └── nlu_tts_dialogue_prompt.md              # Prompt huấn luyện mô hình hội thoại NLU/TTS
│
├── shared/                                      # 🤝 Quy Trình Phối Hợp & Handoff Nhiệm Vụ
│   ├── HANDOFF_TEMPLATE.md                     # Mẫu bàn giao công việc giữa các phiên làm việc
│   └── HOW_TO_SETUP_AI_AGENT.md                # Quy trình 10 bước thiết lập AI Coding Agent
│
├── bugs/                                        # 🐛 Hồ Sơ Khắc Phục Lỗi & Kiểm Định Hồi Quy
│   └── README.md
│
└── AI-Human-logs/                               # 📝 Nhật Ký Quyết Định Làm Việc Đội Ngũ
    ├── Manh.md                                 # Nhật ký công việc: Mạnh
    ├── Tuan.md                                 # Nhật ký công việc: Tuấn
    ├── Uyen.md                                 # Nhật ký công việc: Uyên
    └── Vuong.md                                # Nhật ký công việc: Vượng
```

---

## 🎯 Chỉ Mục Truy Cập Nhanh Theo Lĩnh Vực

### 1. Hệ Thống Thị Giác Robot (AI Vision Benchmark)
- **Báo cáo phân tích pháp y 44 ca lỗi**: [docs/reports/master_failure_cases_and_low_detection_rates_report.md](reports/master_failure_cases_and_low_detection_rates_report.md)
- **Báo cáo trực quan hóa biểu đồ cột ngang**: [docs/reports/horizontal_label_detection_visualization_report.md](reports/horizontal_label_detection_visualization_report.md)
- **Quy hoạch giải pháp công nghệ khắc phục lỗi**: [docs/plans/plan_bad_detection_cases_deep_dive.md](plans/plan_bad_detection_cases_deep_dive.md)
- **Đặc tả thực nghiệm 48 ca baseline**: [docs/experiments/EXP_VISION_48CASES_1200_BENCHMARK.md](experiments/EXP_VISION_48CASES_1200_BENCHMARK.md)
- **Đặc tả thực nghiệm 36 ca lâm sàng mở rộng**: [docs/experiments/EXP_VISION_EXTENDED_NPC_BENCHMARK.md](experiments/EXP_VISION_EXTENDED_NPC_BENCHMARK.md)

### 2. Kế Hoạch Đồ Án & Điều Khiển Di Chuyển (Navigation & RL)
- **Đề cương đồ án tốt nghiệp**: [docs/plans/de_cuong_do_an_robot_benh_vien.md](plans/de_cuong_do_an_robot_benh_vien.md)
- **Kế hoạch huấn luyện Reinforcement Learning (PPO)**: [docs/plans/RL_TRAINING_PLAN.md](plans/RL_TRAINING_PLAN.md)

### 3. Giao Tiếp & Xử Lý Ngôn Ngữ Tự Nhiên (Voice & NLU)
- **Kịch bản thu âm 80 câu thoại tiếng Việt**: [docs/plans/kich_ban_thu_am.md](plans/kich_ban_thu_am.md)
- **Prompts thiết kế hội thoại NLU / STT**: [docs/PROMPT/](PROMPT/)

### 4. Quy Chuẩn Kỹ Thuật & Cẩm Nang Lập Trình
- **Sổ tay phát triển dự án**: [docs/references/guide.md](references/guide.md)
- **Quy chuẩn Git & Release**: [docs/references/GIT_AND_RELEASE_BEST_PRACTICES.md](references/GIT_AND_RELEASE_BEST_PRACTICES.md)
- **Hướng dẫn kiến trúc ML Pipeline**: [docs/references/ML_PIPELINE_REFERENCE_v3.md](references/ML_PIPELINE_REFERENCE_v3.md)
