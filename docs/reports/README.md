# 📊 THƯ MỤC BÁO CÁO KỸ THUẬT & ĐÁNH GIÁ THỊ GIÁC (REPORTS)

Thư mục chứa các báo cáo nghiệm thu chuyên sâu, đánh giá hiệu năng phát hiện mục tiêu và phân tích pháp y các ca thất bại của hệ thống AI Vision trên Robot bệnh viện.

---

## 📑 Danh Mục Báo Cáo

| Tên Báo Cáo | Định Dạng / Quy Mô | Mục Tiêu & Nội Dung Trọng Tâm |
| :--- | :--- | :--- |
| [cases_under_70pct_detection_forensic_report.md](cases_under_70pct_detection_forensic_report.md) | Báo cáo kiểm định <70% | Danh mục đầy đủ tất cả các kịch bản kiểm thử có tỷ lệ nhận diện $<70\%$ hoặc $\text{IoU}<0.40$, kèm link ảnh mẫu, link ảnh gốc dataset và phân tích nguyên lý thị giác để kiểm tra thủ công. |
| [master_failure_cases_and_low_detection_rates_report.md](master_failure_cases_and_low_detection_rates_report.md) | Báo cáo chi tiết 44 ca lỗi | Phân loại toàn diện 44 ca phát hiện lỗi trên 4 mô hình AI (`YOLOv8n`, `YOLO11n`, `YOLO26n`, `OpenVINO`), phân tích 4 nhóm bệnh lý thị giác First Principles và tiêu chuẩn an toàn ISO 3691-4. |
| [horizontal_label_detection_visualization_report.md](horizontal_label_detection_visualization_report.md) | Báo cáo trực quan hóa | Phân tích 4 biểu đồ cột ngang (`ax.barh`), đo đạc tỷ lệ nhận diện theo 14 phân nhóm trục, 12 nhãn thực thể lâm sàng đặc thù, bảng xếp hạng 83 ca kiểm thử và ma trận độ trễ/sai số góc. |

---

> [!NOTE]
> Các báo cáo này cũng được đồng bộ và sao lưu tự động tại `sim_vision/reports/` để phục vụ trực tiếp cho mô-đun mô phỏng Godot 4.
