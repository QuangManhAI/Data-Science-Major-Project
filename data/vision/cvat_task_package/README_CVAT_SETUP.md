# 🚀 HƯỚNG DẪN 3 BƯỚC NẠP DỮ LIỆU VÀO CVAT
Thư mục gói dữ liệu: `/home/bush/Projects/Data-Science-Major-Project/data/vision/cvat_task_package`

---

### Bước 1: Tạo Task trên CVAT
1. Mở CVAT (truy cập `http://localhost:8080` hoặc `https://app.cvat.ai`).
2. Bấm **"+ Create new task"**.
3. **Name**: `Hospital_Vision_Gold_Dataset`
4. **Labels**: Bấm tab **"Raw"**, mở file `cvat_labels_schema.json` copy toàn bộ nội dung dán vào (hoặc tự tạo 2 class: `person` màu xanh lá, `wheelchair` màu xanh cyan).
5. **Select files**: Kéo thả file `images.zip` vào mục tải ảnh lên.
6. Bấm **"Submit & Open"** (Chờ CVAT upload xong).

---

### Bước 2: Nạp sẵn Bounding Box nháp (AI Pre-annotation)
1. Trong màn hình Task vừa tạo, bấm vào dấu 3 chấm góc phải (`...`) hoặc menu **Actions**.
2. Chọn **"Upload Annotations"** $\rightarrow$ Chọn format **"YOLO 1.1"**.
3. Chọn file `annotations.zip` trong thư mục này tải lên.
4. Xong! Mở Job ra bạn sẽ thấy **toàn bộ 696 frames đã có sẵn Bounding Box**, không ai phải vẽ tay từ đầu.

---

### Bước 3: Duyệt và sửa lỗi (Chỉ 5-10s/frame)
- Bấm `Space` để lướt qua các ảnh đã chuẩn.
- Bấm `N` để vẽ thêm người bị sót (ví dụ em bé bế trên tay).
- Click vào box chọn và bấm `Delete` nếu AI vẽ nhầm lên Standee poster.
- Bấm `Ctrl + S` lưu định kỳ.

Sau khi review xong, bấm **"Export Task Dataset"** $\rightarrow$ chọn **"YOLO 1.1"** để tải nhãn hoàn chỉnh về!
