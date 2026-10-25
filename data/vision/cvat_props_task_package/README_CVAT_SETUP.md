# 🚀 HƯỚNG DẪN 3 BƯỚC NẠP DỮ LIỆU VÀO CVAT ĐỂ CHECK TAY
Thư mục gói dữ liệu: `/home/bush/Projects/Data-Science-Major-Project/data/vision/cvat_props_task_package`

Danh mục 13 lớp đối tượng trong bộ dữ liệu này:
- **`0`: `queue_kiosk`**
- **`1`: `reception_counter`**
- **`2`: `sanitizer_stand`**
- **`3`: `pccc_station`**
- **`4`: `medical_waste_bin`**
- **`5`: `water_dispenser`**
- **`6`: `wc_station`**
- **`7`: `elevator_doors`**
- **`8`: `hospital_bed`**
- **`9`: `stretcher_cart`**
- **`10`: `iv_stand`**
- **`11`: `medical_trolley`**
- **`12`: `vital_monitor`**

---

### Bước 1: Tạo Task trên CVAT
1. Mở CVAT (truy cập `http://localhost:8080` hoặc `https://app.cvat.ai`).
2. Bấm **"+ Create new task"**.
3. **Name**: `Hospital_Props_1600_13Class`
4. **Labels**:
   - Bấm vào tab **"Raw"**.
   - Mở file `cvat_labels_schema.json` trong thư mục này, copy toàn bộ nội dung JSON và dán vào ô cấu hình.
   - Bấm **"Done"** để CVAT nạp đủ 13 lớp với mã màu tương ứng.
5. **Select files**: Kéo thả file `images.zip` (78.4 MB) vào vùng upload.
6. Bấm **"Submit & Open"** (Chờ CVAT khởi tạo Job xong trong vài giây).

---

### Bước 2: Nạp Bounding Box nháp đã sinh tự động (AI Pre-annotation)
1. Trong màn hình Task vừa tạo, bấm vào dấu 3 chấm góc trên bên phải (`...`) hoặc menu **Actions**.
2. Chọn **"Upload Annotations"** $\rightarrow$ Chọn định dạng **"YOLO 1.1"**.
3. Chọn file `annotations.zip` trong thư mục này tải lên.
4. Xong! Mở Job ra bạn sẽ thấy **toàn bộ 1600 frames đã có sẵn 2310 Bounding Box**, không phải vẽ tay từ đầu!

---

### Bước 3: Thao tác Kiểm Duyệt Nhanh (5-10s / frame)
- **Space** / **F** / **D**: Lướt qua khung hình tiếp theo / lùi khung hình.
- **N**: Vẽ thêm bounding box nếu thấy vật thể bị sót góc.
- **Click vào box + Delete**: Xóa bounding box bị che khuất quá mức hoặc sai góc nhìn.
- **Ctrl + S**: Lưu định kỳ (CVAT cũng tự động auto-save).

---

### Bước 4: Xuất kết quả sau khi duyệt xong
Khi duyệt hoàn tất:
1. Bấm **Menu** góc trái $\rightarrow$ **Export Task Dataset**.
2. Chọn định dạng **"YOLO 1.1"** (hoặc **"CVAT for images 1.1"**).
3. Tải file ZIP về máy để cập nhật trực tiếp vào dataset chính thức!
