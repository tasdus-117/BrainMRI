# Brain MRI Segmentation Project



Dự án này tập trung vào bài toán phân vùng ảnh y tế (Medical Image Segmentation) trên dữ liệu chụp cộng hưởng từ (MRI) sọ não. Quá trình huấn luyện và thử nghiệm được chia thành các giai đoạn với cả dữ liệu 2D và 3D, áp dụng nhiều kiến trúc mô hình khác nhau để so sánh hiệu năng.

## ⚙️ Hyperparameters (Cài đặt chung)
- **BATCH_SIZE:** 8
- **EPOCHS:** 20
- **LEARNING_RATE (LR):** 1e-4

---

## 🔬 Giai đoạn 1: Huấn luyện 2D với dữ liệu thô (.nii)

### Tổng quan Dữ liệu
- **Số lượng:** 368 bệnh nhân (bỏ qua bệnh nhân `355` do lỗi tên thư mục/file không đúng định dạng).
- **Đầu vào gốc:** Voxel 3D có shape `(240, 240, 155)`.
- **Phương pháp trích xuất:** Lấy 1 slice duy nhất ở chính giữa voxel (slice thứ 77) của từng bệnh nhân vì đây là lát cắt chứa nhiều thông tin nhất của hộp sọ. Mỗi bệnh nhân tương ứng với 1 ảnh 2D.

### Pipeline & Kết quả
- Tiền xử lý dữ liệu và đưa vào huấn luyện với 3 mô hình. 
- Log lại các chỉ số `Loss`, `Dice`, `IoU` qua từng epoch.
- **Thời gian huấn luyện:** ~2.5 giờ cho 3 models (20 Epochs).
- **Chỉ số hiện tại:** Dice Score ~ `0.69`, IoU ~ `0.53`.

### ⚠️ Known Issues
- Xảy ra lỗi mất code trên môi trường VSCode nên chưa thực hiện đánh giá (evaluate) trên tập Validation được. Đã khắc phục bằng cách **chuyển môi trường phát triển sang PyCharm**.

### 📄 Logs file
- `log_3D_UNet-3D.csv`
- `log_DeepLabV3+.csv`
- `log_Swin-UNet.csv`

---

## ⚡ Giai đoạn 2: Tối ưu hóa tốc độ với dữ liệu 2D (.npy)

### Cải tiến Pipeline
- Thay vì đọc trực tiếp file `.nii` nặng nề, dữ liệu được tiền xử lý và lưu dưới dạng mảng Numpy (`.npy`).
- Vẫn tiếp tục sử dụng slice thứ 77 (chính giữa).

### Kết quả Cải thiện
- **Tốc độ huấn luyện:** Giảm đột phá từ **2.5 giờ xuống chỉ còn 30 phút** khi train cùng 3 mô hình như Giai đoạn 1. Việc giải nén trước dữ liệu giúp loại bỏ thắt cổ chai ở khâu I/O.

### 📄 Logs file
- `log_2D_npy_DeepLabV3+.csv`
- `log_2D_npy_Swin-UNet.csv`
- `log_2D_npy_UNet++.csv`

---

## 🧊 Giai đoạn 3: Huấn luyện 3D Segmentation

### Tổng quan & Pipeline
- **Tiền xử lý:** Biến đổi toàn bộ dữ liệu `.nii` sang `.npy` cho từng bệnh nhân. Resize ảnh về kích thước `160x160` để vừa với input của mô hình.
- **Trích xuất Data:** Mỗi file `.npy` lưu 64 slices (từ slice thứ 50 đến 114) thay vì lấy 1 slice. Đây là vùng giữa chứa lượng thông tin chẩn đoán dày đặc nhất.
- **Hiệu năng I/O:** Tốc độ đọc/tính toán nhanh hơn ~20 lần so với việc để mô hình tự giải nén file `.nii` trong quá trình train.
- Ghi log các chỉ số `Loss`, `Dice`, `IoU` theo từng epoch.

### ⚠️ Known Issues
- Mô hình **Swin UNet (3D)** yêu cầu tài nguyên quá lớn, dẫn đến lỗi **Out of Memory (OOM)** trên GPU hiện tại (*RTX 3050 4GB VRAM*). Tạm thời ngưng chạy mô hình này để tìm phương hướng tối ưu bộ nhớ (Gradient accumulation, giảm batch size, hoặc mixed precision training).

### 📄 Logs file
- `log_3D_DeepLabV3Plus-3D.csv`
- `log_3D_UNet-3D.csv`

---

## 🚀 Phân tích & Hướng cải thiện (Future Works)

Dựa trên kết quả hiện tại (Accuracy cao nhưng Dice/IoU còn thấp), dưới đây là các bước cải thiện tiếp theo cho dự án:

1. **Xử lý mất cân bằng dữ liệu (Class Imbalance):**
   - Vùng Background (nhãn 0) đang chiếm thể tích quá lớn, lấn át các vùng tổn thương.
   - **Giải pháp:** Sử dụng hàm loss kết hợp `Loss = Loss_Dice + Loss_CE` (Weighted Cross Entropy). Tăng trọng số (weight) cho các nhãn `1, 2, 3` lên gấp **5 đến 10 lần** so với nhãn `0`.

2. **Data Augmentation:**
   - Bổ sung các kỹ thuật tăng cường dữ liệu: *Xoay (Rotate), Lật (Flip), và Thêm nhiễu (Noise)* để tăng tính tổng quát cho mô hình.

3. **Tối ưu hóa Learning Rate:**
   - Áp dụng callback `ReduceLROnPlateau`: Tự động theo dõi `Dice Score`. Nếu chỉ số này không được cải thiện sau 3-5 epochs, LR sẽ tự động giảm đi 10 lần (VD: từ `1e-4` xuống `1e-5`) để mô hình hội tụ tốt hơn ở các vùng tối ưu cục bộ.

4. **Khai thác tối đa phần cứng & Mở rộng huấn luyện:**
   - Do đã tối ưu được tốc độ I/O với định dạng `.npy`, sẽ nâng số lượng **EPOCHS lên 100+** để quan sát ngưỡng hội tụ cuối cùng của mô hình.
   - Khi có đủ tài nguyên tính toán, tiến hành chạy thử nghiệm trên **toàn bộ không gian 3D** (full 155 slices từ 1 -> 155) thay vì chỉ cắt 64 slices ở giữa.
