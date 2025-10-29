# Xử lý ảnh và thị giác máy tính - CS2203.CH200  
## Đề tài: **Phân Loại Các Loại Tổn Thương Trên Da**

### 👨‍💻 Thành viên nhóm
| STT | Họ và Tên         | MSHV |
|----:|-------------------|------------------|
| 1   | Trịnh Tuấn Nam     |                  |
| 2   | Cao Đức Trí        |                  |
| 3   | Nguyễn Dương Hải   |                  |

---

### 🎯 Mục tiêu đề tài
Xây dựng mô hình học máy có khả năng **phân loại các loại tổn thương trên da** từ hình ảnh, hỗ trợ nhận diện sớm các dấu hiệu bệnh lý về da liễu.  
Đề tài hướng đến việc ứng dụng **Thị giác máy tính (Computer Vision)** và **Học sâu (Deep Learning)** để xử lý và phân tích hình ảnh da người.

---

### 📂 Dataset sử dụng
Dataset lấy từ bài báo khoa học Nature:  
**SkinExplainer: A Comprehensive Dataset and Benchmark for Skin Disease Classification**  
Link dataset: https://www.nature.com/articles/s41597-024-03743-w

---

### 🧠 Phương pháp tiếp cận (Tóm tắt)

---

### 🛠️ Công nghệ / Thư viện dự kiến sử dụng

---

### 📈 Kết quả mong đợi

---

### 📑 Tài liệu tham khảo

---

### ⚙️ Quy tắc chung trong project

Hệ thống cho phép tổ chức và thực thi các tác vụ thông qua từng **script module**.  
Mỗi script được xây dựng dưới dạng **một Class chính**, trong đó bao gồm các **phương thức xử lý** logic cụ thể.  
Việc thực thi script được điều phối tập trung thông qua file `main.py`.

---

#### 1. Cấu trúc lưu trữ
- Tất cả các script phải được đặt trong thư mục `scripts/`.
- Dataset phải được đặt trong thư mục `dataset/`.
- Mỗi file script tương ứng với **một tác vụ**.

#### 2. Cấu trúc một script
- Mỗi script **phải có một class chính** đại diện cho tác vụ cần thực thi.
- Bên trong class bao gồm các **hàm con (method)** phục vụ cho từng bước xử lý.
- Class **bắt buộc phải có hàm `run()`** làm điểm vào chính của tác vụ.

#### 3. Cách thực thi script
Chạy chương trình thông qua `main.py`, truyền tên script cần thực thi vào tham số `--run`:

```bash
python main.py --run "<tên_script_1>" "<tên_script_2>" ...

