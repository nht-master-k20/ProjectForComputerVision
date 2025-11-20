# Phân Loại Các Loại Tổn Thương Trên Da

**Môn học: Xử lý ảnh và Thị giác máy tính (CS2203.CH200)**

------------------------------------------------------------------------

## 🎯 1. Mục tiêu đề tài

Đề tài hướng đến việc xây dựng một hệ thống học sâu có khả năng:

-   **Phân loại các loại tổn thương trên da** từ hình ảnh chụp lâm sàng
-   Hỗ trợ **nhận diện sớm các dấu hiệu bệnh lý da liễu**
-   Ứng dụng các kỹ thuật xử lý ảnh, tăng cường dữ liệu và mô hình học
    sâu hiện đại
-   Tối ưu hóa mô hình với các kỹ thuật:
    -   Focal Loss
    -   Class Weighting
    -   Weighted Sampling
    -   Dynamic Thresholding
    -   Bias Initialization

------------------------------------------------------------------------

## 📂 2. Dataset sử dụng

Dataset lấy từ nghiên cứu đăng trên tạp chí Nature:

**SkinExplainer: A Comprehensive Dataset and Benchmark for Skin Disease
Classification**
https://www.nature.com/articles/s41597-024-03743-w

------------------------------------------------------------------------

## 🧠 3. Tóm tắt phương pháp tiếp cận

### 1️⃣ Xử lý dữ liệu

-   Load metadata
-   Stratified split Train/Val/Test
-   Hair removal + resize 300x300
-   Clean đa luồng bằng ProcessPoolExecutor

### 2️⃣ Augmentation

-   Chỉ áp dụng cho lớp malignant
-   Albumentations: flip, rotate, distortion, color jitter
-   Sinh ảnh offline tăng số lượng mẫu thiểu số

### 3️⃣ Huấn luyện mô hình
- **v1**: CrossEntropyLoss baseline
- **v2**: Focal Loss + WeightedRandomSampler
- **v3**: Focal + Sampler + BiasInit + Dynamic Threshold

### 4️⃣ Tracking

-   MLflow log toàn bộ chỉ số Train/Val/Test
-   Lưu best model theo F1-malignant

------------------------------------------------------------------------

## 🛠️ 4. Công nghệ sử dụng

-   PyTorch, timm
-   Albumentations
-   OpenCV
-   Pandas, NumPy
-   MLflow
-   scikit-learn

------------------------------------------------------------------------

## 📈 5. Kết quả mong đợi

-   F1-malignant cao
-   Giảm overfitting
-   Cải thiện độ chính xác nhận diện tổn thương ác tính
-   Xuất classification report + confusion matrix trên tập Test

------------------------------------------------------------------------

## 📁 6. Cấu trúc project

    project/
    │── main.py
    │── README.md
    │
    ├── scripts/
    │   ├── ReadData.py
    │   └── ISICDataset.py
    │
    ├── models/
    │   ├── EfficientNetB3_v1.py
    │   ├── EfficientNetB3_v2.py
    │   └── EfficientNetB3_v3.py
    │
    ├── dataset/
    │   ├── ISIC_2024_Training_Input/
    │   └── ISIC_2024_Training_GroundTruth.csv
    │
    └── dataset_splits/
        ├── processed_train.csv
        ├── processed_val.csv
        └── processed_test.csv

------------------------------------------------------------------------

## ⚙️ 7. Cách chạy project

### Xử lý dữ liệu

    python main.py data

### Train mô hình

Baseline:

    python main.py v1

Focal + Sampler:

    python main.py v2

BiasInit + Dynamic Threshold:

    python main.py v3

------------------------------------------------------------------------

## 📑 8. Tài liệu tham khảo

-   SkinExplainer Dataset (Nature)
-   EfficientNet
-   Albumentations
-   PyTorch Docs
-   MLflow Docs
