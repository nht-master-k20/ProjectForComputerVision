import pandas as pd
from sklearn.model_selection import train_test_split
import os
import random
import cv2
import albumentations
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import functools
import shutil


class ReadData:
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    GT_PATH = 'dataset/ISIC_2024_Training_GroundTruth.csv'
    IMAGES_DIR = 'dataset/ISIC_2024_Training_Input'

    # Thư mục lưu ảnh sau xử lý
    OUTPUT_IMG_DIR = 'dataset/ISIC_Processed_Images'

    # Thư mục lưu file CSV
    CSV_OUTPUT_DIR = 'dataset_splits'

    ID_COLUMN = 'isic_id'
    TARGET_COLUMN = 'malignant'

    @classmethod
    def load_metadata(cls):
        try:
            df = pd.read_csv(cls.GT_PATH)
            df['image_path'] = df[cls.ID_COLUMN].apply(lambda x: os.path.join(cls.IMAGES_DIR, f"{x}.jpg"))
            print(f"✅ Đã tải metadata: {len(df)} ảnh.")
            return df
        except Exception as e:
            print(f"❌ Lỗi tải CSV gốc: {e}")
            return None

    @classmethod
    def split_data(cls, df):
        """Chia Stratified: Train/Val/Test"""
        # Chia 20% cho Test
        train_val, test = train_test_split(df, test_size=0.2, stratify=df[cls.TARGET_COLUMN], random_state=42)
        # Chia 10% tổng (0.125 của 80%) cho Val
        train, val = train_test_split(train_val, test_size=0.125, stratify=train_val[cls.TARGET_COLUMN],
                                      random_state=42)

        print(f"📊 Thống kê: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return train, val, test

    # --- WORKER XỬ LÝ ẢNH (CLEAN) ---
    @staticmethod
    def remove_hair(image):
        """Thuật toán xóa lông với Kernel 5x5"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
            _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
            return cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
        except:
            return image

    @staticmethod
    def _process_worker(row_tuple, output_dir):
        """Hàm chạy song song: Đọc -> Resize -> Xóa lông -> Lưu"""
        idx, row = row_tuple
        src_path = row['image_path']
        fname = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, fname)

        # Nếu ảnh đã tồn tại thì bỏ qua (Resume)
        if os.path.exists(dst_path): return dst_path

        try:
            img = cv2.imread(src_path)
            if img is not None:
                # [TỐI ƯU] Resize về 300x300 để tiết kiệm ổ cứng và tăng tốc train
                img = cv2.resize(img, (300, 300))
                clean = ReadData.remove_hair(img)
                cv2.imwrite(dst_path, clean)
                return dst_path
        except:
            pass
        return src_path  # Fallback về ảnh gốc nếu lỗi

    @classmethod
    def clean_dataset(cls, df, folder_name):
        """Áp dụng xóa lông đa luồng"""
        save_dir = os.path.join(cls.OUTPUT_IMG_DIR, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"🧹 Đang làm sạch {len(df)} ảnh vào '{folder_name}'...")

        # Tự động dùng tối đa số nhân CPU
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            func = functools.partial(cls._process_worker, output_dir=save_dir)
            # Chạy map và lấy kết quả đường dẫn mới
            new_paths = list(tqdm(ex.map(func, df.iterrows()), total=len(df)))

        df_new = df.copy()
        df_new['image_path'] = new_paths
        return df_new

    # --- AUGMENTATION (CHỈ CHO TRAIN) ---
    @classmethod
    def augment_minority_class(cls, train_df):
        """Sinh ảnh Offline cho lớp thiểu số (Ác tính)"""
        aug_dir = os.path.join(cls.OUTPUT_IMG_DIR, 'Augmented_Train')
        os.makedirs(aug_dir, exist_ok=True)

        # Pipeline biến đổi mạnh cho lớp ác tính
        pipeline = albumentations.Compose([
            albumentations.Resize(300, 300),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            albumentations.GridDistortion(p=0.3),  # Méo da
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3)
        ])

        # Tính số lượng cần sinh
        counts = train_df[cls.TARGET_COLUMN].value_counts()
        maj_count = counts.idxmax()
        min_count = counts.idxmin()  # Lớp ác tính
        diff = counts[maj_count] - counts[min_count]

        if diff <= 0: return train_df

        print(f"🎨 Đang sinh thêm {diff} ảnh cho lớp Ác tính...")
        minority_imgs = train_df[train_df[cls.TARGET_COLUMN] == counts.idxmin()]['image_path'].tolist()

        new_rows = []
        for i in tqdm(range(diff)):
            src = random.choice(minority_imgs)
            try:
                img = cv2.imread(src)
                if img is None: continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                aug_img = pipeline(image=img)['image']

                fname = f"aug_{i}_{os.path.basename(src)}"
                dst = os.path.join(aug_dir, fname)

                # Lưu lại
                cv2.imwrite(dst, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

                # Thêm vào metadata
                new_rows.append({
                    cls.ID_COLUMN: f"aug_{i}",
                    cls.TARGET_COLUMN: counts.idxmin(),
                    'image_path': dst
                })
            except:
                continue

        return pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True)

    @classmethod
    def run(cls):
        """Hàm chạy duy nhất: Split -> Clean -> Augment -> Save"""
        print("🚀 Bắt đầu quy trình xử lý dữ liệu (Clean > Augment)...")

        # 1. Load
        df = cls.load_metadata()
        if df is None: return False

        # 2. Split
        train, val, test = cls.split_data(df)

        # 3. Clean (Xóa lông cho cả 3 tập)
        train = cls.clean_dataset(train, 'Train_Clean')
        val = cls.clean_dataset(val, 'Val_Clean')
        test = cls.clean_dataset(test, 'Test_Clean')

        # 4. Augment (Chỉ tập Train)
        train = cls.augment_minority_class(train)

        # 5. Save CSV
        os.makedirs(cls.CSV_OUTPUT_DIR, exist_ok=True)
        print(f"💾 Đang lưu CSV vào {cls.CSV_OUTPUT_DIR}...")

        # Tên file chuẩn để khớp với các file train v1, v2, v3
        train.to_csv(f'{cls.CSV_OUTPUT_DIR}/processed_train.csv', index=False)
        val.to_csv(f'{cls.CSV_OUTPUT_DIR}/processed_val.csv', index=False)
        test.to_csv(f'{cls.CSV_OUTPUT_DIR}/processed_test.csv', index=False)

        print("✅ Hoàn tất toàn bộ!")
        return True