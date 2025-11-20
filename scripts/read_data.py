import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import albumentations
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import functools


class ReadData:
    GT_PATH = 'dataset/ISIC_2024_Training_GroundTruth.csv'
    IMAGES_DIR = 'dataset/ISIC_2024_Training_Input'

    # Thư mục chứa ảnh đã làm sạch
    CLEAN_IMAGES_DIR = 'dataset/ISIC_2024_Clean_Input'
    # Thư mục chứa ảnh tăng cường (chỉ cho tập Train)
    AUG_CLEAN_IMAGES_DIR = 'dataset/ISIC_2024_Augmented_Clean'

    CLASS_MAP = {0: 'Lành tính', 1: 'Ác tính'}

    CSV_OUTPUT_DIR = 'dataset_splits'
    ID_COLUMN = 'isic_id'
    TARGET_COLUMN = 'malignant'

    @classmethod
    def load_isic_metadata(cls) -> pd.DataFrame or None:
        try:
            df = pd.read_csv(cls.GT_PATH)
            df['image_path'] = df[cls.ID_COLUMN].apply(lambda x: os.path.join(cls.IMAGES_DIR, f"{x}.jpg"))
            print(f"✅ Tải thành công {len(df)} bản ghi.")
            return df
        except Exception as e:
            print(f"❌ Lỗi tải metadata: {e}")
            return None

    @classmethod
    def split_data(cls, df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
        """Chia dữ liệu trước khi xử lý để tránh rò rỉ thông tin (Data Leakage)"""
        if (test_size + val_size) >= 1.0:
            raise ValueError("Tổng test_size và val_size phải < 1.0")

        # Stratify split để giữ nguyên tỉ lệ 99.9% vs 0.1%
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df[cls.TARGET_COLUMN], random_state=random_state
        )
        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=relative_val_size, stratify=train_val_df[cls.TARGET_COLUMN],
            random_state=random_state
        )

        print(f"📊 Split Stats: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    @staticmethod
    def remove_hair(image: np.ndarray) -> np.ndarray:
        """
        Xóa lông với kernel 5x5 và xử lý nhẹ nhàng để giữ chi tiết vết thương.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Kernel 5x5 theo yêu cầu
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # BlackHat transform để tìm lông
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

            # Gaussian blur nhẹ để giảm nhiễu
            blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

            # Thresholding
            _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

            # Inpainting
            inpainted = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
            return inpainted
        except Exception:
            return image  # Fallback nếu lỗi

    # --- MULTIPROCESSING WORKER ---
    @staticmethod
    def _clean_single_image(row_tuple, output_dir):
        """Hàm xử lý 1 ảnh (Static method để picklable cho Multiprocessing)"""
        idx, row = row_tuple
        orig_path = row['image_path']
        filename = os.path.basename(orig_path)
        save_path = os.path.join(output_dir, filename)

        # Nếu ảnh đã tồn tại thì bỏ qua (Resume capability)
        if os.path.exists(save_path):
            return save_path

        try:
            img = cv2.imread(orig_path)
            if img is not None:
                # Resize về 256 trước khi remove hair để tăng tốc độ xử lý 400k ảnh
                # img = cv2.resize(img, (256, 256))

                clean_img = ReadData.remove_hair(img)
                cv2.imwrite(save_path, clean_img)
                return save_path
        except Exception:
            pass
        return orig_path  # Trả về ảnh gốc nếu lỗi

    @classmethod
    def clean_dataset_parallel(cls, df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        """Làm sạch dữ liệu sử dụng đa luồng (ProcessPoolExecutor)"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"🚀 Đang xử lý đa luồng {len(df)} ảnh vào: {output_dir}...")

        # Sử dụng số core CPU tối đa - 1 để tránh treo máy
        max_workers = max(1, os.cpu_count() - 1)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Cố định tham số output_dir
            worker = functools.partial(cls._clean_single_image, output_dir=output_dir)

            # Chạy song song và hiện thanh tiến trình
            results = list(tqdm(executor.map(worker, df.iterrows()), total=len(df), unit="img"))

        df_clean = df.copy()
        df_clean['image_path'] = results
        return df_clean

    @staticmethod
    def get_augmentation_pipeline(img_size=256):
        """Pipeline nâng cao cho da liễu"""
        return albumentations.Compose([
            albumentations.Resize(img_size, img_size),

            # Hình học (Geometric)
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),

            # Biến dạng (Distortion) - Giúp model học tính co giãn của da
            albumentations.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),

            # Màu sắc (Color)
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            # p=0.2 theo yêu cầu

            # Nhiễu (Noise) - Optional, thêm vào nếu muốn robust hơn
            # albumentations.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        ])

    @classmethod
    def balance_and_augment(cls, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment chỉ áp dụng cho Train Set.
        Lưu ý: Với 400k ảnh, bước này có thể sinh ra RẤT NHIỀU ảnh.
        """
        os.makedirs(cls.AUG_CLEAN_IMAGES_DIR, exist_ok=True)

        class_counts = train_df[cls.TARGET_COLUMN].value_counts()
        majority_label = class_counts.idxmax()
        minority_label = class_counts.idxmin()

        # Giới hạn số lượng sinh thêm để tránh tràn ổ cứng (ví dụ max 50k ảnh thêm)
        # Bạn có thể bỏ limit này nếu ổ cứng đủ lớn
        n_diff = class_counts[majority_label] - class_counts[minority_label]
        n_to_generate = n_diff  # Hoặc min(n_diff, 50000)

        if n_to_generate <= 0:
            return train_df

        print(f"🎨 Augmenting: Sinh thêm {n_to_generate} ảnh cho lớp {minority_label}...")

        minority_df = train_df[train_df[cls.TARGET_COLUMN] == minority_label]
        minority_paths = minority_df['image_path'].tolist()
        pipeline = cls.get_augmentation_pipeline()

        new_records = []

        # Dùng tqdm để theo dõi tiến độ sinh ảnh
        for i in tqdm(range(n_to_generate), unit="img"):
            src_path = random.choice(minority_paths)
            try:
                img = cv2.imread(src_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Apply Augmentation
                augmented = pipeline(image=img)['image']

                # Save
                fname = f"aug_{i}_{os.path.basename(src_path)}"
                save_path = os.path.join(cls.AUG_CLEAN_IMAGES_DIR, fname)
                cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

                new_records.append({
                    cls.ID_COLUMN: f"aug_{i}",
                    cls.TARGET_COLUMN: minority_label,
                    'image_path': save_path
                })
            except:
                continue

        return pd.concat([train_df, pd.DataFrame(new_records)], ignore_index=True)

    @classmethod
    def run(cls, mode='raw', clean=True):
        # 1. Load Metadata
        full_df = cls.load_isic_metadata()
        if full_df is None: return False

        # 2. Split Data (QUAN TRỌNG: Split trước khi làm bất cứ gì để tránh Leakage)
        train_df, val_df, test_df = cls.split_data(full_df)

        # 3. Clean Data (Áp dụng cho cả 3 tập, nhưng độc lập)
        if clean:
            print("\n🧹 Bắt đầu quy trình làm sạch (Multiprocessing)...")
            # ProcessPoolExecutor được gọi bên trong hàm này
            train_df = cls.clean_dataset_parallel(train_df, cls.CLEAN_IMAGES_DIR)
            val_df = cls.clean_dataset_parallel(val_df, cls.CLEAN_IMAGES_DIR)
            test_df = cls.clean_dataset_parallel(test_df, cls.CLEAN_IMAGES_DIR)

        # 4. Augment Data (CHỈ ÁP DỤNG CHO TRAIN SET)
        if mode == 'augment':
            print("\n🎨 Bắt đầu quy trình Augmentation (Chỉ Train Set)...")
            train_df = cls.balance_and_augment(train_df)

        # 5. Save CSVs
        output_dir = cls.CSV_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        prefix = "clean_" if clean else "raw_"
        suffix = "_augmented" if mode == 'augment' else ""

        print(f"\n💾 Lưu file CSV tại {output_dir}...")
        train_df.to_csv(os.path.join(output_dir, f'{prefix}train{suffix}.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, f'{prefix}val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, f'{prefix}test.csv'), index=False)

        print("✅ Hoàn tất toàn bộ quy trình ReadData.")
        return True