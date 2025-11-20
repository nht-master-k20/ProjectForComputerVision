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


class ReadData:
    GT_PATH = 'dataset/ISIC_2024_Training_GroundTruth.csv'
    IMAGES_DIR = 'dataset/ISIC_2024_Training_Input'

    # Thư mục chứa ảnh đã làm sạch (Cleaned)
    CLEAN_IMAGES_DIR = 'dataset/ISIC_2024_Clean_Input'
    # Thư mục chứa ảnh tăng cường TỪ ẢNH SẠCH (Augmented from Clean)
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
            return df
        except Exception as e:
            print(f"Lỗi tải metadata: {e}")
            return None

    @classmethod
    def split_data(cls, df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
        # Giữ nguyên logic split cũ của bạn
        if (test_size + val_size) >= 1.0:
            raise ValueError("Tổng test_size và val_size phải < 1.0")

        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df[cls.TARGET_COLUMN], random_state=random_state
        )
        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=relative_val_size, stratify=train_val_df[cls.TARGET_COLUMN],
            random_state=random_state
        )
        return train_df, val_df, test_df

    @staticmethod
    def remove_hair(image: np.ndarray) -> np.ndarray:
        # Giữ nguyên logic xóa lông
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            blackhat = cv2.GaussianBlur(blackhat, (5, 5), 0)
            _, thresh = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
            inpainted = cv2.inpaint(image, thresh, 3, cv2.INPAINT_TELEA)
            return inpainted
        except:
            return image

    @classmethod
    def clean_dataset(cls, df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        """
        Bước 1: Làm sạch dữ liệu GỐC trước.
        Kiểm tra nếu ảnh đã tồn tại thì bỏ qua để tiết kiệm thời gian.
        """
        os.makedirs(output_dir, exist_ok=True)
        new_paths = []

        print(f"-> Đang làm sạch {len(df)} ảnh và lưu vào {output_dir}...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            orig_path = row['image_path']
            filename = os.path.basename(orig_path)
            save_path = os.path.join(output_dir, filename)

            # Cache: Nếu đã clean rồi thì dùng luôn
            if os.path.exists(save_path):
                new_paths.append(save_path)
                continue

            try:
                img = cv2.imread(orig_path)
                if img is not None:
                    # Resize về 256 trước khi clean để tăng tốc độ (tùy chọn)
                    # img = cv2.resize(img, (256, 256))
                    clean_img = cls.remove_hair(img)
                    cv2.imwrite(save_path, clean_img)
                    new_paths.append(save_path)
                else:
                    new_paths.append(orig_path)
            except Exception as e:
                new_paths.append(orig_path)

        df_clean = df.copy()
        df_clean['image_path'] = new_paths
        return df_clean

    @staticmethod
    def get_augmentation_pipeline(img_size=256):
        return albumentations.Compose([
            albumentations.Resize(img_size, img_size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            # Lưu ý: CoarseDropout nên dùng cẩn thận trong y tế kẻo che mất vết thương
        ])

    @classmethod
    def balance_and_augment(cls, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Bước 2: Augment dữ liệu TỪ DỮ LIỆU ĐÃ CLEAN.
        """
        os.makedirs(cls.AUG_CLEAN_IMAGES_DIR, exist_ok=True)

        class_counts = train_df[cls.TARGET_COLUMN].value_counts()
        majority_label = class_counts.idxmax()
        minority_label = class_counts.idxmin()
        n_to_generate = class_counts[majority_label] - class_counts[minority_label]

        if n_to_generate <= 0:
            return train_df

        print(f"-> Augmenting: Sinh thêm {n_to_generate} ảnh cho lớp {minority_label} từ dữ liệu Clean...")

        minority_df = train_df[train_df[cls.TARGET_COLUMN] == minority_label]
        minority_paths = minority_df['image_path'].tolist()
        pipeline = cls.get_augmentation_pipeline()

        new_records = []

        for i in tqdm(range(n_to_generate)):
            # Lấy ảnh random từ danh sách đã clean
            src_path = random.choice(minority_paths)
            img = cv2.imread(src_path)
            if img is None: continue

            # Augmentation
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = pipeline(image=img)['image']

            fname = f"aug_{i}_{os.path.basename(src_path)}"
            save_path = os.path.join(cls.AUG_CLEAN_IMAGES_DIR, fname)

            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

            new_records.append({
                cls.ID_COLUMN: f"aug_{i}",
                cls.TARGET_COLUMN: minority_label,
                'image_path': save_path
            })

        return pd.concat([train_df, pd.DataFrame(new_records)], ignore_index=True)

    @classmethod
    def run(cls, mode='raw', clean=True):
        # 1. Load Metadata
        full_df = cls.load_isic_metadata()
        if full_df is None: return False

        # 2. Split Data
        train_df, val_df, test_df = cls.split_data(full_df)

        # 3. Xử lý CLEAN (Quan trọng: Clean trước Augment)
        if clean:
            print("\n[PROCESS] Bắt đầu làm sạch dữ liệu (Hair Removal)...")
            # Lưu vào folder Clean
            train_df = cls.clean_dataset(train_df, cls.CLEAN_IMAGES_DIR)
            val_df = cls.clean_dataset(val_df, cls.CLEAN_IMAGES_DIR)
            test_df = cls.clean_dataset(test_df, cls.CLEAN_IMAGES_DIR)

        # 4. Xử lý AUGMENT (Nếu mode='augment')
        if mode == 'augment':
            print("\n[PROCESS] Bắt đầu Augmentation (Balancing)...")
            # Hàm này sẽ lấy ảnh từ train_df (đã clean ở bước 3 nếu clean=True)
            train_df = cls.balance_and_augment(train_df)

        # 5. Lưu file CSV
        output_dir = cls.CSV_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        prefix = "clean_" if clean else "raw_"
        suffix = "_augmented" if mode == 'augment' else ""

        print(f"\n[OUTPUT] Lưu CSV tại {output_dir}...")
        train_df.to_csv(os.path.join(output_dir, f'{prefix}train{suffix}.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, f'{prefix}val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, f'{prefix}test.csv'), index=False)

        print("Done.")
        return True