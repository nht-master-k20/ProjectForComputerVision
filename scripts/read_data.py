import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import albumentations
from tqdm import tqdm


class ReadData:
    GT_PATH = 'dataset/ISIC_2024_Training_GroundTruth.csv'
    IMAGES_DIR = 'dataset/ISIC_2024_Training_Input'
    AUG_IMAGES_DIR = 'dataset/ISIC_2024_Training_Input_Augmented'
    CLASS_MAP = {0: 'Lành tính', 1: 'Ác tính'}
    CSV_OUTPUT_DIR = 'dataset_splits'
    CSV_OUTPUT_DIR_AUG = 'dataset_splits_aug'


    ID_COLUMN = 'isic_id'
    TARGET_COLUMN = 'malignant'

    @classmethod
    def load_isic_metadata(cls) -> pd.DataFrame or None:
        try:
            df = pd.read_csv(cls.GT_PATH)
            df['image_path'] = df[cls.ID_COLUMN].apply(lambda x: os.path.join(cls.IMAGES_DIR, f"{x}.jpg"))
            print(f"Tải thành công {len(df)} bản ghi từ {cls.GT_PATH}")
            print(f"Ví dụ đường dẫn ảnh: {df['image_path'].iloc[0]}")
            return df
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file tại {cls.GT_PATH}")
            return None
        except KeyError:
            print(f"Lỗi: Không tìm thấy cột ID hình ảnh '{cls.ID_COLUMN}' trong file CSV.")
            return None

    @classmethod
    def split_data(cls, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        if (test_size + val_size) >= 1.0:
            raise ValueError("Tổng của test_size và val_size phải nhỏ hơn 1.0")

        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[cls.TARGET_COLUMN],
            random_state=random_state
        )

        relative_val_size = val_size / (1.0 - test_size)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            stratify=train_val_df[cls.TARGET_COLUMN],
            random_state=random_state
        )

        print("Hoàn tất chia dữ liệu:")
        print(f"  Tổng số mẫu: {len(df)}")
        print(f"  Tập Train:   {len(train_df)} mẫu")
        print(f"  Tập Val:     {len(val_df)} mẫu")
        print(f"  Tập Test:    {len(test_df)} mẫu")

        return train_df, val_df, test_df

    @classmethod
    def plot_class_distribution(cls, df: pd.DataFrame, title: str = ""):
        """
        Vẽ biểu đồ cột thể hiện số lượng mẫu của mỗi lớp.
        """
        plt.figure(figsize=(8, 5))
        class_counts = df[cls.TARGET_COLUMN].value_counts()
        
        class_labels = {0: 'Lành tính (0)', 1: 'Ác tính (1)'}
        class_counts.index = class_counts.index.map(class_labels.get)
        
        sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
        
        plt.title(title, fontsize=16)
        plt.xlabel("Loại tổn thương", fontsize=12)
        plt.ylabel("Số lượng mẫu", fontsize=12)
        
        for i, count in enumerate(class_counts.values):
            plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=11)
            
        plt.show()

    @classmethod
    def show_sample_images(cls, df: pd.DataFrame, n_samples_per_class):
        classes = df[cls.TARGET_COLUMN].unique()
        num_classes = len(classes)
        
        fig, axes = plt.subplots(num_classes, n_samples_per_class, figsize=(n_samples_per_class * 3, num_classes * 3))
        
        for i, item in enumerate(classes):
            image_paths = df[df[cls.TARGET_COLUMN] == item]['image_path'].tolist()
            
            sample_paths = random.sample(image_paths, min(n_samples_per_class, len(image_paths)))
            
            class_name = cls.CLASS_MAP.get(item, f"Lớp {item}")
            
            for j, img_path in enumerate(sample_paths):
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    ax = axes[i, j] if num_classes > 1 else axes[j]
                    ax.imshow(img)
                    ax.axis('off')
                    if j == 0:
                        ax.set_title(class_name, fontsize=14, loc='left', pad=10)
                except Exception as e:
                    print(f"Lỗi khi đọc ảnh {img_path}: {e}")
                    
        plt.suptitle(f"Ảnh mẫu ({n_samples_per_class} ảnh mỗi lớp)", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def get_augmentation_pipeline(img_size=256) -> albumentations.Compose:
        return albumentations.Compose([
            albumentations.Resize(img_size, img_size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            albumentations.CoarseDropout(
                max_holes=8, max_height=img_size//16, max_width=img_size//16,
                min_holes=4, min_height=img_size//20, min_width=img_size//20, 
                p=0.3
            )
        ])

    @classmethod
    def balance_training_data(cls, train_df: pd.DataFrame) -> pd.DataFrame:
        print("\nĐang bắt đầu cân bằng dữ liệu tập train...")
        class_counts = train_df[cls.TARGET_COLUMN].value_counts()
        
        majority_label = class_counts.idxmax()
        minority_label = class_counts.idxmin()
        
        majority_count = class_counts.loc[majority_label]
        minority_count = class_counts.loc[minority_label]
        
        n_to_generate = majority_count - minority_count
        
        if n_to_generate <= 0:
            print("Tập train đã cân bằng hoặc không xác định được lớp thiểu số.")
            return train_df

        print(f"Lớp đa số ({majority_label}): {majority_count} mẫu")
        print(f"Lớp thiểu số ({minority_label}): {minority_count} mẫu")
        print(f"-> Cần tạo thêm {n_to_generate} mẫu cho lớp '{minority_label}'")
        
        minority_paths = train_df[train_df[cls.TARGET_COLUMN] == minority_label]['image_path'].tolist()
        
        # Lấy pipeline tăng cường
        augmentation_pipeline = cls.get_augmentation_pipeline()
        
        new_records = []
        
        for i in tqdm(range(n_to_generate), desc="Đang tạo ảnh tăng cường"):
            original_path = random.choice(minority_paths)
            
            try:
                img = cv2.imread(original_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Lỗi đọc ảnh {original_path}: {e}")
                continue
            
            augmented = augmentation_pipeline(image=img)['image']
            
            original_filename = os.path.basename(original_path)
            new_filename = f"aug_{i}_{original_filename}"
            new_path = os.path.join(cls.AUG_IMAGES_DIR, new_filename)
            
            cv2.imwrite(new_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            
            new_record = {
                cls.ID_COLUMN: f"aug_{i}",
                cls.TARGET_COLUMN: minority_label,
                'image_path': new_path
            }
            new_records.append(new_record)

        df_augmented = pd.DataFrame(new_records)
        train_df_balanced = pd.concat([train_df, df_augmented], ignore_index=True)
        
        train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Đã tạo và thêm {len(new_records)} mẫu mới.")
        print(f"Tổng số mẫu train mới: {len(train_df_balanced)}")
        
        return train_df_balanced

    @classmethod
    def show_augmentation_effect(cls, df: pd.DataFrame, n_examples: int = 5):
        if df.empty:
            print("Lỗi: DataFrame rỗng, không thể chọn ảnh.")
            return
        
        original_image_paths = df[df['image_path'].str.contains(cls.IMAGES_DIR, na=False)]['image_path'].tolist()
        if not original_image_paths:
            original_image_paths = df['image_path'].tolist()
            
        random_path = random.choice(original_image_paths)
        
        try:
            original_img = cv2.imread(random_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Lỗi khi đọc ảnh {random_path}: {e}")
            return
        
        n_cols = n_examples + 1
        plt.figure(figsize=(n_cols * 3, 4))
        
        plt.subplot(1, n_cols, 1)
        plt.imshow(original_img)
        plt.title("Ảnh Gốc")
        plt.axis('off')

        augmentation_pipeline = cls.get_augmentation_pipeline()
        for i in range(n_examples):
            augmented = augmentation_pipeline(image=original_img)['image']
            
            plt.subplot(1, n_cols, i + 2)
            plt.imshow(augmented)
            plt.title(f"Tăng cường #{i+1}")
            plt.axis('off')
            
        plt.suptitle(f"Tác dụng của Augmentation Pipeline (từ {os.path.basename(random_path)})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @classmethod
    def run(cls, mode):
        if mode in ['raw', 'augment']:
            full_df = cls.load_isic_metadata()

            if full_df is not None:
                train_df, val_df, test_df = cls.split_data(df=full_df, test_size=0.2, val_size=0.1, random_state=42)
                final_train_df = train_df
                csv_dir = cls.CSV_OUTPUT_DIR

                cls.plot_class_distribution(train_df, title="Phân bổ Lớp Tập Train - BAN ĐẦU")

                if mode == 'augment':
                    print(f'TĂNG CƯỜNG DỮ LIỆU (lưu tại: {cls.AUG_IMAGES_DIR})')
                    os.makedirs(cls.AUG_IMAGES_DIR, exist_ok=True)

                    csv_dir = cls.CSV_OUTPUT_DIR_AUG
                    train_df_balanced = cls.balance_training_data(train_df=train_df)
                    final_train_df = train_df_balanced

                    cls.plot_class_distribution(train_df_balanced, title="Phân bổ Lớp Tập Train - ĐÃ Cân Bằng")
                    cls.show_sample_images(train_df_balanced, n_samples_per_class=4)
                    cls.show_augmentation_effect(train_df_balanced, n_examples=5)
                os.makedirs(csv_dir, exist_ok=True)

                train_csv_path = os.path.join(csv_dir, f'train_{mode}.csv')
                val_csv_path = os.path.join(csv_dir, 'val.csv')
                test_csv_path = os.path.join(csv_dir, 'test.csv')

                final_train_df.to_csv(train_csv_path, index=False)
                val_df.to_csv(val_csv_path, index=False)
                test_df.to_csv(test_csv_path, index=False)
                return True
        print(f'ReadData with mode = {mode} is not support.')
        return False
