CSV_PATH = 'ISIC_2024_Training_GroundTruth.csv'
IMAGES_DIR = 'ISIC_2024_Training_Input'
ID_COLUMN = 'isic_id'
TARGET_COLUMN = 'malignant'

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_isic_metadata(
    csv_path: str, 
    images_dir: str, 
    img_id_col: str = 'isic_id', 
    img_ext: str = '.jpg'
) -> pd.DataFrame:

    try:
        df = pd.read_csv(csv_path)
        
        df['image_path'] = df[img_id_col].apply(
            lambda x: os.path.join(images_dir, f"{x}{img_ext}")
        )
        
        print(f"Tải thành công {len(df)} bản ghi từ {csv_path}")
        print(f"Ví dụ đường dẫn ảnh: {df['image_path'].iloc[0]}")
        
        return df
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại {csv_path}")
        return None
    except KeyError:
        print(f"Lỗi: Không tìm thấy cột ID hình ảnh '{img_id_col}' trong file CSV.")
        return None

def split_data(df: pd.DataFrame, 
    target_col: str, 
    test_size: float = 0.2, 
    val_size: float = 0.1, 
    random_state: int = 42
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    if (test_size + val_size) >= 1.0:
        raise ValueError("Tổng của test_size và val_size phải nhỏ hơn 1.0")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )

    relative_val_size = val_size / (1.0 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[target_col],
        random_state=random_state
    )

    print("Hoàn tất chia dữ liệu:")
    print(f"  Tổng số mẫu: {len(df)}")
    print(f"  Tập Train:   {len(train_df)} mẫu")
    print(f"  Tập Val:     {len(val_df)} mẫu")
    print(f"  Tập Test:    {len(test_df)} mẫu")
    
    return train_df, val_df, test_df



full_df = load_isic_metadata(
    csv_path=CSV_PATH, 
    images_dir=IMAGES_DIR,
    img_id_col=ID_COLUMN
)

if full_df is not None:
    train_df, val_df, test_df = split_data(
        df=full_df,
        target_col=TARGET_COLUMN,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    print("\n--- Phân bổ nhãn trong tập Train ---")
    print(train_df[TARGET_COLUMN].value_counts(normalize=True))

    print("\n--- Phân bổ nhãn trong tập Validation ---")
    print(val_df[TARGET_COLUMN].value_counts(normalize=True))

    print("\n--- Phân bổ nhãn trong tập Test ---")
    print(test_df[TARGET_COLUMN].value_counts(normalize=True))