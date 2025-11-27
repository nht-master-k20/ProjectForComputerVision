import pandas as pd
from sklearn.model_selection import train_test_split
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import functools


class ReadData:
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    GT_PATH = 'dataset/ISIC_2024_Training_GroundTruth.csv'
    IMAGES_DIR = 'dataset/ISIC_2024_Training_Input'

    # Thư mục lưu ảnh sau khi đã Clean (Xóa lông + Resize)
    OUTPUT_IMG_DIR = 'dataset/ISIC_Processed_Images'

    # Thư mục lưu file CSV (metadata)
    CSV_OUTPUT_DIR = 'dataset_splits'

    ID_COLUMN = 'isic_id'
    TARGET_COLUMN = 'malignant'

    @classmethod
    def load_metadata(cls):
        try:
            df = pd.read_csv(cls.GT_PATH)
            # Tạo đường dẫn đầy đủ tới ảnh gốc
            df['image_path'] = df[cls.ID_COLUMN].apply(lambda x: os.path.join(cls.IMAGES_DIR, f"{x}.jpg"))
            print(f"✅ Đã tải metadata: {len(df)} ảnh.")
            return df
        except Exception as e:
            print(f"❌ Lỗi tải CSV gốc: {e}")
            return None

    @classmethod
    def split_data(cls, df):
        """Chia Stratified: Train/Val/Test"""
        # Giữ nguyên logic chia tập dữ liệu của bạn
        train_val, test = train_test_split(df, test_size=0.2, stratify=df[cls.TARGET_COLUMN], random_state=42)
        train, val = train_test_split(train_val, test_size=0.125, stratify=train_val[cls.TARGET_COLUMN],
                                      random_state=42)

        print(f"📊 Thống kê: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return train, val, test

    # --- WORKER XỬ LÝ ẢNH (CLEAN ONLY) ---
    @staticmethod
    def remove_hair(image):
        """Thuật toán xóa lông (Giữ nguyên)"""
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
        """
        Đọc ảnh gốc -> Resize 300x300 -> Xóa lông -> Lưu ra file mới.
        Mục đích: Giảm tải cho CPU khi train (không phải resize/xóa lông on-the-fly).
        """
        idx, row = row_tuple
        src_path = row['image_path']
        fname = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, fname)

        # Resume: Nếu ảnh đã xử lý rồi thì bỏ qua
        if os.path.exists(dst_path): return dst_path

        try:
            img = cv2.imread(src_path)
            if img is not None:
                # Resize về 300x300 để nhẹ ổ cứng và load nhanh hơn
                img = cv2.resize(img, (300, 300))

                # Xóa lông (Pre-processing tĩnh)
                clean = ReadData.remove_hair(img)

                cv2.imwrite(dst_path, clean)
                return dst_path
        except:
            pass
        return src_path  # Fallback nếu lỗi

    @classmethod
    def clean_dataset(cls, df, folder_name):
        """Chạy đa luồng để clean ảnh"""
        save_dir = os.path.join(cls.OUTPUT_IMG_DIR, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"🧹 Đang xử lý (Clean & Resize) {len(df)} ảnh vào '{folder_name}'...")

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            func = functools.partial(cls._process_worker, output_dir=save_dir)
            new_paths = list(tqdm(ex.map(func, df.iterrows()), total=len(df)))

        # Cập nhật đường dẫn trong DataFrame sang ảnh đã clean
        df_new = df.copy()
        df_new['image_path'] = new_paths
        return df_new

    @classmethod
    def run(cls):
        print("🚀 Bắt đầu quy trình chuẩn bị dữ liệu (Online Augmentation Ready)...")

        # 1. Load
        df = cls.load_metadata()
        if df is None: return False

        # 2. Split
        train, val, test = cls.split_data(df)

        # 3. Clean (Chỉ Pre-process tĩnh, KHÔNG Augment sinh ảnh mới)
        train = cls.clean_dataset(train, 'Train_Clean')
        val = cls.clean_dataset(val, 'Val_Clean')
        test = cls.clean_dataset(test, 'Test_Clean')

        # 4. Save CSV (Lưu danh sách file gốc + đường dẫn ảnh clean)
        os.makedirs(cls.CSV_OUTPUT_DIR, exist_ok=True)
        print(f"💾 Đang lưu CSV vào {cls.CSV_OUTPUT_DIR}...")

        train.to_csv(f'{cls.CSV_OUTPUT_DIR}/processed_train.csv', index=False)
        val.to_csv(f'{cls.CSV_OUTPUT_DIR}/processed_val.csv', index=False)
        test.to_csv(f'{cls.CSV_OUTPUT_DIR}/processed_test.csv', index=False)

        print("✅ Hoàn tất! Dữ liệu đã sẵn sàng cho Dataset Class.")
        return True


if __name__ == '__main__':
    ReadData.run()