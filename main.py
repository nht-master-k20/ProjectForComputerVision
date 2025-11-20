import argparse
import sys
import os

# --- CẤU HÌNH CỐ ĐỊNH ---
TRAIN_CONFIG = {
    "image_size": 300,
    "batch_size": 32,
    "epochs": 15,
    "lr": 1e-3,
    # Luôn dùng dữ liệu đã qua xử lý
    "data_mode": "processed"
}

# --- IMPORT ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from scripts.ReadData import ReadData
    import models.EfficientNetB3_v1 as v1
    import models.EfficientNetB3_v2 as v2
    import models.EfficientNetB3_v3 as v3
except ImportError as e:
    print(f"⚠️ Lỗi Import: {e}")
    sys.exit(1)

MODEL_MAP = {'v1': v1, 'v2': v2, 'v3': v3}


def main():
    parser = argparse.ArgumentParser(description="Skin Cancer Pipeline (Simlified)")

    # Chỉ nhận 1 tham số: data HOẶC version model
    parser.add_argument("task", type=str,
                        choices=['data', 'v1', 'v2', 'v3'],
                        help="Chọn tác vụ: 'data' để xử lý ảnh, hoặc 'v3' để train model v3")

    args = parser.parse_args()

    # --- 1. XỬ LÝ DỮ LIỆU ---
    if args.task == 'data':
        # Gọi hàm run không cần tham số
        ReadData.run()

    # --- 2. TRAIN MODEL ---
    elif args.task in MODEL_MAP:
        print(f"\n🚀 Đang khởi động Train Model {args.task.upper()}...")
        print(f"   ⚙️ Config: {TRAIN_CONFIG}")

        module = MODEL_MAP[args.task]
        try:
            module.train(
                mode=TRAIN_CONFIG['data_mode'],  # Luôn là 'processed'
                image_size=TRAIN_CONFIG['image_size'],
                batch_size=TRAIN_CONFIG['batch_size'],
                epochs=TRAIN_CONFIG['epochs'],
                base_lr=TRAIN_CONFIG['lr']
            )
        except Exception as e:
            print(f"❌ Lỗi Training: {e}")


if __name__ == "__main__":
    main()