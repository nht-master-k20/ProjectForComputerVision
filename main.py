import argparse
import sys
import os

# --- IMPORT SCRIPTS ---
# Đảm bảo Python tìm thấy các module trong thư mục con
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from scripts.ReadData import ReadData
    import models.EfficientNetB3_v1 as v1
    import models.EfficientNetB3_v2 as v2
    import models.EfficientNetB3_v3 as v3
    import models.EfficientNetB3_v4 as v4
except ImportError as e:
    print(f"⚠️ Lỗi Import: {e}")
    print("👉 Hãy đảm bảo cấu trúc thư mục: scripts/ReadData.py và models/EfficientNetB3_vX.py tồn tại.")
    sys.exit(1)

# --- CONFIGURATION ---
# Ánh xạ tên version sang module tương ứng
MODEL_MAP = {
    'v1': v1,
    'v2': v2,
    'v3': v3,
    'v4': v4
}


def run_data_prep(args):
    """Xử lý lệnh chuẩn bị dữ liệu"""
    print(f"\n[DATA] 🛠️ Đang chạy chuẩn bị dữ liệu...")
    print(f"   - Mode: {args.mode}")
    print(f"   - Clean: {args.clean}")

    try:
        success = ReadData.run(mode=args.mode, clean=args.clean)
        if success:
            print("\n✅ Chuẩn bị dữ liệu hoàn tất!")
        else:
            print("\n❌ Có lỗi xảy ra trong quá trình xử lý dữ liệu.")
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng: {e}")


def run_training(args):
    """Xử lý lệnh huấn luyện"""
    print(f"\n[TRAIN] 🚀 Khởi động huấn luyện...")
    print(f"   - Version: {args.version}")
    print(f"   - Mode: {args.mode}")
    print(f"   - Config: {args.image_size}px | Batch: {args.batch_size} | Epochs: {args.epochs}")

    if args.version not in MODEL_MAP:
        print(f"❌ Version '{args.version}' không hợp lệ. Các lựa chọn: {list(MODEL_MAP.keys())}")
        return

    selected_module = MODEL_MAP[args.version]

    # Kiểm tra xem module có hàm train không
    if not hasattr(selected_module, 'train'):
        print(f"❌ Module {args.version} thiếu hàm 'train'.")
        return

    try:
        # Gọi hàm train với các tham số đã parse
        selected_module.train(
            mode=args.mode,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            base_lr=args.lr
        )
    except Exception as e:
        print(f"❌ Lỗi trong quá trình Training: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Cancer Detection Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Chọn tác vụ: data hoặc train', required=True)

    # --- 1. Lệnh DATA ---
    # Ví dụ: python main.py data --mode clean --clean
    parser_data = subparsers.add_parser('data', help='Chạy xử lý dữ liệu (ReadData)')
    parser_data.add_argument('--mode', type=str, default='raw', choices=['raw', 'augment', 'clean'],
                             help='Chế độ xử lý')
    parser_data.add_argument('--clean', action='store_true', help='Bật cờ này để thực hiện xóa lông')
    parser_data.add_argument('--no-clean', action='store_false', dest='clean', help='Tắt xóa lông')
    parser_data.set_defaults(func=run_data_prep)

    # --- 2. Lệnh TRAIN ---
    # Ví dụ: python main.py train --version v4 --mode clean --epochs 20
    parser_train = subparsers.add_parser('train', help='Chạy huấn luyện mô hình')
    parser_train.add_argument('--version', type=str, required=True, choices=list(MODEL_MAP.keys()),
                              help='Chọn phiên bản (v1-v4)')
    parser_train.add_argument('--mode', type=str, default='clean', choices=['raw', 'clean', 'augment'],
                              help='Loại dữ liệu đầu vào')
    parser_train.add_argument('--image_size', type=int, default=300, help='Kích thước ảnh')
    parser_train.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    parser_train.add_argument('--epochs', type=int, default=10, help='Số lượng epochs')
    parser_train.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser_train.set_defaults(func=run_training)

    # Xử lý arguments
    args = parser.parse_args()
    args.func(args)
