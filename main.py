import argparse
import sys
import os

# --- 1. CẤU HÌNH CHUNG ---
CONFIG = {
    "image_size": 300,
    "batch_size": 32,
    "epochs": 20,
    "lr": 1e-3,
    # LƯU Ý: Đổi thành 'processed' để khớp với file 'processed_train.csv' do ReadData sinh ra
    "data_mode": "processed"
}

# --- 2. SETUP PATHS ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))


# --- 3. XỬ LÝ CHÍNH ---
def run_task(task_name):
    print(f"\n[MAIN] 🚀 Đang khởi chạy tác vụ: {task_name.upper()}")

    # --- TRƯỜNG HỢP 1: CHUẨN BỊ DỮ LIỆU ---
    if task_name == 'data':
        try:
            # Lazy import
            from scripts.ReadData import ReadData

            print(f"   ⚙️ Cấu hình: Chạy quy trình chuẩn (Split -> Clean -> Augment)")

            # --- SỬA LỖI TẠI ĐÂY ---
            # Gọi hàm run() không tham số
            ReadData.run()

        except ImportError as e:
            print(f"❌ Lỗi Import ReadData: {e}")
        except Exception as e:
            print(f"❌ Lỗi xử lý dữ liệu: {e}")
            raise e
        return

    # --- TRƯỜNG HỢP 2: HUẤN LUYỆN (Lazy Import Model) ---
    module = None
    try:
        if task_name == 'v1':
            import models.EfficientNetB3_v1 as module
        elif task_name == 'v2':
            import models.EfficientNetB3_v2 as module
        elif task_name == 'v3':
            import models.EfficientNetB3_v3 as module
        else:
            print(f"❌ Lệnh '{task_name}' không hợp lệ. Chọn: data, v1, v2, v3")
            return
    except ImportError as e:
        print(f"❌ Lỗi Import Model {task_name}: {e}")
        print("👉 Kiểm tra xem file model có tồn tại trong thư mục 'models/' chưa.")
        return

    # Chạy Training
    if module:
        print(f"   ⚙️ Cấu hình Train: {CONFIG}")
        try:
            module.train(
                mode=CONFIG['data_mode'],  # Sẽ truyền vào là 'processed'
                image_size=CONFIG['image_size'],
                batch_size=CONFIG['batch_size'],
                epochs=CONFIG['epochs'],
                base_lr=CONFIG['lr']
            )
        except Exception as e:
            print(f"❌ Lỗi Training: {e}")
            raise e


# --- 4. ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Skin Cancer CLI')

    parser.add_argument('task', type=str,
                        choices=['data', 'v1', 'v2', 'v3'],
                        help='Chọn tác vụ: data (xử lý ảnh), hoặc version model (v1, v2, v3)')

    args = parser.parse_args()

    run_task(args.task)