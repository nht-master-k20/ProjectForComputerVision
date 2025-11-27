import argparse
import random
import sys
import os
import torch
import numpy as np

# --- 1. SETUP PATHS (QUAN TRỌNG NHẤT) ---
# Thêm thư mục hiện tại vào sys.path để Python tìm thấy 'scripts' và 'models'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- 2. CẤU HÌNH CHUNG ---
CONFIG = {
    "image_size": 300,
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-3,
    "tta_steps": 5,  # Dành riêng cho V4
    "seed": 42
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 3. XỬ LÝ CHÍNH ---
def run_task(task_name):
    print(f"\n[MAIN] 🚀 Kích hoạt tác vụ: {task_name.upper()}")
    seed_everything(CONFIG['seed'])

    # --- TÁC VỤ 1: CHUẨN BỊ DỮ LIỆU ---
    if task_name == 'data':
        try:
            # Import từ scripts/prepare_data.py
            from scripts.prepare_data import ReadData
            print(f"   ⚙️ [DATA] Bắt đầu quy trình: Clean -> Resize -> Split")
            ReadData.run()
        except ImportError:
            print("❌ Lỗi: Không tìm thấy file 'scripts/prepare_data.py'")
        except Exception as e:
            print(f"❌ Lỗi xử lý dữ liệu: {e}")
        return

    # --- TÁC VỤ 2: TRAINING (V1, V2, V3) ---
    if task_name in ['v1', 'v2', 'v3']:
        try:
            # Dynamic Import: models.v1, models.v2, ...
            # Giả định bạn lưu file code train vào folder 'models/' với tên v1.py, v2.py...
            module = __import__(f"models.{task_name}", fromlist=['train'])

            print(f"   ⚙️ [TRAIN] Cấu hình: {CONFIG}")
            module.train(
                image_size=CONFIG['image_size'],
                batch_size=CONFIG['batch_size'],
                epochs=CONFIG['epochs'],
                base_lr=CONFIG['lr']
            )
        except ImportError as e:
            print(f"❌ Lỗi Import: Không tìm thấy file 'models/{task_name}.py'.\n   Chi tiết: {e}")
        except AttributeError:
            print(f"❌ Lỗi Code: File 'models/{task_name}.py' không có hàm 'train()'.")
        except Exception as e:
            print(f"❌ Lỗi trong quá trình Train {task_name}: {e}")
            raise e
        return

    # --- TÁC VỤ 3: INFERENCE TTA (V4) ---
    if task_name == 'v4':
        try:
            # V4 là Inference, gọi hàm run_tta
            from models import v4

            print(f"   ⚙️ [INFERENCE] Cấu hình TTA: {CONFIG['tta_steps']} steps")
            v4.run_tta(
                image_size=CONFIG['image_size'],
                batch_size=CONFIG['batch_size'],
                tta_steps=CONFIG['tta_steps']
            )
        except ImportError as e:
            print(f"❌ Lỗi Import: Không tìm thấy file 'models/v4.py'.\n   Chi tiết: {e}")
        except Exception as e:
            print(f"❌ Lỗi trong quá trình Inference V4: {e}")
            raise e
        return

    print(f"❌ Tác vụ '{task_name}' không hợp lệ.")


# --- 4. ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Cancer Classification Pipeline')

    parser.add_argument('task', type=str,
                        choices=['data', 'v1', 'v2', 'v3', 'v4'],
                        help='Chọn tác vụ để chạy: data (xử lý), v1-v3 (train), v4 (inference TTA)')

    args = parser.parse_args()

    run_task(args.task)