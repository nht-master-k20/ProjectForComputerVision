import argparse
import random
import sys
import os
import torch
import numpy as np

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- 2. CẤU HÌNH ---
CONFIG = {
    "image_size": 300,
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-3,
    "tta_steps": 5,
    "threshold": 0.07,
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

    if task_name == 'data':
        from scripts.ReadData2 import ReadData
        print(f"   ⚙️ [DATA] Running Clean -> Resize -> Split")
        ReadData.run()
        return

    # --- TRAINING (V1, V2, V3) ---
    if task_name in ['v1', 'v2', 'v3']:
        # Dynamic Import
        module = __import__(f"models.{task_name}", fromlist=['train'])

        print(f"   ⚙️ [TRAIN] Cấu hình: {CONFIG}")

        # --- QUAN TRỌNG: BỎ TRY-EXCEPT ĐỂ HIỆN LỖI THẬT ---
        module.train(
            image_size=CONFIG['image_size'],
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['epochs'],
            base_lr=CONFIG['lr']
        )
        return

    # --- INFERENCE (V4) ---
    if task_name == 'v4':
        from models import v4
        print(f"   ⚙️ [INFERENCE] TTA Steps: {CONFIG['tta_steps']}")
        v4.run_tta(
            image_size=CONFIG['image_size'],
            batch_size=CONFIG['batch_size'],
            tta_steps=CONFIG['tta_steps'],
            threshold=CONFIG['threshold']
        )
        return

    if task_name == 'v5':
        try:
            from models import v5
            print(f"   ⚙️ [V5 COMBO] TTA Steps: {CONFIG['tta_steps']}")
            v5.run_v5(
                image_size=CONFIG['image_size'],
                batch_size=CONFIG['batch_size'],
                tta_steps=CONFIG['tta_steps']
            )
        except ImportError as e:
            print(f"❌ Lỗi Import V5: {e}")
        except Exception as e:
            print(f"❌ Lỗi chạy V5: {e}")
        return

    print(f"❌ Tác vụ '{task_name}' không hợp lệ.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['data', 'v1', 'v2', 'v3', 'v4', 'v5'])
    args = parser.parse_args()
    run_task(args.task)