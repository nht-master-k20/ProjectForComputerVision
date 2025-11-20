import argparse
from models import EfficientNetB3, EfficientNetB3_v2
from scripts.read_data2 import ReadData

def parse_args_list(args_list, allowed=None):
    if allowed is None:
        allowed = []
    parsed = {}
    for arg in args_list:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key in allowed:
                parsed[key] = value
            else:
                print(f"Param {key} is not in allowed list ({allowed})")
        else:
            print(f"Invalid argument format '{arg}'. Expected 'key=value'")
    return parsed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Cancer Detection Training Pipeline')
    parser.add_argument("--read_data", nargs="*", required=False)
    parser.add_argument("--train_with_efficientnet", nargs="*", required=False)
    parser.add_argument("--train_with_efficientnet_v2", nargs="*", required=False)
    args = parser.parse_args()

    read_data_args_list = args.read_data or []
    train_eff_args_list = args.train_with_efficientnet or []
    train_eff_args_list_v2 = args.train_with_efficientnet_v2 or []

    if read_data_args_list:
        params = parse_args_list(read_data_args_list, allowed=['mode'])
        mode = params.get('mode')
        ReadData.run(mode=mode)
    elif train_eff_args_list:
        params = parse_args_list(train_eff_args_list, allowed=['mode', 'image_size', 'batch_size', 'epochs'])
        mode = params.get('mode')
        train_params = {
            'image_size': int(params.get('image_size', 300)),
            'batch_size': int(params.get('batch_size', 32)),
            'epochs': int(params.get('epochs', 10))
        }
        EfficientNetB3.train(mode=mode, **train_params)
    elif train_eff_args_list_v2:
        params = parse_args_list(train_eff_args_list_v2, allowed=['mode', 'image_size', 'batch_size', 'epochs'])
        mode = params.get('mode')
        train_params = {
            'image_size': int(params.get('image_size', 300)),
            'batch_size': int(params.get('batch_size', 32)),
            'epochs': int(params.get('epochs', 10))
        }
        EfficientNetB3_v2.train(mode=mode, **train_params)
