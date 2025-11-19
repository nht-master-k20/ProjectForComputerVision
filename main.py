import argparse

from models import EfficientNetB3
from scripts.read_data import ReadData


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
    parser.add_argument("--read_data", nargs="*", required=False, help='Read data')
    parser.add_argument("--train_with_method_1", nargs="*", required=False, help='Train with method 1')
    parser.add_argument("--train_with_efficientnet", nargs="*", required=False, help='Train with EfficientNetB3')
    args = parser.parse_args()

    read_data_args_list = args.read_data or []
    train_with_method_1_args_list = args.train_with_method_1 or []
    train_with_efficientnet_args_list = args.train_with_efficientnet or []

    if read_data_args_list:

        params = parse_args_list(read_data_args_list, allowed=['mode', 'clean'])
        
        mode = params.get('mode')
        clean = params.get('clean')
        print(f"Debug parsed params: mode={mode}, clean={clean}")
        is_clean = True if clean == '1' else False
        
        ReadData.run(mode=mode, clean=is_clean) 
        print(f"Calling ReadData with: mode={mode}, clean={is_clean}")

    elif train_with_method_1_args_list:
        params = parse_args_list(train_with_method_1_args_list, ['epochs', 'batches'])
        print('METHOD 1')
    elif train_with_efficientnet_args_list:
        params = parse_args_list(train_with_efficientnet_args_list, ['mode', 'image_size', 'batch_size', 'epochs'])

        mode = params.get('mode')
        image_size = params.get('image_size')
        batch_size = params.get('batch_size')
        epochs = params.get('epochs')
        print(f'EfficientNetB3 Model: mode={mode}, image_size={image_size}, batch_size={batch_size}, epochs={epochs}')
        EfficientNetB3.train(mode=mode, image_size=image_size, batch_size=batch_size, epochs=epochs)
    else:
        print('Cannot find any argument. Supported arguments:')
        print('  --read_data')
        print('  --train_with_method_1')
        print('  --train_with_efficientnet')