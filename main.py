import argparse

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
                print(f"Param {key} is not in allowed list: {allowed}")
        else:
            print(f"Invalid argument format '{arg}'. Expected 'key=value'")
    print(parsed)
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Cancer Detection Training Pipeline')
    parser.add_argument("--read_data", nargs=1, required=False, help='Read data')
    parser.add_argument("--train_with_method_1", nargs="*", required=False, help='Train with method 1')
    parser.add_argument("--train_with_method_2", nargs="*", required=False, help='Train with method 2')
    args = parser.parse_args()

    read_data_args_list = args.read_data or []
    train_with_method_1_args_list = args.train_with_method_1 or []
    train_with_method_2_args_list = args.train_with_method_2 or []

    if read_data_args_list:
        params = parse_args_list(read_data_args_list, ['mode'])
        mode = params.get('mode')
        ReadData.run(mode=mode)
    elif train_with_method_1_args_list:
        params = parse_args_list(train_with_method_1_args_list, [])
        print('METHOD 1')
    elif train_with_method_2_args_list:
        params = parse_args_list(train_with_method_2_args_list, [])
        print('METHOD 2')
    else:
        print('Cannot find any argument. Supported arguments:')
        print('  --read_data')
        print('  --train_with_method_1')
        print('  --train_with_method_2')