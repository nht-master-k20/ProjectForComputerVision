import argparse
from scripts.read_data import ReadData

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs="+", help="List scripts in order", required=True)
    args = parser.parse_args()

    for script in args.run:
        if script == "read_data":
            ReadData.run()
        else:
            print(f"Can not found this scripts: {script}")
