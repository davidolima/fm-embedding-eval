import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract JSON files from a directory and save them in a new directory.')
    parser.add_argument('--json_only_path', type=str, required=True, help='Path to save the extracted JSON files.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the directory containing JSON files.')
    args = parser.parse_args()
    
    os.makedirs(args.json_only_path, exist_ok=True)

    for path, dirs, files in os.walk(args.json_path):
        for file in files:
            if file.endswith('.json'):
                out_dir = os.path.join(args.json_only_path, *[x for x in path.split('/')[2:]])
                os.makedirs(out_dir, exist_ok=True)
                shutil.copyfile(os.path.join(path, file), os.path.join(out_dir,file))