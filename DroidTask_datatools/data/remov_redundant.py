import os
import shutil

def delete_non_yaml_files(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            # if not name.endswith('.yaml') and not name.endswith('json'):
            if 'utg.yaml' not in name:
                os.remove(os.path.join(root, name))
        for name in dirs:
            dir_path = os.path.join(root, name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

folder_path = '/Users/haowen/Desktop/scientific_work/autodroid/code/DroidTask/data/raw_utgs'
delete_non_yaml_files(folder_path)