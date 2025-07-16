import os
import shutil
import random

input_folder_path = 'features'
output_folder_base = 'features'

train_folder = os.path.join(output_folder_base, 'train')
val_folder = os.path.join(output_folder_base, 'val')
test_folder = os.path.join(output_folder_base, 'test')

for folder in [train_folder, val_folder, test_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

all_files = [f for f in os.listdir(input_folder_path) if f.endswith('.png') or f.endswith('.jpg')]
random.seed(42)
random.shuffle(all_files)

total_files = len(all_files)
train_split = int(total_files * 0.6)
val_split = int(total_files * 0.2)

train_files = all_files[:train_split]
val_files = all_files[train_split:train_split + val_split]
test_files = all_files[train_split + val_split:]


def copy_files(file_list, source_folder, destination_folder):
    for file_name in file_list:
        source = os.path.join(source_folder, file_name)
        destination = os.path.join(destination_folder, file_name)
        shutil.copyfile(source, destination)

copy_files(train_files, input_folder_path, train_folder)
copy_files(val_files, input_folder_path, val_folder)
copy_files(test_files, input_folder_path, test_folder)

print(f'file over：train_num={len(train_files)}, val_num={len(val_files)}, test_num={len(test_files)}')
