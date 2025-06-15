import os
import random
import glob

root_dir = "data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal"

all_txt_files = []
for cat in os.listdir(root_dir):
    cat_path = os.path.join(root_dir, cat)
    if os.path.isdir(cat_path):
        all_txt_files += glob.glob(os.path.join(cat_path, '*.txt'))

random.shuffle(all_txt_files)
n_total = len(all_txt_files)
n_train = int(0.85 * n_total)

train_list = all_txt_files[:n_train]
test_list = all_txt_files[n_train:]

with open(os.path.join(root_dir, '../train_split.txt'), 'w') as f:
    for item in train_list:
        f.write(item.split('shapenetcore_partanno_segmentation_benchmark_v0_normal/')[1] + '\n')

with open(os.path.join(root_dir, '../test_split.txt'), 'w') as f:
    for item in test_list:
        f.write(item.split('shapenetcore_partanno_segmentation_benchmark_v0_normal/')[1] + '\n')

print(f"Generated split: {len(train_list)} train, {len(test_list)} test")
