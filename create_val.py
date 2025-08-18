import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    for class_name in os.listdir(source_dir):
        src_class = os.path.join(source_dir, class_name)
        imgs = os.listdir(src_class)
        random.shuffle(imgs)
        
        train_len = int(len(imgs) * split_ratio)
        train_imgs = imgs[:train_len]
        val_imgs = imgs[train_len:]

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(src_class, img), os.path.join(train_dir, class_name, img))

        for img in val_imgs:
            shutil.copy(os.path.join(src_class, img), os.path.join(val_dir, class_name, img))

# Example usage:
split_dataset('data/raw', 'data/train', 'data/val')
