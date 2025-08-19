import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Total rasio harus 1.0"

    for class_name in os.listdir(source_dir):
        src_class = os.path.join(source_dir, class_name)
        if not os.path.isdir(src_class):
            continue  # skip file yang bukan folder kelas

        imgs = os.listdir(src_class)
        random.shuffle(imgs)

        total = len(imgs)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_imgs = imgs[:train_end]
        val_imgs = imgs[train_end:val_end]
        test_imgs = imgs[val_end:]

        # Buat folder tujuan
        for target_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)

        # Copy gambar
        for img in train_imgs:
            shutil.copy(os.path.join(src_class, img), os.path.join(train_dir, class_name, img))

        for img in val_imgs:
            shutil.copy(os.path.join(src_class, img), os.path.join(val_dir, class_name, img))

        for img in test_imgs:
            shutil.copy(os.path.join(src_class, img), os.path.join(test_dir, class_name, img))

    print("Dataset split selesai.")

# Contoh penggunaan:
split_dataset(
    source_dir='data/raw', 
    train_dir='data/train', 
    val_dir='data/validation', 
    test_dir='data/test',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
