import os
import zipfile
import subprocess
from pathlib import Path
import pandas as pd

url = "https://www.kaggle.com/api/v1/datasets/download/mohamedhanyyy/chest-ctscan-images"
output_dir = "dataset"
output_file = os.path.join(output_dir, "archive.zip")

if not os.path.exists(output_file):
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading dataset...")
    try:
        subprocess.run(["curl", "-L", "-o", output_file, url], check=True)
    except subprocess.CalledProcessError as e:
        print("Download failed:", e)
        exit(1)

    print("Extracting dataset...")
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    DATASET_PATH = Path('dataset/Data')

    train_path = DATASET_PATH / 'train'
    valid_path = DATASET_PATH / 'valid'
    test_path = DATASET_PATH / 'test'

    def get_images_path(path):
        images = []
        for cls in os.listdir(path):
            for image in os.listdir(path / cls):
                images.append([path / cls / image, cls])
        return images

    train_images = get_images_path(train_path)
    valid_images = get_images_path(valid_path)
    test_images = get_images_path(test_path)

    train_df = pd.DataFrame(columns=['path', 'class'], data=train_images)
    valid_df = pd.DataFrame(columns=['path', 'class'], data=valid_images)
    test_df = pd.DataFrame(columns=['path', 'class'], data=test_images)

    train_df.to_csv(DATASET_PATH / 'train_paths.csv', index=False)
    valid_df.to_csv(DATASET_PATH / 'valid_paths.csv', index=False)
    test_df.to_csv(DATASET_PATH / 'test_paths.csv', index=False)

    print(f"Extraction complete. Dataset saved in '{output_dir}' directory.")
    os.remove(output_file)
else:
    print("Dataset already exists.")
