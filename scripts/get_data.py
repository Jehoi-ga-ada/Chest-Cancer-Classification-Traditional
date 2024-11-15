import os
import zipfile
import subprocess

url = "https://www.kaggle.com/api/v1/datasets/download/mohamedhanyyy/chest-ctscan-images"
output_dir = "dataset"
output_file = os.path.join(output_dir, "archive.zip")

os.makedirs(output_dir, exist_ok=True)

print("Downloading dataset...")
try:
    subprocess.run(["curl", "-L", "-o", output_file, url], check=True)
except subprocess.CalledProcessError as e:
    print("Download failed:", e)
    exit(1)

if os.path.exists(output_file):
    print("Extracting dataset...")
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"Extraction complete. Dataset saved in '{output_dir}' directory.")
    os.remove(output_file)
else:
    print("Download failed. Please check the URL and try again.")
