import os
import json
import shutil
from PIL import Image

# Path to your folders and JSON file
folder_a = '/project/dsp/loijilai/cvpdl/hw1_dataset/train'
folder_b = '/project/dsp/loijilai/cvpdl/140photos'
json_file = '/tmp2/loijilai/cvpdl/hw3/Fine-Tuning-DETR/test.json'

# Read JSON file
with open(json_file, 'r') as file:
    data = json.load(file)

# Assuming the JSON structure is a list of image names
for d in data:
    image_name = d['file_name']
    source_path = os.path.join(folder_a, image_name)
    destination_path = os.path.join(folder_b, image_name)

    # Check if the image exists in folder A and then move it to folder B
    if os.path.exists(source_path):
        with Image.open(source_path) as img:
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            img.save(destination_path)
        print(f"Copy and resize {image_name} from Folder A to Folder B.")
    else:
        print(f"{image_name} not found in Folder A.")
