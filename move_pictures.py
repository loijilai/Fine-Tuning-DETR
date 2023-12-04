import os
import json
import shutil

# Path to your folders and JSON file
folder_a = '/tmp2/loijilai/cvpdl/hw3/Fine-Tuning-DETR/GLIGEN/generation_samples/text_img_grounding'
folder_b = '/project/dsp/loijilai/cvpdl/hw1_dataset_text_image/train'
json_file = '/project/dsp/loijilai/cvpdl/hw1_dataset_text_image/annotations/newtrain.json'

# Read JSON file
with open(json_file, 'r') as file:
    data = json.load(file)

# Assuming the JSON structure is a list of image names
images = data['images']
for id in range(448, 528):
    image_name = images[id]['file_name']
    source_path = os.path.join(folder_a, image_name)
    destination_path = os.path.join(folder_b, image_name)

    # Check if the image exists in folder A and then move it to folder B
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Copy {image_name} from Folder A to Folder B.")
    else:
        print(f"{image_name} not found in Folder A.")
