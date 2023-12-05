import os
import json
import shutil

label_to_image_id = {
    "fish": 1,
    "jellyfish": 6,
    "penguin": 11,
    "puffin": 16,
    "shark": 21,
    "starfish": 26,
    "stingray": 31,
}

source_path = "/project/dsp/loijilai/cvpdl/generation_samples/text_img_grounding"
dest_path = "/project/dsp/loijilai/cvpdl/generation_samples/vis"
anno_path = "/project/dsp/loijilai/cvpdl/hw1_dataset/annotations/for_gligen.json"

with open(anno_path) as f:
    image_data = json.load(f)

gen_id = 0
for img in image_data:
    file_name = img["file_name"]
    label = img["label"]
    image_id = label_to_image_id[label]

    source_img = f"{file_name}{gen_id}.png"
    shutil.copy(os.path.join(source_path, source_img), os.path.join(dest_path, f"image{image_id}.png"))
    gen_id += 4
    label_to_image_id[label] += 1