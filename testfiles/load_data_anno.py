import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict
import json

# Load annotations
annoataion_path = "/project/dsp/loijilai/cvpdl/hw1_dataset/annotations/newtrain.json"
with open(annoataion_path) as file:
    annotation_file = json.load(file)

image_path = "/tmp2/loijilai/cvpdl/hw3/Fine-Tuning-DETR/GLIGEN/generation_samples/text_grounding_template_1/IMG_8590_MOV-4_jpg.rf.1691f0958ffea266daa9011c203cd726.jpg63.png"
file_name = image_path.split("/")[-1]
image_id = [x["id"] for x in annotation_file['images'] if x['file_name'] == file_name][0]
annotations = [x for x in annotation_file['annotations'] if x['image_id'] == image_id]
img = Image.open(image_path)
fig, ax = plt.subplots(1)
ax.imshow(img)

# plot the image and bounding boxes
for ann in annotations:
    bbox = ann['bbox']
    name = ann['category_id']

    x = float(bbox[0])
    y = float(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

print("Saving image...")
plt.savefig("test.png")
print("Done!")