# %%
#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict
import json

# %%
# {image_id: (annotation:list)}
image_ids_annotations = defaultdict(list)

# Load annotations
annoataion_path = "/home/loijilai/CS-hub/DL/cvpdl/deter/hw1_dataset/annotations/val.json"
file = open(annoataion_path)
anns = json.load(file)

# Add into data structure
for ann in anns['annotations']:
    image_id = ann['image_id'] # Are integers
    image_ids_annotations[image_id].append(ann)
# %%
# {category_id: category_name} e.g. {0:None}, {1:fish}
category_id_to_name = {}
for ann in anns['categories']:
    category_id_to_name[ann['id']] = ann['name']
# %%

image_path = "/home/loijilai/CS-hub/DL/cvpdl/deter/hw1_dataset/valid/IMG_2277_jpeg_jpg.rf.86c72d6192da48d941ffa957f4780665.jpg"
img = Image.open(image_path)
fig, ax = plt.subplots()

image_anns = image_ids_annotations[44] # This picture has id 44

for image_ann in image_anns:
    bbox = image_ann['bbox']
    name = category_id_to_name[image_ann['category_id']]

    x = float(bbox[0])
    y = float(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

    plt.text(x, y, name, fontdict={'fontsize': 10, 'color': 'white', 'backgroundcolor': 'red'})
    bb = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(bb)
ax.imshow(img)
plt.show()
