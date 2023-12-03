import json
with open('/project/dsp/loijilai/cvpdl/hw1_dataset/annotations/for_gligen.json') as f:
    data = json.load(f)

with open('/project/dsp/loijilai/cvpdl/hw1_dataset/annotations/train.json') as f:
    train_data = json.load(f)

images = train_data['images']
annotations = train_data['annotations']
image_id = len(images)
anno_id = len(annotations)
name_idx = 0
for d in data:
    if d["label"] == "fish" or d["label"] == "shark" or d["label"] == "stingray":
        name_idx += 4
        continue

    for i in range(4):
        images.append(
            {"id":image_id, 
            "license": 1,
            "file_name": d["file_name"] + str(name_idx) + ".png",
            "height": d["height"],
            "width": d["width"],
            "date_captured": ""
            }
        )
        for loc in d["locations"]:
            loc = [loc*512 for loc in loc]
            annotations.append(
                {"id": anno_id,
                "image_id": image_id,
                "category_id": d["category_id"],
                "bbox": [loc[0], loc[1], loc[2]-loc[0], loc[3]-loc[1]],
                "area": loc[2]*512*loc[3]*512,
                "segmentation": [],
                "iscrowd": 0
                }
            )
            anno_id += 1
        image_id += 1
        name_idx += 1

print("Writing to file...")
with open('/project/dsp/loijilai/cvpdl/hw1_dataset/annotations/newtrain.json', 'w') as f:
    # indent=2 is not necessary but makes the file human-readable
    json.dump(train_data, f, indent=2)
print("Done!")