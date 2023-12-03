import json

id_to_phrases = {
    1: "fish",
    2: "jellyfish",
    3: "penguin",
    4: "puffin",
    5: "shark",
    6: "starfish",
    7: "stingray",
}

def normalize(bbox, height, width):
    x, y, w, h = bbox
    # ronud all result to 2 decimal places
    result = [x/width, y/height, (x+w)/width, (y+h)/height]
    return [round(i, 2) for i in result]

with open('/project/dsp/loijilai/cvpdl/hw1_dataset/annotations/train.json') as f:
    data = json.load(f)

categories = data['categories'] # list
images = data['images'] # list
annotations = data['annotations'] # list

result = []

i = 0
# {1: 69, 2: 6, 3: 23, 4: 33, 5: 21, 6: 27, 7: 24}
num_of_each_category = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0
}
for img in images:
    img_id = img['id']
    anno = []
    bbox = []
    category = []

    while i < len(annotations) and annotations[i]['image_id'] == img_id:
        anno.append(annotations[i])
        bbox.append(annotations[i]['bbox'])
        category.append(annotations[i]['category_id'])
        i += 1

    if len(set(category)) == 1:
        if num_of_each_category[category[0]] >= 5:
            continue
        if((category[0] == 2 and len(bbox) <= 6) or len(bbox) <= 1):
            result.append(
                {"file_name": img["file_name"],
                    "height": 512,
                    "width": 512,
                    "category_id": category[0],
                    # "bbox": bbox, # a list of bbox
                    "label": id_to_phrases[category[0]], 
                    "locations": [normalize(b, img["height"], img["width"]) for b in bbox], 
                    "prompt": ""
                    }
            )
            num_of_each_category[category[0]] += 1
        # result['images'].append(img)
        # result['annotations'].extend(anno)

print(num_of_each_category)

print("Writing to file...")
with open('/project/dsp/loijilai/cvpdl/hw1_dataset/annotations/for_blip2.json', 'w') as f:
    # indent=2 is not necessary but makes the file human-readable
    json.dump(result, f, indent=2)
print("Done!")