import json
from PIL import Image
import os
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

with open("/project/dsp/loijilai/cvpdl/hw1_dataset/annotations/for_blip2.json") as f:
    data = json.load(f)

# image captioning
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b-coco",
                                        cache_dir="/project/dsp/loijilai/cvpdl/.cache")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b-coco", 
                                                    cache_dir="/project/dsp/loijilai/cvpdl/.cache",
                                                    load_in_8bit = True,
                                                    device_map = "auto"
                                                    )

device = "cuda" if torch.cuda.is_available() else "cpu"

image_path = "/project/dsp/loijilai/cvpdl/hw1_dataset/train"
for d in data:
    image = Image.open(os.path.join(image_path, d["file_name"]))
    inputs = processor(image, text="", return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    gen_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if d["label"] == "puffin":
        gen_text = gen_text.replace("penguin", "puffin")
    d["prompt"] = gen_text

print("Writing to file...")
with open('/project/dsp/loijilai/cvpdl/hw1_dataset/annotations/for_gligen.json', 'w') as f:
    # indent=2 is not necessary but makes the file human-readable
    json.dump(data, f, indent=2)
print("Done!")