from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

image = Image.open("/project/dsp/loijilai/cvpdl/hw1_dataset/train/IMG_2382_jpeg_jpg.rf.b431ad0ed94761ef82281dbe844170cc.jpg").convert("RGB")


processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b",
                                          cache_dir="/project/dsp/loijilai/cvpdl/.cache")
# by default `from_pretrained` loads the weights in float32
# we load in float16 instead to save memory
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", 
                                                      cache_dir="/project/dsp/loijilai/cvpdl/.cache",
                                                    #   torch_dtype=torch.float16
                                                      )

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

inputs = processor(image, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

prompt = "this is a picture of"
inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)