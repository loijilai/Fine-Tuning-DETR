CUDA_VISIBLE_DEVICES=1 python gligen_inference.py --generation_strategy text1 --input_file /project/dsp/loijilai/cvpdl/hw1_dataset/annotations/for_gligen.json
CUDA_VISIBLE_DEVICES=1 python gligen_inference.py --generation_strategy text2 --input_file /project/dsp/loijilai/cvpdl/hw1_dataset/annotations/for_gligen.json
CUDA_VISIBLE_DEVICES=1 python gligen_inference.py --generation_strategy text_image --input_file /project/dsp/loijilai/cvpdl/hw1_dataset/annotations/for_gligen.json