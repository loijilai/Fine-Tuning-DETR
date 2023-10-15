# Fine-Tuning DETR on Custom Dataset

## Environment

* System Information  
    OS: Ubuntu 18.04  
    CPU: Intel Xeon Silver 4110 (32) @ 1.7GHz  
    GPU: Matrox Electronics Systems Ltd. Integrated Matrox G200eW3 Graphics Controller  
    GPU: NVIDIA Tesla V100 PCIe 16GB  
    Memory: 20343MiB / 385656MiB  
    GPU Driver: NVIDIA 460.91.03  

## How to run my code
First, clone the repository locally:
```
https://github.com/loijilai/Fine-Tuning-DETR.git
```
Then, install PyTorch and torchvision:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
### Training
`cd` into the cloned repository
```
CUDA_VISIBLE_DEVICES=<YOUR_GPU_NUM> \
python ./detr/main.py \
--dataset_file your_dataset \
--coco_path <PATH_TO_DATASET>
--epochs 350 \
--lr=1e-4  \
--batch_size=2 \
--num_workers=4 \
--output_dir=./outputs \
--resume=<PATH_TO_CHECKPOINT>
```
### Inference
To get output.json
```
CUDA_VISIBLE_DEVICES=<YOUR_GPU_NUM> \
python ./detr/infer_json.py \
--data_path <PATH_TO_DATASET> \
--resume <PATH_TO_CHECKPOINT> \
--output_dir <PATH_TO_OUTPUT_DIR>
```
To get visualization result
```
CUDA_VISIBLE_DEVICES=<YOUR_GPU_NUM> \
python ./detr/infer_visualize.py \
--data_path <PATH_TO_DATASET> \
--resume <PATH_TO_CHECKPOINT> \
--output_dir <PATH_TO_OUTPUT_DIR>
```