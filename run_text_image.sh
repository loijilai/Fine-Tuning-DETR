CUDA_VISIBLE_DEVICES=2 python ./detr/main.py \
--dataset_file your_dataset \
--coco_path /project/dsp/loijilai/cvpdl/hw1_dataset_text_image \
--epochs 300 \
--lr 1e-4  \
--batch_size 8 \
--num_workers 4 \
--output_dir ./outputs/text_image \
--resume /tmp2/loijilai/cvpdl/hw3/Fine-Tuning-DETR/detr-r50_no-class-head.pth