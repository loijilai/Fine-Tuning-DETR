git clone git@github.com:gligen/GLIGEN.git
mkdir gligen_checkpoints
cd gligen_checkpoints
wget -O checkpoint_generation_text.pth https://huggingface.co/gligen/gligen-generation-text-box/resolve/main/diffusion_pytorch_model.bin
wget -O checkpoint_generation_text_image.pth https://huggingface.co/gligen/gligen-generation-text-image-box/resolve/main/diffusion_pytorch_model.bin