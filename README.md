# OmniCam-Wan

## Clone

```bash
git clone https://github.com/Yangxiaoda1/OmniCam-Wan.git
cd OmniCam-Wan
git clone --recursive https://github.com/naver/dust3r
```
Rename the dust3r to Dust3rP.
## Package

Python 3.10 or later is required (verified with 3.10).

Create a virtual environment and install PyTorch and torchvision matching your CUDA version. 

PyTorch 2.5.1 or later is required (see [note](#PyTorch-version)).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

Optionally, you can use FlashAttention and SageAttention (for inference only; see [SageAttention Installation](#sageattention-installation) for installation instructions).

Optional dependencies for additional features:
- `ascii-magic`: Used for dataset verification
- `matplotlib`: Used for timestep visualization
- `tensorboard`: Used for logging training progress

```bash
pip install ascii-magic matplotlib tensorboard
```

## Download the model

Download the T5 `models_t5_umt5-xxl-enc-bf16.pth` and CLIP `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` and the VAE from the above page `Wan2.1_VAE.pth`, from the following page: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main

Download the DiT weights from the following page: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models

Download Dust3R weights from the following page: https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

[Optional] Wan2.1 Fun Control model weights can be downloaded from [here](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control). Navigate to each weight page and download. The Fun Control model seems to support not only T2V but also I2V tasks. `fp16` and `bf16` models can be used, and `fp8_e4m3fn` models can be used if `--fp8` (or `--fp8_base`) is specified without specifying `--fp8_scaled`. **Please note that `fp8_scaled` models are not supported even with `--fp8_scaled`.**

### Structure

    |-weights
        |-mycheckpoint
            |-pretrain
                |-3.safetensors
        |-diffusionmodel
            |-wan2.1_i2v_480p_14B_bf16.safetensors
        |-DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
        |-models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
        |-models_t5_umt5-xxl-enc-bf16.pth
        |-Wan2.1_VAE.pth

### Data Preparation
Download dataset from:

https://huggingface.co/datasets/yangxiaoda/OmniCam-Sep-1-OpenSoraPlan; https://huggingface.co/datasets/yangxiaoda/OmniCam-Sep-2-Simple; https://huggingface.co/datasets/yangxiaoda/OmniCam-Sep-3-ImageNet; https://huggingface.co/datasets/yangxiaoda/OmniCam-Sep-3-CO3D; https://huggingface.co/datasets/yangxiaoda/OmniCam-Uni-4-RealCam;

Follow the README.md in tools to finish the data preparation. The final structure will be like, take OmniCam-Sep-1-OpenSoraPlan as example:

    |-OmniCam-Sep-1-OpenSoraPlan
        |-cache
        |-object_stay_camera_move
        |-tools
        metadata.jsonl

modify the toml file in the dataset-toml

### Data Process
CLIP:
```bash
python wan_cache_latents.py \
--dataset_config /data/musubi-tuner/dataset-toml/OmniCam-Sep3.toml \
--vae  /data/musubi-tuner/weights/Wan2.1_VAE.pth \
--clip /data/musubi-tuner/weights/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
--keep_cache
```
T5:
```bash
python wan_cache_text_encoder_outputs.py \
--dataset_config /data/musubi-tuner/dataset-toml/OmniCam-Sep2.toml  \
--t5 /data/musubi-tuner/weights/models_t5_umt5-xxl-enc-bf16.pth \
--batch_size 16
```
Camera:
```bash
python wan_cache_camera_encoder_outputs.py \
--dataset_config /data/musubi-tuner/dataset-toml/OmniCam-Sep2.toml \
--t5 /data/musubi-tuner/weights/models_t5_umt5-xxl-enc-bf16.pth \
--batch_size 16
```

### Train

```bash
accelerate launch --config_file /data/default_config.yaml --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py \
    --task i2v-14B \
    --dit /data/musubi-tuner/weights/mycheckpoint/totckpt-OmniCam-Sep-1-3/3.safetensors \
    --dataset_config /data/musubi-tuner/dataset-toml/OmniCam-Sep.toml \
    --sdpa --mixed_precision bf16  \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 64 \
    --timestep_sampling shift --discrete_flow_shift 3.0 \
    --max_train_epochs 10 --save_every_n_epochs 1 --seed 42 \
    --output_dir /data/musubi-tuner/weights/mycheckpoint/totckpt-OmniCam-Sep-4- --output_name lora
```

### Inference
```bash
python wan_generate_video.py --task i2v-14B --video_size 936 504 --video_length 81 --infer_steps 20 \
--prompt "Object: A triangular piece of cake is placed on a white plate, and the white plate is on a black plate. ; Camera: <starttime>1</starttime> <endtime>2</endtime> <speed>high</speed> <direction>forward</direction> <rotate>stay</rotate> <sep> <starttime>2</starttime> <endtime>3</endtime> <speed>low</speed> <direction>right</direction> <rotate>stay</rotate> <sep> <starttime>3</starttime> <endtime>4</endtime> <speed>high</speed> <direction>leftdown30</direction> <rotate>stay</rotate> <sep>" --save_path /data/musubi-tuner/inference/lora_text/co3d36.mp4 --output_type both \
--dit /data/musubi-tuner/weights/mycheckpoint/totckpt-OmniCam-Sep-1-3/3.safetensors --vae /data/musubi-tuner/weights/Wan2.1_VAE.pth \
--t5 /data/musubi-tuner/weights/models_t5_umt5-xxl-enc-bf16.pth --clip /data/musubi-tuner/weights/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
--attn_mode torch --image_path /data/musubi-tuner/inference/cake.png
```

