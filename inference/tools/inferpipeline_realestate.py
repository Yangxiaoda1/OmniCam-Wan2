#!/usr/bin/env python3
import subprocess


for i in range(2, 37,2):
    save_mp4 = f"co3d{i}.mp4"
    dit_file = f"total-{i}.safetensors"
    lora_file = f"lora-{i:06d}.safetensors"

    cmd = [
        "python", "/data/musubi-tuner/wan_generate_video.py",
        "--task", "i2v-14B",
        "--video_size", "936", "504",
        "--video_length", "81",
        "--infer_steps", "20",
        "--prompt",
        ("Object: A triangular piece of cake is placed on a white plate, and the white plate is on a black plate. ; Camera: <starttime>1</starttime> <endtime>2</endtime> <speed>high</speed> <direction>forward</direction> <rotate>stay</rotate> <sep> <starttime>2</starttime> <endtime>3</endtime> <speed>low</speed> <direction>right</direction> <rotate>stay</rotate> <sep> <starttime>3</starttime> <endtime>4</endtime> <speed>high</speed> <direction>leftdown30</direction> <rotate>stay</rotate> <sep>"),
        "--save_path", f"/data/musubi-tuner/inference/lora_text/{save_mp4}",
        "--output_type", "both",
        "--dit", f"/data/musubi-tuner/weights/mycheckpoint/totckpt-RealEstate/{dit_file}",
        "--vae", "/data/musubi-tuner/weights/Wan2.1_VAE.pth",
        "--t5", "/data/musubi-tuner/weights/models_t5_umt5-xxl-enc-bf16.pth",
        "--clip", "/data/musubi-tuner/weights/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "--attn_mode", "torch",
        "--image_path", "/data/musubi-tuner/inference/cake.png"
    ]

    print(f"\n>>> Running {i}/36: save → {save_mp4}")
    subprocess.run(cmd, check=True)