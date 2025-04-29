import os
os.sys.path.append('./Dust3rP')

import cv2
from PIL import Image
import torch
import numpy as np

# 导入 dust3r 相关模块
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    将 PIL.Image 转换为 torch.Tensor，通道顺序为 (C, H, W)，
    并将像素值归一化到 [-1, 1]。
    """
    arr = np.array(img).astype(np.float32) / 255.0  # 转换为 [0, 1]
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    arr = np.transpose(arr, (2, 0, 1))  # 转为 channel-first
    arr = arr * 2 - 1  # 归一化到 [-1, 1]
    tensor = torch.from_numpy(arr)
    return tensor

def get_frames_from_video(video_path: str, frame_indices=[0, 1], size=512) -> list:
    """
    从视频中提取指定帧，转换为 tensor 后包装成字典格式，
    字典包含以下字段：
      - 'img': 经过 resize 后的图像 tensor，形状为 (1, C, H, W)
      - 'true_shape': 设为 resize 后的尺寸，格式为 numpy 数组，例如 array([[size, size]], dtype=int32)
      - 'idx': 当前帧的采集索引
      - 'instance': 当前帧的实例标识（字符串形式）
    参数:
      video_path: 视频文件路径。
      frame_indices: 需要提取的帧索引列表（从 0 开始）。
      size: 调整后的图像尺寸（bucket 分辨率），统一 resize 到 size x size。
    返回:
      一个列表，每个元素为上述字典。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frames = []
    current_idx = 0
    while cap.isOpened() and len(frames) < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in frame_indices:
            # 转换 BGR 到 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            # 调整尺寸到指定大小（例如 size x size）
            resized_img = img.resize((size, size))
            # 将 resized PIL.Image 转换为 tensor，并增加 batch 维度
            img_tensor = pil_to_tensor(resized_img).unsqueeze(0)
            # 使用 resize 后的尺寸作为 true_shape
            true_shape = np.array([[size, size]], dtype=np.int32)
            frame_dict = {
                'img': img_tensor,
                'true_shape': true_shape,
                'idx': current_idx,
                'instance': str(current_idx)
            }
            frames.append(frame_dict)
        current_idx += 1
    cap.release()
    
    if len(frames) < len(frame_indices):
        raise ValueError("视频帧不足，无法提取指定的帧")
    return frames

def get_intrinsics_from_video(video_path: str) -> list:
    """
    输入视频路径，通过提取第一帧和第五帧构造图像对，
    调用 dust3r 推理和全局对齐优化，返回内参 [fx, fy, px, py]，
    其中 (px, py) 假定为图像中心。
    """
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "/data/musubi-tuner/weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    
    # 提取指定帧，返回的 'img' 为 tensor 且 'true_shape' 固定为 (size, size)
    frames = get_frames_from_video(video_path, frame_indices=[0, 1], size=512)
    
    # 构造图像对，scene_graph 参数设为 'complete'
    pairs = make_pairs(frames, scene_graph='complete', prefilter=None, symmetrize=True)
    
    # 调用推理函数
    output = inference(pairs, model, device, batch_size=batch_size)
    
    # 全局对齐优化
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    
    # 从 scene 中提取焦距参数
    focals = scene.get_focals()  # 期望返回 tensor，例如 tensor([[fx], [fy]])
    imgs = scene.imgs  # 获取图像列表

    # from IPython import embed;embed()
    height,width=imgs[0].shape[:2]
    
    # 计算主点（图像中心）
    px = width / 2
    py = height / 2

    # 提取焦距
    if isinstance(focals, torch.Tensor):
        focal_values = focals.squeeze().tolist()
    else:
        focal_values = focals
    fx, fy = focal_values[0], focal_values[1]
    fx=fx/width
    fy=fy/height
    px=px/width
    py=py/height
    
    return [fx, fy, px, py]





def get_intrinsics_from_image(image_path: str) -> list:
    """
    输入图像路径，通过图像推断焦距和主点，返回内参 [fx, fy, px, py]。
    假设主点 (px, py) 是图像中心，焦距通过预训练模型估算。
    参数:
      image_path: 输入图像路径。
    返回:
      包含 [fx, fy, px, py] 的列表，归一化到 [0, 1] 范围。
    """
    batch_size = 1
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    
    # 使用预训练模型（例如 DUSt3R）来估算焦距，这里暂时假设该模型处理单张图像
    device = 'cuda'
    model_name = "/data/musubi-tuner/weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    images = load_images([image_path, image_path], size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    
    # 假设模型输出焦距，处理过程与视频推理相同
    focals = scene.get_focals()  # 期望返回 tensor，例如 tensor([[fx], [fy]])
    # 获取图像尺寸
    height, width = images[0]['img'].shape[2],images[0]['img'].shape[3]
    
    # 计算主点（图像中心）
    px = width / 2
    py = height / 2
    
    # 提取焦距并归一化
    if isinstance(focals, torch.Tensor):
        focal_values = focals.squeeze().tolist()
    else:
        focal_values = focals
    
    fx, fy = focal_values[0], focal_values[1]
    fx = fx / width  # 归一化到 [0, 1]
    fy = fy / height  # 归一化到 [0, 1]
    px = px / width  # 归一化到 [0, 1]
    py = py / height  # 归一化到 [0, 1]
    
    return [fx, fy, px, py]







if __name__ == '__main__':
    image_path = '/data/musubi-tuner/inference/cake.png'
    intrinsics = get_intrinsics_from_image(image_path)
    print('intrinsics: [fx, fy, px, py] =', intrinsics)
