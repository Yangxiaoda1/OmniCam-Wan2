# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights

import logging

from utils.safetensors_utils import MemoryEfficientSafeOpen, load_safetensors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils.device_utils import clean_memory_on_device

from .attention import flash_attention
from utils.device_utils import clean_memory_on_device
from modules.custom_offloading_utils import ModelOffloader
from modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8
import numpy as np

__all__ = ["WanModel"]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
# no autocast is needed for rope_apply, because it is already in float64
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    device_type = x.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        n, c = x.size(2), x.size(3) // 2

        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
            freqs_i = torch.cat(
                [
                    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(seq_len, 1, -1)

            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        return torch.stack(output).float()


def calculate_freqs_i(fhw, c, freqs):
    f, h, w = fhw
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_i = torch.cat(
        [
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    return freqs_i


# inplace version of rope_apply
def rope_apply_inplace_cached(x, grid_sizes, freqs_list):
    # with torch.amp.autocast(device_type=device_type, enabled=False):
    rope_dtype = torch.float64  # float32 does not reduce memory usage significantly

    n, c = x.size(2), x.size(3) // 2

    # loop over samples
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(rope_dtype).reshape(seq_len, n, -1, 2))
        freqs_i = freqs_list[i]

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # x_i = torch.cat([x_i, x[i, seq_len:]])

        # inplace update
        x[i, :seq_len] = x_i.to(x.dtype)

    return x




class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # return self._norm(x.float()).type_as(x) * self.weight
        # support fp8
        return self._norm(x.float()).type_as(x) * self.weight.to(x.dtype)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    # def forward(self, x):
    #     r"""
    #     Args:
    #         x(Tensor): Shape [B, L, C]
    #     """
    #     # inplace version, also supports fp8 -> does not have significant performance improvement
    #     original_dtype = x.dtype
    #     x = x.float()
    #     y = x.pow(2).mean(dim=-1, keepdim=True)
    #     y.add_(self.eps)
    #     y.rsqrt_()
    #     x *= y
    #     x = x.to(original_dtype)
    #     x *= self.weight.to(original_dtype)
    #     return x


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attn_mode="torch", split_attn=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # # query, key, value function
        # def qkv_fn(x):
        #     q = self.norm_q(self.q(x)).view(b, s, n, d)
        #     k = self.norm_k(self.k(x)).view(b, s, n, d)
        #     v = self.v(x).view(b, s, n, d)
        #     return q, k, v
        # q, k, v = qkv_fn(x)
        # del x
        # query, key, value function

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        del x
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        v = v.view(b, s, n, d)

        rope_apply_inplace_cached(q, grid_sizes, freqs)
        rope_apply_inplace_cached(k, grid_sizes, freqs)
        qkv = [q, k, v]
        del q, k, v
        x = flash_attention(
            qkv, k_lens=seq_lens, window_size=self.window_size, attn_mode=self.attn_mode, split_attn=self.split_attn
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        # q = self.norm_q(self.q(x)).view(b, -1, n, d)
        # k = self.norm_k(self.k(context)).view(b, -1, n, d)
        # v = self.v(context).view(b, -1, n, d)
        q = self.q(x)
        del x
        k = self.k(context)
        v = self.v(context)
        del context
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        # compute attention
        qkv = [q, k, v]
        del q, k, v
        x = flash_attention(qkv, k_lens=context_lens, attn_mode=self.attn_mode, split_attn=self.split_attn)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x











# ---------------------------
# 辅助函数和占位实现
# ---------------------------
def custom_meshgrid(x, y):
    # 使用 torch.meshgrid，并保持和 numpy.meshgrid 一致的行为（使用 ij 索引）
    j, i = torch.meshgrid(x, y, indexing='ij')
    return j, i

class Camera:
    def __init__(self, params):
        # 假设 params = [fx, fy, cx, cy, ... (其他参数)]
        # 此处仅关注四个内参参数，其他参数可根据需求扩展
        self.fx = params[0]
        self.fy = params[1]
        self.cx = params[2]
        self.cy = params[3]

def get_relative_pose(cam_params):
    # 占位实现：输入为 Camera 对象列表，返回形状 [n_frame, 4, 4] 的相机位姿矩阵
    n_frame = len(cam_params)
    c2ws = []
    for i in range(n_frame):
        c2ws.append(np.eye(4, dtype=np.float32))
    return np.stack(c2ws, axis=0)

# ---------------------------
# ray_condition 函数：根据内参 K 和相机外参 c2w 计算射线并构造 Plücker embedding
# ---------------------------
def ray_condition(K, c2w, H, W, device):
    # K: [B, V, 4]，分别存放 fx, fy, cx, cy
    # c2w: [B, V, 4, 4]
    B = K.shape[0]
    # 生成像素网格（j 对应高度，i 对应宽度）
    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    # 将 (i,j) 坐标调整至像素中心
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, 1, HW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, 1, HW]

    # 拆分内参：fx, fy, cx, cy，形状均为 [B, V, 1]
    fx, fy, cx, cy = K.chunk(4, dim=-1)
    # 假设深度均为 1
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    # 生成归一化方向向量（每个像素的射线方向）
    directions = torch.stack((xs, ys, zs), dim=-1)  # [B, V, HW, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # 将方向向量旋转到世界坐标系中
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # [B, V, HW, 3]
    # 提取光线原点（相机位置）
    rays_o = c2w[..., :3, 3]  # [B, V, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # [B, V, HW, 3]

    # 计算交叉乘积（相机中心与射线方向），作为 Plücker embedding 的一部分
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # [B, V, HW, 3]
    # 拼接形成 6 维表示：[rays_dxo, rays_d]
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)  # [B, V, HW, 6]
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # [B, V, H, W, 6]
    return plucker

# ---------------------------
# TransformerBlock 实现
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        """
        embed_dim: 输入 token 的维度
        num_heads: 多头注意力头数
        mlp_ratio: 前馈网络隐藏层相对维度
        dropout: dropout 概率
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: [B, N, embed_dim]，其中 N = H * W（每一帧展平后的 token 数量）
        """
        # from IPython import embed;embed()
        # 自注意力
        res = x
        x = self.norm1(x)
        # nn.MultiheadAttention 要求输入 shape 为 [seq_len, batch, embed_dim]
        x_trans = x.transpose(0, 1)  # [N, B, embed_dim]
        attn_out, _ = self.attn(x_trans, x_trans, x_trans)
        attn_out = attn_out.transpose(0, 1)  # [B, N, embed_dim]
        x = res + attn_out

        # 前馈网络
        res2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res2 + x
        return x

# ---------------------------
# 网络结构实现：使用 TransformerBlock 替换原 ResBlock
# ---------------------------
class PluckerNet(nn.Module):
    def __init__(self, 
                 hidden_channels=64,   # initial_conv 输出的通道数，同时为 transformer 的 embed_dim
                 final_channels=32,    # 最终输出通道数
                 num_heads=8,          # Transformer 多头注意力头数
                 num_transformer_layers=2  # TransformerBlock 的层数原来是2，现在是1
                ):
        super().__init__()
        # 1. 初始 3D 卷积将 Plücker embedding 从 6 通道升维到 hidden_channels
        # 输入形状：[B, 6, V, H, W]
        self.initial_conv = nn.Conv3d(
            in_channels=6, 
            out_channels=hidden_channels,
            kernel_size=1
        )
        # 2. TransformerBlock 层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim=hidden_channels, num_heads=num_heads)
            for _ in range(num_transformer_layers)
        ])
        # 3. 最终 2D 卷积：将 transformer 提取后的特征从 hidden_channels 转换为 final_channels
        self.final_conv2d = nn.Conv2d(
            in_channels=hidden_channels, 
            out_channels=final_channels,
            kernel_size=3, 
            padding=1
        )
        self.final_bn2d   = nn.BatchNorm2d(final_channels)
        self.relu         = nn.ReLU(inplace=True)
        # from IPython import embed;embed()
    
    def forward(self, K, c2ws, H, W, device):
        
        # K: [B, V, 4]
        # c2ws: [B, V, 4, 4]
        # 1. 计算 Plücker embedding，形状为 [B, V, H, W, 6]
        plucker = ray_condition(K, c2ws, H, W, device=device)  # [B, V, H, W, 6]
        # 调整维度，把通道放到第二维：[B, V, 6, H, W]
        plucker = plucker.permute(0, 1, 4, 2, 3).contiguous()
        # 转换为 3D 卷积输入格式：[B, 6, V, H, W]
        plucker = plucker.permute(0, 2, 1, 3, 4).contiguous()
        # 2. 通过初始 3D 卷积提取特征 -> [B, hidden_channels, V, H, W]
        x = self.initial_conv(plucker)
        B, C, V, H, W = x.shape

        # 3. 针对每一帧（即每个视角）进行 Transformer 处理：
        # 将每一帧的特征展平成序列形式，然后送入 TransformerBlock 进行自注意力计算
        # 形状从 [B, hidden_channels, V, H, W] 调整为 [B*V, H*W, hidden_channels]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, V, hidden_channels, H, W]
        x = x.view(B * V, C, H * W)                  # [B*V, hidden_channels, H*W]
        x = x.transpose(1, 2).contiguous()            # [B*V, H*W, hidden_channels]

        # 依次通过 transformer 层
        for transformer in self.transformer_layers:
            x = transformer(x)  # 维度保持 [B*V, H*W, hidden_channels]

        # 将序列还原为 2D 特征图： [B*V, hidden_channels, H, W]
        x = x.transpose(1, 2).contiguous()            # [B*V, hidden_channels, H*W]
        x = x.view(B * V, C, H, W)
        
        # 4. 通过 2D 卷积得到最终输出 -> [B*V, final_channels, H, W]
        x = self.final_conv2d(x)
        x = self.final_bn2d(x)
        x = self.relu(x)
        # 恢复视角维度： [B, V, final_channels, H, W]
        x = x.view(B, V, self.final_conv2d.out_channels, H, W)
        # 可选地调整维度为 [B, final_channels, V, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x










class CombinedLinear(nn.Module):
    def __init__(self, dim, cam_dim):
        super(CombinedLinear, self).__init__()
        self.linear1 = nn.Linear(dim, cam_dim)  # 第一个线性层，将 dim 映射到 cam_dim
        self.linear2 = nn.Linear(cam_dim, dim)  # 第二个线性层，将 cam_dim 映射回 dim

    def forward(self, x):
        x = self.linear1(x)  # 先通过第一个线性层
        x = self.linear2(x)  # 再通过第二个线性层
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attn_mode="torch", split_attn=False,max_camera_len=150, cam_middim=640):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, attn_mode, split_attn)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.q1 = nn.Linear(dim, dim)
        # self.norm_q1 = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # self.q2 = nn.Linear(dim, dim)
        # self.norm_q2 = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.k_camera1 = CombinedLinear(dim, cam_middim)
        self.v_camera1 = CombinedLinear(dim, cam_middim)
        self.norm_k_cam1 = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # self.k_camera2 = CombinedLinear(dim, cam_middim)
        # self.v_camera2 = CombinedLinear(dim, cam_middim)
        # self.norm_k_cam2 = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
        
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()


    def forward(self, x, context, context_lens, adapteroutput, grid_sizes):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]; [C_in, F, H, W], F就是T, C_in就是C, 这里省略了Bs
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        # print(f"x.shape.latent:{x.shape}")
        H,W=grid_sizes[0][1],grid_sizes[0][2]
        T=grid_sizes[0][0]
        B=x.shape[0]
        C=x.shape[2]
        # print(f"B:{B}; T:{T}; H:{H}; W:{W}; C:{C}")
        camx=x.reshape(B,T,H,W,C)
        x1=camx.reshape(B,T*H*W,C)
        x2=camx.reshape(B*T,H*W,C)
        context_img = context[:, :257]
        context = context[:, 257:]
        # from IPython import embed;embed()
        
        # print('cameradict:',cameradict)
        # intrinsicstensor= cameradict['intrinsics'].to(x.device)
        # extrinsicstensor= cameradict['extrinsics'].to(x.device)

        

        # slidenum=(T-1)*4+1 #9
        # totframe=extrinsicstensor.shape[1] #114
        # indices = torch.linspace(0, totframe-1, steps=slidenum).long()#1，18，...
        # # print(f"indices:{indices}")
        # extrinsicstensor = extrinsicstensor[:, indices, :, :]#[:,9,:,:]
        # iter_=1+(slidenum-1)//4
        # # from IPython import embed;embed()
        # for i in range(iter_):
        #     if i==0:
        #         finalextrinsics=extrinsicstensor[:,:1,:,:]
        #     else:
        #         # from IPython import embed;embed()
        #         finalextrinsics_=self.conv(extrinsicstensor[:,1+4*(i-1):1+4*i,:,:])
        #         finalextrinsics=torch.cat([finalextrinsics,finalextrinsics_],1)

        # print(f"intrinsicstensor.shape:{intrinsicstensor.shape}; extrinsicstensor.shape:{finalextrinsics.shape}")
        
        # from IPython import embed;embed()
        # fx = (intrinsicstensor[0][0] * W).cpu().item()
        # fy = (intrinsicstensor[0][1] * H).cpu().item()
        # cx = (intrinsicstensor[0][2] * W).cpu().item()
        # cy = (intrinsicstensor[0][3] * H).cpu().item()
        # intrinsic_all = np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (T, 1))
        # K = torch.as_tensor(intrinsic_all).unsqueeze(0).to(x.device)
        # finalextrinsics = torch.as_tensor(finalextrinsics).to(x.device)

        
        
        
        # from IPython import embed;embed()
        

        # #x-cam1
        # cam1=adapteroutput.reshape(B,T,H*W*camC) #[B,T,H*W*C]
        # q=self.q()




        
        
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x)
        del x
        q = self.norm_q(q)
        q = q.view(b, -1, n, d)
        k = self.k(context)
        k = self.norm_k(k).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        del context

        # compute attention
        qkv = [q, k, v]
        del k, v
        x = flash_attention(qkv, k_lens=context_lens, attn_mode=self.attn_mode, split_attn=self.split_attn)

        # compute query, key, value
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        del context_img

        # compute attention
        qkv = [q, k_img, v_img]
        del q, k_img, v_img
        img_x = flash_attention(qkv, k_lens=None, attn_mode=self.attn_mode, split_attn=self.split_attn)


        #x-cam1
        qx1 = self.norm_q(self.q(x1)).view(b, -1, n, d)
        cam1=adapteroutput.reshape(B,T*H*W,C) #[B,T*H*W,C]
        k_camera1 = self.norm_k_cam1(self.k_camera1(cam1)).view(b, -1, n, d)
        v_camera1 = self.v_camera1(cam1).view(b, -1, n, d)
        qkv_camera1 = [qx1, k_camera1, v_camera1]
        cam_x1 = flash_attention(qkv_camera1, k_lens=None, attn_mode=self.attn_mode, split_attn=self.split_attn)

        
        # #x-cam2
        qx2 = self.norm_q(self.q(x2)).view(b, -1, n, d)
        cam2=adapteroutput.reshape(B*T,H*W,C) #[B*T,H*W,C]
        k_camera2 = self.norm_k_cam1(self.k_camera1(cam2)).view(b, -1, n, d)
        v_camera2 = self.v_camera1(cam2).view(b, -1, n, d)
        qkv_camera2 = [qx2, k_camera2, v_camera2]
        cam_x2 = flash_attention(qkv_camera2, k_lens=None, attn_mode=self.attn_mode, split_attn=self.split_attn)

        # from IPython import embed;embed()


        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        cam_x1 = cam_x1.flatten(2)
        cam_x2 = cam_x2.flatten(2)
        # from IPython import embed;embed()
        if self.training:
            x = x + img_x + cam_x1 + cam_x2  # avoid inplace  
        else:
            x += img_x
        del img_x

        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode="torch",
        split_attn=False,
        max_camera_len=150,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, attn_mode, split_attn)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps, attn_mode, split_attn,max_camera_len)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.gradient_checkpointing = False

        

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, adapteroutput):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        #     e = (self.modulation + e).chunk(6, dim=1)
        # support fp8
        e = self.modulation.to(torch.float32) + e
        e = e.chunk(6, dim=1)
        assert e[0].dtype == torch.float32



        # from IPython import embed;embed()
        # self-attention
        y = self.self_attn(self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)#[B,L,C]*[B,1,C]
        # with amp.autocast(dtype=torch.float32):
        #     x = x + y * e[2]
        x = x + y.to(torch.float32) * e[2]
        del y

        # cross-attention & ffn function
        # def cross_attn_ffn(x, context, context_lens, e):
        #     x += self.cross_attn(self.norm3(x), context, context_lens)
        #     y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        #     # with amp.autocast(dtype=torch.float32):
        #     #     x = x + y * e[5]
        #     x += y.to(torch.float32) * e[5]
        #     return x
        # x = cross_attn_ffn(x, context, context_lens, e)

        # x += self.cross_attn(self.norm3(x), context, context_lens) # backward error
        x = x + self.cross_attn(self.norm3(x), context, context_lens, adapteroutput, grid_sizes)
        del context
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        x = x + y.to(torch.float32) * e[5]
        del y
        return x

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, adapteroutput):
        if self.training and self.gradient_checkpointing:
            # from IPython import embed;embed()
            return checkpoint(self._forward, x, e, seq_lens, grid_sizes, freqs, context, context_lens,adapteroutput, use_reentrant=False)
        # print('白色')
        return self._forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens, adapteroutput)


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        #     e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        #     x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        # support fp8
        e = (self.modulation.to(torch.float32) + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x

# WanModel
class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(nn.Module):  # ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    # @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        max_camera_len=150,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        attn_mode=None,
        split_attn=False,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.max_camera_len=max_camera_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attn_mode = attn_mode if attn_mode is not None else "torch"
        self.split_attn = split_attn

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, attn_mode, split_attn,max_camera_len
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1
        )
        self.freqs_fhw = {}

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        # offloading
        self.blocks_to_swap = None
        self.offloader = None

        self.conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1).to(torch.float)
        self.myadapter = PluckerNet(hidden_channels=64, final_channels=dim, num_heads=8, num_transformer_layers=2)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def fp8_optimization(
        self, state_dict: dict[str, torch.Tensor], device: torch.device, move_to_device: bool, use_scaled_mm: bool = False
    ) -> int:
        """
        Optimize the model state_dict with fp8.

        Args:
            state_dict (dict[str, torch.Tensor]):
                The state_dict of the model.
            device (torch.device):
                The device to calculate the weight.
            move_to_device (bool):
                Whether to move the weight to the device after optimization.
        """
        TARGET_KEYS = ["blocks"]
        EXCLUDE_KEYS = [
            "norm",
            "patch_embedding",
            "text_embedding",
            "time_embedding",
            "time_projection",
            "head",
            "modulation",
            "img_emb",
        ]

        # inplace optimization
        state_dict = optimize_state_dict_with_fp8(state_dict, device, TARGET_KEYS, EXCLUDE_KEYS, move_to_device=move_to_device)

        # apply monkey patching
        apply_fp8_monkey_patch(self, state_dict, use_scaled_mm=use_scaled_mm)

        return state_dict

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

        for block in self.blocks:
            block.enable_gradient_checkpointing()

        print(f"WanModel: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

        for block in self.blocks:
            block.disable_gradient_checkpointing()

        print(f"WanModel: Gradient checkpointing disabled.")

    def enable_block_swap(self, blocks_to_swap: int, device: torch.device, supports_backward: bool):
        self.blocks_to_swap = blocks_to_swap
        self.num_blocks = len(self.blocks)

        assert (
            self.blocks_to_swap <= self.num_blocks - 1
        ), f"Cannot swap more than {self.num_blocks - 1} blocks. Requested {self.blocks_to_swap} blocks to swap."

        self.offloader = ModelOffloader(
            "wan_attn_block", self.blocks, self.num_blocks, self.blocks_to_swap, supports_backward, device  # , debug=True
        )
        print(
            f"WanModel: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {self.num_blocks} blocks. Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print(f"WanModel: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print(f"WanModel: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None,cameradict=None, skip_block_indices=None):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W], F就是T, C_in就是C, 这里省略了Bs
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # print(f"x.shape:{x.shape}")
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # print(f"x.shape:{x.shape}")
        # print(f"y.shape:{y.shape}")

        # from IPython import embed;embed()
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            y = None
        

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]# 对输入 x 的每个元素进行 patch embedding
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])# 获取每个经过 patch embedding 后张量的空间尺寸

        freqs_list = []# 初始化一个列表，用于存放每个 grid 尺寸对应的频率信息
        for fhw in grid_sizes:
            fhw = tuple(fhw.tolist())# 将 grid 尺寸转换为元组形式，便于作为字典的键
            if fhw not in self.freqs_fhw: # 如果当前 grid 尺寸未在字典 self.freqs_fhw 中，则计算并存储对应的频率信息
                c = self.dim // self.num_heads // 2# 计算每个头部一半的维度数，作为频率计算的参数
                self.freqs_fhw[fhw] = calculate_freqs_i(fhw, c, self.freqs)
            freqs_list.append(self.freqs_fhw[fhw]) # 将该 grid 尺寸对应的频率信息加入列表中

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, f"Sequence length exceeds maximum allowed length {seq_len}. Got {seq_lens.max()}"#确保序列的最大长度不超过预设的 seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])# 对每个张量 u，在其后拼接一个全零张量，使得每个张量的序列长度达到 seq_len

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float32):#时间嵌入
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float()) #t是加噪的时间步，这一步相当于把t注入模型
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))# 进一步将嵌入通过 time_projection 层处理，并将第二个维度重塑为 (6, self.dim) 的形状
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # intrinsicslist=cameradict['intrinsics']
        # extrinsicslist=cameradict['extrinsics']
        # stacked_extrinsics = torch.stack([torch.cat([u, u.new_zeros(self.max_camera_len - u.size(0), u.size(1), u.size(2))], dim=0) for u in extrinsicslist])
        # stacked_intrinsics = torch.stack([u for u in intrinsicslist])
        # cameradict={
        #     "intrinsics":stacked_intrinsics,
        #     "extrinsics":stacked_extrinsics
        # }


        # context
        context_lens = None
        if type(context) is list:
            context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        # from IPython import embed;embed()
        context = self.text_embedding(context)# 将 context 通过 text_embedding 层进行嵌入

        

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)# 将图像嵌入和文本嵌入在序列维度上进行拼接
            clip_fea = None
            context_clip = None
        # print('context.shape:',context.shape)
        # arguments， 将所有计算得到的参数整理为一个字典
        # from IPython import embed;embed()









        H,W=grid_sizes[0][1],grid_sizes[0][2]
        T=grid_sizes[0][0]

        intrinsicslist=cameradict['intrinsics']
        extrinsicslist=cameradict['extrinsics']
        slidenum=(T-1)*4+1
        vec=[]
        for exitem in extrinsicslist:
            totframe=exitem.shape[0]
            if slidenum>totframe:
                #indecs=[0,1,2,....,totframe,totframe,...,totframe]直到slidenum
                indices = torch.cat([torch.arange(totframe),torch.full((slidenum - totframe,), totframe - 1)])
            else:
                indices = torch.linspace(0, totframe-1, steps=slidenum).long()
            # from IPython import embed;embed()
            extrinsicstensor = exitem[indices, :, :].unsqueeze(0)
            iter_=1+(slidenum-1)//4
            for i in range(iter_):
                if i==0:
                    finalextrinsics=extrinsicstensor[:,:1,:,:]
                else:
                    # from IPython import embed;embed()
                    finalextrinsics_=self.conv(extrinsicstensor[:,1+4*(i-1):1+4*i,:,:])
                    finalextrinsics=torch.cat([finalextrinsics,finalextrinsics_],1)
            vec.append(finalextrinsics)
        # from IPython import embed;embed()
        finalextrinsics=torch.cat(vec,dim=0)

        vecin=[]
        for initem in intrinsicslist:
            fx = (initem[0] * W).cpu().item()
            fy = (initem[1] * H).cpu().item()
            cx = (initem[2] * W).cpu().item()
            cy = (initem[3] * H).cpu().item()
            intrinsic_all = np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (T, 1))
            K = torch.as_tensor(intrinsic_all).unsqueeze(0).to(x.device)
            vecin.append(K)
        finalK=torch.cat(vecin,dim=0)

        adapteroutput = self.myadapter(finalK, finalextrinsics, H, W, device=x.device)#[B, C, T, H, W]
        adapteroutput = adapteroutput.permute(0, 2, 3, 4, 1).contiguous()# [B, T, H, W, C]
















        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs_list, context=context, context_lens=context_lens,adapteroutput=adapteroutput)

        # from IPython import embed;embed()
        if self.blocks_to_swap:
            clean_memory_on_device(device)

        # print(f"x: {x.shape}, e: {e0.shape}, context: {context.shape}, seq_lens: {seq_lens}")
        for block_idx, block in enumerate(self.blocks):
            is_block_skipped = skip_block_indices is not None and block_idx in skip_block_indices

            if self.blocks_to_swap and not is_block_skipped:
                self.offloader.wait_for_block(block_idx)

            if not is_block_skipped:
                # print('x.shape',x.shape)
                x = block(x, **kwargs)

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, block_idx)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


def detect_wan_sd_dtype(path: str) -> torch.dtype:
    # get dtype from model weights
    with MemoryEfficientSafeOpen(path) as f:
        # from IPython import embed;embed()
        keys = set(f.keys())
        key1 = "model.diffusion_model.blocks.0.cross_attn.k.weight"  # 1.3B
        key2 = "blocks.0.cross_attn.k.weight"  # 14B
        if key1 in keys:
            dit_dtype = f.get_tensor(key1).dtype
        elif key2 in keys:
            dit_dtype = f.get_tensor(key2).dtype
        else:
            raise ValueError(f"Could not find the dtype in the model weights: {path}")
    logger.info(f"Detected DiT dtype: {dit_dtype}")
    return dit_dtype



def init_weights(m):
    # 针对卷积层（Conv2d, Conv3d）使用 Kaiming Uniform 初始化
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        # nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)
            # nn.init.constant_(m.bias, 0)
    # 针对全连接层（Linear）使用 Kaiming Uniform 初始化
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        # nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)
            # nn.init.constant_(m.bias, 0)
    # 针对 BatchNorm2d 层，一般将 weight 初始化为 1，bias 初始化为 0
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    # 针对 LayerNorm 层，一般将 weight 初始化为 1，bias 初始化为 0
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    # 针对 WanRMSNorm 层也可以做类似的初始化
    elif isinstance(m, WanRMSNorm):
        nn.init.constant_(m.weight, 1)
    # 针对 nn.MultiheadAttention 内部
    elif isinstance(m, nn.MultiheadAttention):
        m._reset_parameters()
    # else:
    #     print(f"未初始化的 {m.__class__.__name__}")




def load_wan_model(
    config: any,
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
) -> WanModel:
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    with init_empty_weights():
        logger.info(f"Creating WanModel")
        model = WanModel(
            model_type="i2v" if config.i2v else "t2v",
            dim=config.dim,
            eps=config.eps,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            in_dim=config.in_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            out_dim=config.out_dim,
            text_len=config.text_len,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # if fp8_scaled, load model weights to CPU to reduce VRAM usage. Otherwise, load to the specified device (CPU for block swap or CUDA for others)
    wan_loading_device = torch.device("cpu") if fp8_scaled else loading_device
    logger.info(f"Loading DiT model from {dit_path}, device={wan_loading_device}, dtype={dit_weight_dtype}")

    # load model weights with the specified dtype or as is
    sd = load_safetensors(dit_path, wan_loading_device, disable_mmap=True, dtype=dit_weight_dtype)

    # remove "model.diffusion_model." prefix: 1.3B model has this prefix
    for key in list(sd.keys()):
        if key.startswith("model.diffusion_model."):
            sd[key[22:]] = sd.pop(key)

    if fp8_scaled:
        # fp8 optimization: calculate on CUDA, move back to CPU if loading_device is CPU (block swap)
        logger.info(f"Optimizing model weights to fp8. This may take a while.")
        sd = model.fp8_optimization(sd, device, move_to_device=loading_device.type == "cpu")

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)
    # from IPython import embed;embed()
    model=model.to_empty(device=device)
    
    # from IPython import embed;embed()
    # myadapter.apply(init_weights)
    # init_weights(myadapter)
        # myadapter.apply(init_weights)
        # init_weights(myadapter)

    
    model.myadapter.apply(init_weights)
    model.conv.apply(init_weights)
    for block in model.blocks:
        # block.cross_attn.conv.apply(init_weights)
        # block.cross_attn.q1.apply(init_weights)
        # block.cross_attn.norm_q1.apply(init_weights)
        block.cross_attn.k_camera1.apply(init_weights)
        block.cross_attn.v_camera1.apply(init_weights)
        block.cross_attn.norm_k_cam1.apply(init_weights)

        # block.cross_attn.q2.apply(init_weights)
        # block.cross_attn.norm_q2.apply(init_weights)
        # block.cross_attn.k_camera2.apply(init_weights)
        # block.cross_attn.v_camera2.apply(init_weights)
        # block.cross_attn.norm_k_cam2.apply(init_weights)
    info = model.load_state_dict(sd, strict=False, assign=True)
    out_path = "/data/musubi-tuner/load_state_report.txt"
    with open(out_path, "w") as f:
        f.write("=== Missing Keys ===\n")
        for k in info.missing_keys:
            f.write(k + "\n")
        f.write("\n=== Unexpected Keys ===\n")
        for k in info.unexpected_keys:
            f.write(k + "\n")


    logger.info(f"Loaded DiT model from {dit_path}, info={info}")

    return model
