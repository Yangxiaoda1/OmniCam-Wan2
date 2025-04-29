import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

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
                 num_transformer_layers=2  # TransformerBlock 的层数
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

# ---------------------------
# 主流程：加载相机参数、计算 Plücker embedding、构建网络前向传播
# ---------------------------
if __name__ == "__main__":
    # 模拟命令行参数，根据实际情况调整
    class Args:
        safetensors_file = '/data/midjourney/yangxiaoda/RealCam-Vid/realcamcache/raw/1b3f09d0f408cdc6.safetensors'
        image_width = 640
        image_height = 480
    args = Args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading intrinsics and extrinsics from safetensors')
    # 使用 safetensors 加载文件
    from safetensors.torch import load_file
    data = load_file(args.safetensors_file)
    # data 包含两个字段：
    #   'intrinsics': [0.6, 1.1, 0.5, 0.5] (归一化内参: fx, fy, cx, cy)
    #   'extrinsics': shape [1, T, 12]，每行代表一帧的 6DoF 信息
    print(data.keys())
    # 获取内参和外参
    intrinsics = data['intrinsics']    # 形如 [0.6, 1.1, 0.5, 0.5]
    extrinsics = data['extrinsics']    # shape [1, T, 12]
    
    # 如果 intrinsics 为 Tensor，则转换为 numpy 数组
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    else:
        intrinsics = np.array(intrinsics, dtype=np.float32)
    
    # 处理 extrinsics：将形状 [1, T, 12] squeeze 成 [T, 12]
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.squeeze(0).cpu().numpy()
    else:
        extrinsics = np.array(extrinsics, dtype=np.float32)
        extrinsics = np.squeeze(extrinsics, axis=0)
    
    # 获取帧数 T
    T = extrinsics.shape[0]
    
    # 将 extrinsics 的每一行（12 个数）转换为 4x4 的齐次矩阵
    c2ws_list = []
    for i in range(T):
        row = extrinsics[i]  # 形如 [r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3]
        R = row[:9].reshape(3, 3)
        t = row[9:].reshape(3, 1)
        c2w = np.concatenate([np.concatenate([R, t], axis=1),
                              np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
        c2ws_list.append(c2w)
    c2ws = np.stack(c2ws_list, axis=0)  # shape [T, 4, 4]

    # 将归一化内参转换为像素单位：
    # fx, fy 分别乘以 image_width, image_height；cx, cy 同理
    fx = intrinsics[0] * args.image_width
    fy = intrinsics[1] * args.image_height
    cx = intrinsics[2] * args.image_width
    cy = intrinsics[3] * args.image_height
    # 构造每帧的内参，这里假设所有帧使用相同内参，shape 为 [T, 4]
    intrinsic_all = np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (T, 1))
    
    # 构造 K：要求形状为 [B, V, 4]，这里 B=1, V=T
    K = torch.as_tensor(intrinsic_all)[None].to(device)
    
    # 构造 c2ws：要求形状为 [B, V, 4, 4]，这里 B=1, V=T
    c2ws = torch.as_tensor(c2ws)[None].to(device)

    # 构造网络并移至 device
    model = PluckerNet(hidden_channels=64, final_channels=32, num_heads=8, num_transformer_layers=2).to(device)
    # 前向传播，输出形状预期为 [B, final_channels, V, H, W]
    print(f" K.shape:{K.shape}; c2ws.shape:{c2ws.shape}")
    output = model(K, c2ws, args.image_height, args.image_width, device=device)
    print("Output shape:", output.shape)

