import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
import accelerate

from dataset.image_video_dataset import ARCHITECTURE_WAN, ItemInfo, save_text_encoder_output_cache_wan, save_camera_encoder_output_cache_wan

# for t5 config: all Wan2.1 models have the same config for t5
from wan.configs import wan_t2v_14B

import cache_text_encoder_outputs
import cache_camera_encoder_outputs
import logging

from utils.model_utils import str_to_dtype
from wan.modules.t5 import T5EncoderModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)







import re
from safetensors.torch import load_file



import numpy as np
import copy
import os
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from tqdm import tqdm
import matplotlib.colors as mcolors
from PIL import Image
import torchvision
from scipy.interpolate import UnivariateSpline
import torch
from myintrinsics_dust3r import get_intrinsics_from_video
def sphere2pose(c2ws_input, theta, phi, r):
    c2ws = copy.deepcopy(c2ws_input)
    # 先沿着世界坐标系z轴方向平移再旋转
    c2ws[:,2,3] += r

    theta = torch.deg2rad(torch.tensor(theta))
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = torch.tensor([
        [1, 0, 0, 0],
        [0, cos_value_x, -sin_value_x, 0],
        [0, sin_value_x, cos_value_x, 0],
        [0, 0, 0, 1]
    ]).unsqueeze(0).repeat(c2ws.shape[0], 1, 1)

    phi = torch.deg2rad(torch.tensor(phi))
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = torch.tensor([
        [cos_value_y, 0, sin_value_y, 0],
        [0, 1, 0, 0],
        [-sin_value_y, 0, cos_value_y, 0],
        [0, 0, 0, 1]
    ]).unsqueeze(0).repeat(c2ws.shape[0], 1, 1)

    c2ws = np.matmul(rot_mat_x, c2ws)
    c2ws = np.matmul(rot_mat_y, c2ws)

    return c2ws 

def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew

def generate_traj_txt(viewcrafter_lines):
    phi = np.array(list(map(float, viewcrafter_lines[0].split())), dtype=np.float32)[:25]
    theta = np.array(list(map(float, viewcrafter_lines[1].split())), dtype=np.float32)[:25]
    r = np.array(list(map(float, viewcrafter_lines[2].split())), dtype=np.float32)[:25]
    c2ws_anc = np.array(
        [[
            [ 1.0000e+00, -8.6427e-07,  2.3283e-09,  1.4707e-09],
            [ 8.6069e-07, -9.9609e-01, -8.7158e-02,  8.7708e-02],
            [-7.7647e-08,  8.7159e-02, -9.9609e-01,  1.0029e+00],
            [-2.6349e-15,  3.0323e-09, -6.1118e-09,  1.0000e+00]
        ]],
        dtype=np.float32,
    )

    frame = len(phi)
    print('frame:', frame)
    if len(phi) > 3:
        phis = txt_interpolation(phi, frame, mode='smooth')
        phis[0] = phi[0]
        phis[-1] = phi[-1]
    else:
        phis = txt_interpolation(phi, frame, mode='smooth')

    if len(theta) > 3:
        thetas = txt_interpolation(theta, frame, mode='smooth')
        thetas[0] = theta[0]
        thetas[-1] = theta[-1]
    else:
        thetas = txt_interpolation(theta, frame, mode='smooth')

    if len(r) > 3:
        rs = txt_interpolation(r, frame, mode='smooth')
        rs[0] = r[0]
        rs[-1] = r[-1]        
    else:
        rs = txt_interpolation(r, frame, mode='smooth')
    rs = rs * c2ws_anc[0, 2, 3]

    c2ws_list = []
    for th, ph, r_val in zip(thetas, phis, rs):
        c2w_new = sphere2pose(c2ws_anc, np.float32(th), np.float32(ph), np.float32(r_val))
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)

    R, T = c2ws[:, :3, :3], c2ws[:, :3, 3:]
    # 将dust3r坐标系转成pytorch3d坐标系
    R = np.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], 2)
    new_c2w = np.concatenate([R, T], axis=2)
    w2c = np.linalg.inv(
        np.concatenate(
            (new_c2w, np.broadcast_to(np.array([[[0, 0, 0, 1]]]), (new_c2w.shape[0], 1, 4))),
            axis=1,
        )
    )

    return w2c

def parse_tagged_text_to_json(tagged_text):
    """
    将模型输出的标签化文本解析为JSON格式。
    假设标签化文本结构类似：
    <starttime>0</starttime><endtime>1</endtime><speed>high</speed><direction>down</direction><rotate>stay</rotate><sep>...
    """
    segments = tagged_text.split("<sep>")
    results = []
    for seg in segments:
        starttime_match = re.search(r"<starttime>(.*?)</starttime>", seg)
        endtime_match = re.search(r"<endtime>(.*?)</endtime>", seg)
        speed_match = re.search(r"<speed>(.*?)</speed>", seg)
        direction_match = re.search(r"<direction>(.*?)</direction>", seg)
        rotate_match = re.search(r"<rotate>(.*?)</rotate>", seg)
        if starttime_match and endtime_match and speed_match and direction_match and rotate_match:
            starttime = starttime_match.group(1)
            endtime = endtime_match.group(1)
            speed = speed_match.group(1)
            direction = direction_match.group(1)
            rotate = rotate_match.group(1)
            try:
                starttime = float(starttime)
            except ValueError:
                pass
            try:
                endtime = float(endtime)
            except ValueError:
                pass
            results.append({
                "starttime": starttime,
                "endtime": endtime,
                "speed": speed,
                "direction": direction,
                "rotate": rotate
            })
    final_json = {"1": results}
    return final_json

class Text2VideoSet:
    def __init__(self, json_data, fps=8):
        self.JsonData = json_data
        self.fps = fps

    def process_direction(self, direction, speed):
        dr = 0.2
        if direction == 'backward' or direction == 'forward':
            if direction == 'forward':
                dr = -0.2
            if speed == 'low':
                dr *= 0.5
            return 0, 0, dr

        dphi, dtheta = 20, 15
        if direction == 'left':
            dphi, dtheta = -dphi, 0
        elif direction == 'right':
            dphi, dtheta = dphi, 0
        elif direction == 'up':
            dphi, dtheta = 0, -dtheta
        elif direction == 'down':
            dphi, dtheta = 0, dtheta
        elif direction == 'leftup45':
            dphi, dtheta = -dphi * 0.707, -dtheta * 0.707
        elif direction == 'rightup45':
            dphi, dtheta = dphi * 0.707, -dtheta * 0.707
        elif direction == 'leftdown45':
            dphi, dtheta = -dphi * 0.707, dtheta * 0.707
        elif direction == 'rightdown45':
            dphi, dtheta = dphi * 0.707, dtheta * 0.707
        elif direction == 'leftup30':
            dphi, dtheta = -dphi * 0.5, -dtheta * 0.866
        elif direction == 'rightup30':
            dphi, dtheta = dphi * 0.5, -dtheta * 0.866
        elif direction == 'leftdown30':
            dphi, dtheta = -dphi * 0.5, dtheta * 0.866
        elif direction == 'rightdown30':
            dphi, dtheta = dphi * 0.5, dtheta * 0.866
        elif direction == 'leftup60':
            dphi, dtheta = -dphi * 0.866, -dtheta * 0.5
        elif direction == 'rightup60':
            dphi, dtheta = dphi * 0.866, -dtheta * 0.5
        elif direction == 'leftdown60':
            dphi, dtheta = -dphi * 0.866, dtheta * 0.5
        elif direction == 'rightdown60':
            dphi, dtheta = dphi * 0.866, dtheta * 0.5
        else:
            dphi, dtheta = 0, 0

        if speed == 'low':
            dphi *= 0.5
            dtheta *= 0.5

        return dphi, dtheta, 0

    def tune_pose(self, key, records):
        phi, theta, r = 0, 0, 0
        out = np.array([[phi], [theta], [r]])
        last_time = 0
        for record in records:
            if last_time != record['starttime']:
                still_frame_num = int(self.fps * (record['starttime'] - last_time))
                still_out = np.array([[out[0, -1]] * still_frame_num,
                                      [out[1, -1]] * still_frame_num,
                                      [out[2, -1]] * still_frame_num])
                out = np.concatenate((out, still_out), axis=1)
            frame_num = int(self.fps * (record['endtime'] - record['starttime']))
            dphi, dtheta, dr = self.process_direction(record['direction'], record['speed'])
            t_out = np.zeros((3, frame_num))
            t_out[:, 0] = out[:, -1] + 1.0 / self.fps * np.array([dphi, dtheta, dr])
            for i in range(1, frame_num):
                t_out[:, i] = t_out[:, i - 1] + 1.0 / self.fps * np.array([dphi, dtheta, dr])
            out = np.concatenate((out, t_out), axis=1)
            last_time = record['endtime']
        return out.T

    def process(self):
        for key, records in tqdm(self.JsonData.items(), desc="Processing trajectories"):
            out = self.tune_pose(key, records)
            out = out.tolist()
            # 保留三位小数
            for i in range(len(out)):
                out[i] = [round(x, 3) for x in out[i]]
            self.JsonData[key] = out
        return self.JsonData

def extract_viewcrafter_input(json_data):
    """
    从处理后的 JSON 数据中提取键 "1" 的轨迹数据，
    将 theta、phi、r 分别保存为三个列表，并返回这三行数据构成的字符串列表
    """
    trajectory = json_data.get("1", [])
    if not trajectory:
        print("警告：在轨迹数据中没有找到键 '1' 对应的轨迹！")
        return None

    theta, phi, r = [], [], []
    for point in trajectory:
        if len(point) == 3:
            theta.append(point[0])
            phi.append(point[1])
            r.append(point[2])
        else:
            print(f"警告: 数据点格式不正确: {point}")

    theta_line = ' '.join(map(str, theta))
    phi_line = ' '.join(map(str, phi))
    r_line = ' '.join(map(str, r))
    return [theta_line, phi_line, r_line]

def process_ss(ss):
    """
    输入 ss 标签化文本，输出 w2c 变换矩阵
    """
    # 解析标签化文本为 JSON
    jsondata = parse_tagged_text_to_json(ss)
    # 处理 JSON 轨迹数据
    t2v = Text2VideoSet(json_data=jsondata, fps=8)
    processed_data = t2v.process()
    # 提取 viewcrafter 格式的字符串列表
    viewcrafter_lines = extract_viewcrafter_input(processed_data)
    # 根据 viewcrafter_lines 生成 w2c 变换矩阵
    w2c = generate_traj_txt(viewcrafter_lines)
    w2c = torch.from_numpy(w2c).to(torch.float32)
    # w2c = w2c[:,:3,:]
    # w2cflat=w2c.reshape(1,w2c.shape[0],-1)
    return w2c











def encode_and_save_batch(
    text_encoder: T5EncoderModel, batch: list[ItemInfo], device: torch.device, accelerator: Optional[accelerate.Accelerator]
):
    
    prompts = [item.caption for item in batch][0]
    
    if "#trajectory" in prompts:
        camerapath=re.findall(r"\{(.*?)\}", prompts)[0]
        camerainfo=load_file(camerapath)
        for key,value in camerainfo.items():
            camerainfo[key]=value.float().detach().cpu()
        pseudocamera=[camerainfo for _ in range(len(batch))]
        for item,emb in zip(batch,pseudocamera):
            save_camera_encoder_output_cache_wan(item,emb)
        
        #提取prompts的中括号中的字符串，获取相对地址，拼接上绝对地址获取.pt
    else:
        imagepath=[item.item_key for item in batch][0]
        intrinsics = get_intrinsics_from_video(imagepath)
        # from IPython import embed;embed()
        embed=process_ss(prompts)
        embed=torch.tensor(embed).to('cuda')
        mydict={
            'intrinsics': torch.tensor(intrinsics),
            'extrinsics': embed,
        }
        pseudocamera=[mydict for _ in range(len(batch))]
        for item,emb in zip(batch,pseudocamera):
            save_camera_encoder_output_cache_wan(item,emb)



def main(args):
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_WAN)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets
    # from IPython import embed;embed()
    # define accelerator for fp8 inference
    config = wan_t2v_14B.t2v_14B  # all Wan2.1 models have the same config for t5
    accelerator = None
    if args.fp8_t5:
        accelerator = accelerate.Accelerator(mixed_precision="bf16" if config.t5_dtype == torch.bfloat16 else "fp16")

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_camera_encoder_outputs.prepare_cache_files_and_paths_camera(datasets)

    # Load T5
    logger.info(f"Loading T5: {args.t5}")
    text_encoder = T5EncoderModel(
        text_len=config.text_len, dtype=config.t5_dtype, device=device, weight_path=args.t5, fp8=args.fp8_t5
    )

    # Encode with T5
    logger.info("Encoding with T5")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(text_encoder, batch, device, accelerator)
    # from IPython import embed;embed()
    cache_camera_encoder_outputs.process_camera_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del text_encoder

    # remove cache files not in dataset
    cache_camera_encoder_outputs.post_process_cache_files(datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset)


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--t5", type=str, default=None, required=True, help="text encoder (T5) checkpoint path")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    return parser


if __name__ == "__main__":
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = wan_setup_parser(parser)

    args = parser.parse_args()
    main(args)
