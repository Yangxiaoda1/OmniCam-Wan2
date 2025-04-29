import numpy as np
import copy
import os
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from PIL import Image
import torchvision
from scipy.interpolate import UnivariateSpline



def sphere2pose(c2ws_input, theta, phi, r):

    c2ws = copy.deepcopy(c2ws_input)

    #先沿着世界坐标系z轴方向平移再旋转
    c2ws[:,2,3] += r

    theta = torch.deg2rad(torch.tensor(theta))
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = torch.tensor([[1, 0, 0, 0],
                    [0, cos_value_x, -sin_value_x, 0],
                    [0, sin_value_x, cos_value_x, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0],1,1)

    phi = torch.deg2rad(torch.tensor(phi))
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = torch.tensor([[cos_value_y, 0, sin_value_y, 0],
                    [0, 1, 0, 0],
                    [-sin_value_y, 0, cos_value_y, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0],1,1)

    c2ws = np.matmul(rot_mat_x, c2ws)
    c2ws = np.matmul(rot_mat_y, c2ws)

    return c2ws 


def txt_interpolation(input_list,n,mode = 'smooth'):
    x = np.linspace(0, 1, len(input_list)) # 生成一个等间隔的 x 数组，长度与 input_list 相同，范围从 0 到 1
    if mode == 'smooth':#三次样条插值
        f = UnivariateSpline(x, input_list, k=3)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)# 生成新的等间隔 x 数组，用于获取 n 个插值后的数据点
    ynew = f(xnew)# 通过插值函数 f 计算新 x 值对应的 y 值
    return ynew


def generate_traj_txt(viewcrafter_lines):
    phi = np.array(list(map(float, viewcrafter_lines[0].split())), dtype=np.float32)[:25]
    theta   = np.array(list(map(float, viewcrafter_lines[1].split())), dtype=np.float32)[:25]
    r     = np.array(list(map(float, viewcrafter_lines[2].split())), dtype=np.float32)[:25]
    c2ws_anc = np.array(
        [
            [
                [ 1.0000e+00, -8.6427e-07,  2.3283e-09,  1.4707e-09],
                [ 8.6069e-07, -9.9609e-01, -8.7158e-02,  8.7708e-02],
                [-7.7647e-08,  8.7159e-02, -9.9609e-01,  1.0029e+00],
                [-2.6349e-15,  3.0323e-09, -6.1118e-09,  1.0000e+00]
            ]
        ],
        dtype=np.float32,
    )

     # Initialize a camera.
    """
     The camera coordinate sysmte in COLMAP is right-down-forward
     Pytorch3D is left-up-forward
     """
    frame=len(phi)
    print('frame:',frame)
    if len(phi)>3:
        phis = txt_interpolation(phi,frame,mode='smooth')
        phis[0] = phi[0]
        phis[-1] = phi[-1]
    else:
        phis = txt_interpolation(phi,frame,mode='linear')

    if len(theta)>3:
        thetas = txt_interpolation(theta,frame,mode='smooth')
        thetas[0] = theta[0]
        thetas[-1] = theta[-1]
    else:
        thetas = txt_interpolation(theta,frame,mode='linear')

    if len(r) >3:
        rs = txt_interpolation(r,frame,mode='smooth')
        rs[0] = r[0]
        rs[-1] = r[-1]        
    else:
        rs = txt_interpolation(r,frame,mode='linear')
    # phis=phi
    # thetas=theta
    # rs=r
    # print('frame:',len(phis))
    rs = rs * c2ws_anc[0, 2, 3]



    c2ws_list = []
    for th, ph, r in zip(thetas, phis, rs):
        c2w_new = sphere2pose(c2ws_anc, np.float32(th), np.float32(ph),np.float32(r))# 利用sphere2pose函数生成新的相机位姿矩阵
        c2ws_list.append(c2w_new)# 将生成的位姿矩阵添加到列表中
    c2ws = torch.cat(c2ws_list,dim=0)# 将列表中的所有相机位姿矩阵在第一维（batch）上连接成一个大的矩阵

    # poses = c2ws
    # frames = [visualizer_frame(poses, i) for i in range(len(poses))]
    # save_video(np.array(frames)/255.,os.path.join('/data/home/yangxiaoda/FlexCaM/descriptionjson-video/viz_traj.mp4'))

    R, T = c2ws[:, :3, :3], c2ws[:, :3, 3:]# 分离出旋转矩阵R和平移向量T
    ## 将dust3r坐标系转成pytorch3d坐标系
    R = np.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], 2)  # 将dust3r坐标系（右-下-前）转换为pytorch3d坐标系（左-上-前）
    new_c2w = np.concatenate([R, T], axis=2)# 将旋转矩阵R和平移向量T拼接，形成新的c2w变换矩阵
    # print(new_c2w.shape, np.array([[[0, 0, 0, 1]]]).shape)# 通过在new_c2w的下方添加一行[0, 0, 0, 1]来构造齐次变换矩阵，然后求其逆，得到从世界到相机的变换矩阵w2c
    w2c = np.linalg.inv(
         np.concatenate(
             (new_c2w, np.broadcast_to(np.array([[[0, 0, 0, 1]]]),(new_c2w.shape[0], 1, 4))),
             axis=1,
         )
     )

    return w2c# 返回最终的从世界到相机的变换矩阵


def parse_tagged_text_to_json(tagged_text):
    """
    将模型输出的标签化文本解析为JSON格式。
    假设标签化文本结构类似：
    <starttime>0</starttime><endtime>1</endtime><speed>high</speed><direction>down</direction><sep>...
    """

    # 先以<sep>分割不同的动作片段
    segments = tagged_text.split("<sep>")

    results = []
    for seg in segments:
        # 对于每个seg，解析其中的字段
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
            rotate=rotate_match.group(1)

            # 尝试转换starttime和endtime为数值类型
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
                "rotate":rotate
            })

    # 假设ID为1，这里可根据需要修改
    final_json = {"1": results}
    return final_json




class Text2VideoSet:
    def __init__(self, json_data, fps=8):
        # 直接传入 json_data，不再需要 output_dir
        self.JsonData = json_data
        self.fps = fps

    # 处理方向，返回变化率
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

    # 调整姿态
    def tune_pose(self, key, records):
        phi, theta, r = 0, 0, 0
        out = np.array([[phi], [theta], [r]])
        last_time = 0
        for record in records:
            # 如果存在间断时间，则复制上一个时刻的姿态
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

    # 处理 description JSON，生成轨迹数据并返回处理后的 JSON 数据
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



def visualizer_frame(camera_poses, highlight_index):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # 获取camera_positions[2]的最大值和最小值
    z_values = [pose[:3, 3][2] for pose in camera_poses]
    z_min, z_max = min(z_values), max(z_values)

    # 创建一个颜色映射对象
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["#00008B", "#ADD8E6"])
    # cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, pose in enumerate(camera_poses):
        camera_positions = pose[:3, 3]
        color = "blue" if i == highlight_index else "blue"
        size = 100 if i == highlight_index else 25
        color = sm.to_rgba(camera_positions[2])  # 根据camera_positions[2]的值映射颜色
        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            color=color,
            marker="o",
            s=size,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("Camera trajectory")
    ax.view_init(90+30, -90)

    plt.ylim(-0.1,0.2)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.close()

    return img


def save_video(data,images_path,folder=None):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder]*len(data)
        images = [np.array(Image.open(os.path.join(folder_name,path))) for folder_name,path in zip(folder,data)]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(images_path, tensor_data, fps=8, video_codec='h264', options={'crf': '10'})




if __name__ == "__main__":
    ss="<starttime>0</starttime><endtime>1</endtime><speed>high</speed><direction>down</direction><rotate>stay</rotate><sep><starttime>1</starttime><endtime>2</endtime><speed>low</speed><direction>right</direction><rotate>stay</rotate><sep><starttime>2</starttime><endtime>3</endtime><speed>high</speed><direction>rightup30</direction><rotate>stay</rotate><sep><starttime>3</starttime><endtime>4</endtime><speed>low</speed><direction>backward</direction><rotate>stay</rotate"
    jsondata=parse_tagged_text_to_json(ss)
    t2v = Text2VideoSet(json_data=jsondata, fps=8)
    processed_data = t2v.process()
    viewcrafter_lines = extract_viewcrafter_input(processed_data)
    w2c = generate_traj_txt(viewcrafter_lines)
    print(w2c.shape)