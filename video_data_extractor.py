# video_data_extractor.py
import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO

# 动作映射表（必须与训练和UI脚本严格一致）
ACTION_MAP = {
    0: "大字站",
    1: "弓箭步",
    2: "举双手",
    3: "蹲下",
    4: "其他"
}

CSV_FILE = "pose_data_optimized.csv"


def init_csv():
    """初始化 CSV 文件，确保表头一致"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = []
            # 1. 基础坐标特征 (34列)
            for i in range(17):
                headers.extend([f"x_{i}", f"y_{i}"])
            # 2. 补充角度特征 (8列)
            headers.extend([
                "angle_L_elbow", "angle_R_elbow",
                "angle_L_shoulder", "angle_R_shoulder",
                "angle_L_hip", "angle_R_hip",
                "angle_L_knee", "angle_R_knee"
            ])
            headers.append("label")
            writer.writerow(headers)
        print(f"✅ 数据集文件已初始化: {CSV_FILE}")


def calculate_angle(p1, p2, p3):
    """计算三点构成的夹角 (p2 为顶点)"""
    if np.all(p1 == 0) or np.all(p2 == 0) or np.all(p3 == 0):
        return 0.0
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def process_single_video(video_path, label):
    """处理单个视频，提取特征并写入 CSV"""
    if not os.path.exists(video_path):
        print(f"⚠️ 找不到视频文件: {video_path}，请检查路径和文件名！")
        return 0

    print(f"🎬 正在处理视频: {os.path.basename(video_path)} -> 对应动作: 【{ACTION_MAP[label]}】")
    cap = cv2.VideoCapture(video_path)

    # 自动加载你的本地模型
    model = YOLO('yolo26m-pose.pt')

    video_features = []
    frame_count = 0
    skipped_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 利用战7000的独显显卡全速加速推理
        try:
            results = model(frame, verbose=False, classes=0, device=0, half=True)
        except Exception:
            results = model(frame, verbose=False, classes=0)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # 1. 过滤路人干扰：锁定画面中面积最大的人体
            boxes = results[0].boxes.xyxy.cpu().numpy()
            target_idx = 0
            if len(boxes) > 1:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                target_idx = np.argmax(areas)

            kpts = results[0].keypoints.xy[target_idx].cpu().numpy()
            confs = results[0].keypoints.conf[target_idx].cpu().numpy() if results[
                                                                               0].keypoints.conf is not None else np.ones(
                17)

            # 2. 脏数据过滤：如下半身置信度均值太低（说明没拍全），则抛弃该帧
            if np.mean(confs[11:17]) < 0.4:
                skipped_count += 1
                continue

            if len(kpts) == 17:
                # 计算归一化基准
                hip_center_x = (kpts[11][0] + kpts[12][0]) / 2
                hip_center_y = (kpts[11][1] + kpts[12][1]) / 2
                sh_center_y = (kpts[5][1] + kpts[6][1]) / 2
                torso_len = max(10.0, abs(hip_center_y - sh_center_y))

                features = []
                # 特征 1：34维归一化相对坐标
                for x, y in kpts:
                    rel_x = (x - hip_center_x) / torso_len if x > 0 else 0
                    rel_y = (y - hip_center_y) / torso_len if y > 0 else 0
                    features.extend([rel_x, rel_y])

                # 特征 2：8维核心关节夹角
                angles = [
                    calculate_angle(kpts[5], kpts[7], kpts[9]),  # 左肘
                    calculate_angle(kpts[6], kpts[8], kpts[10]),  # 右肘
                    calculate_angle(kpts[11], kpts[5], kpts[7]),  # 左肩
                    calculate_angle(kpts[12], kpts[6], kpts[8]),  # 右肩
                    calculate_angle(kpts[5], kpts[11], kpts[13]),  # 左髋
                    calculate_angle(kpts[6], kpts[12], kpts[14]),  # 右髋
                    calculate_angle(kpts[11], kpts[13], kpts[15]),  # 左膝
                    calculate_angle(kpts[12], kpts[14], kpts[16])  # 右膝
                ]
                features.extend(angles)
                features.append(int(label))

                video_features.append(features)

    cap.release()

    # 批量将特征追加写入到本地 CSV 中
    if len(video_features) > 0:
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(video_features)
        print(
            f"💾 写入成功：总读取 {frame_count} 帧，过滤无效帧 {skipped_count} 帧，成功提取 {len(video_features)} 帧特征！\n")
    else:
        print("⚠️ 未在此视频中提取到任何有效帧数据。\n")

    return len(video_features)


if __name__ == '__main__':
    init_csv()

    # ==================== 选手绝对路径绑定区 ====================
    # 已完美对齐你的 C:\Users\zzm\Desktop\xunlianshi 路径与文件名
    video_tasks = [
        {"path": r"C:\Users\zzm\Desktop\xunlianshi\dazizhan.mp4", "label": 0},
        {"path": r"C:\Users\zzm\Desktop\xunlianshi\gongjianbu.mp4", "label": 1},
        {"path": r"C:\Users\zzm\Desktop\xunlianshi\jushuangshou.mp4", "label": 2},
        {"path": r"C:\Users\zzm\Desktop\xunlianshi\dunxia.mp4", "label": 3},
        {"path": r"C:\Users\zzm\Desktop\xunlianshi\suijidongzuo.mp4", "label": 4},  # 其他正常动作
    ]
    # ============================================================

    total_saved = 0
    for task in video_tasks:
        saved_frames = process_single_video(task["path"], task["label"])
        total_saved += saved_frames

    print(f"🎉【全部提取完毕】共向 {CSV_FILE} 成功写入了 {total_saved} 条训练样本数据！")
    print("👉 接下来你可以立即在控制台执行：python train_model.py 训练核心大脑了！")