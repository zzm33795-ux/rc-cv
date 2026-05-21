import cv2
import numpy as np
import csv
import os
import math
import time
from ultralytics import YOLO

# 动作映射表保持不变
ACTION_MAP = {'0': "大字站", '1': "弓箭步", '2': "举双手", '3': "蹲下", '4': "其他"}
DISPLAY_MAP = {'0': "STAR STAND", '1': "LUNGE", '2': "RAISE HANDS", '3': "SQUAT", '4': "OTHERS"}
CSV_FILE = "pose_data_optimized.csv"


def init_csv():
    """初始化 CSV 文件，加入额外的角度特征列"""
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
        print(f"✅ 优化版数据集文件已初始化: {CSV_FILE}")


def calculate_angle(p1, p2, p3):
    """
    计算三点构成的夹角 (p2 为顶点)
    """
    if np.all(p1 == 0) or np.all(p2 == 0) or np.all(p3 == 0):
        return 0.0  # 关键点缺失时返回0

    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # 归一化向量并计算点积
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)


def main():
    init_csv()

    print("⏳ 正在加载 yolo26m-pose 模型权重...")
    model = YOLO('yolo26m-pose.pt')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = "idle"          # idle / countdown / recording
    current_label = None
    memory_buffer = []
    countdown_end = 0
    recording_end = 0

    print("\n" + "=" * 40)
    print("🚀 优化版姿态采集系统已启动（附带主目标锁定&角度特征提取）")
    print("=" * 40 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        try:
            results = model(frame, verbose=False, classes=0, device=0, half=True)
        except Exception:
            results = model(frame, verbose=False, classes=0)

        annotated_frame = results[0].plot()
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): break

        # 状态机：idle → countdown → recording → idle
        if state == "idle":
            key_char = chr(key) if key != 255 else None
            if key_char in ACTION_MAP:
                current_label = key_char
                memory_buffer = []
                countdown_end = time.time() + 5
                state = "countdown"
                print(f"⏳ 3秒倒计时开始，请就位【{ACTION_MAP[current_label]}】...")

        elif state == "countdown":
            if key == ord(' '):
                state = "idle"
                print("⚠️ 已取消。")
                continue
            if time.time() >= countdown_end:
                state = "recording"
                recording_end = time.time() + 15
                print(f"🎬 开始录制【{ACTION_MAP[current_label]}】，15秒后自动结束...")

        elif state == "recording":
            if key == ord(' '):
                state = "idle"
                if len(memory_buffer) > 0:
                    print(f"💾 提前结束，正在写入 {len(memory_buffer)} 帧...")
                    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerows(memory_buffer)
                    print(f"✅ 【{ACTION_MAP[current_label]}】保存成功！\n")
                memory_buffer = []
                continue
            if time.time() >= recording_end:
                state = "idle"
                if len(memory_buffer) > 0:
                    print(f"💾 15秒到，正在写入 {len(memory_buffer)} 帧...")
                    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerows(memory_buffer)
                    print(f"✅ 【{ACTION_MAP[current_label]}】保存成功！\n")
                memory_buffer = []
                continue

        # ---------------- 核心数据提取区域 ----------------
        if (state == "recording" or state == "countdown") and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # 【优化1：防识别错人体】
            # 计算画面中所有人的 Bounding Box 面积，锁定面积最大的那个（通常是操作者本人）
            boxes = results[0].boxes.xyxy.cpu().numpy()
            target_idx = 0
            if len(boxes) > 1:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                target_idx = np.argmax(areas)

            kpts = results[0].keypoints.xy[target_idx].cpu().numpy()
            confs = results[0].keypoints.conf[target_idx].cpu().numpy() if results[
                                                                               0].keypoints.conf is not None else np.ones(
                17)

            # 【优化2：脏数据过滤】
            # 针对侧面弓箭步和深蹲，如果下半身（11-16）置信度太低，说明没拍全，抛弃这一帧
            if np.mean(confs[11:17]) < 0.4:
                cv2.putText(annotated_frame, "WARNING: LEGS NOT VISIBLE!", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                # 不执行 append，跳过这一帧
            elif len(kpts) == 17:
                hip_center_x = (kpts[11][0] + kpts[12][0]) / 2
                hip_center_y = (kpts[11][1] + kpts[12][1]) / 2
                sh_center_y = (kpts[5][1] + kpts[6][1]) / 2
                torso_len = max(10.0, abs(hip_center_y - sh_center_y))

                features = []
                # 提取 1：归一化坐标
                for x, y in kpts:
                    rel_x = (x - hip_center_x) / torso_len if x > 0 else 0
                    rel_y = (y - hip_center_y) / torso_len if y > 0 else 0
                    features.extend([rel_x, rel_y])

                # 【优化3：引入关节角度特征】
                # 这8个角度对分类 这4个特定动作 极其敏感
                angles = [
                    calculate_angle(kpts[5], kpts[7], kpts[9]),  # 左肘夹角 (大字/举手 区分关键)
                    calculate_angle(kpts[6], kpts[8], kpts[10]),  # 右肘夹角
                    calculate_angle(kpts[11], kpts[5], kpts[7]),  # 左肩腋下夹角 (大字站关键)
                    calculate_angle(kpts[12], kpts[6], kpts[8]),  # 右肩腋下夹角
                    calculate_angle(kpts[5], kpts[11], kpts[13]),  # 左髋夹角 (深蹲/弓箭步 区分关键)
                    calculate_angle(kpts[6], kpts[12], kpts[14]),  # 右髋夹角
                    calculate_angle(kpts[11], kpts[13], kpts[15]),  # 左膝夹角 (深蹲/弓箭步 区分关键)
                    calculate_angle(kpts[12], kpts[14], kpts[16])  # 右膝夹角
                ]
                features.extend(angles)
                features.append(int(current_label))
                if state == "recording":
                    memory_buffer.append(features)  # 只在正式录制阶段存数据
        # ----------------------------------------------

        # UI 渲染
        if state == "countdown":
            remaining = max(0, int(countdown_end - time.time()) + 1)
            status_text = f"COUNTDOWN: {remaining}s | {DISPLAY_MAP[current_label]}"
            color = (0, 255, 255)
        elif state == "recording":
            remaining = max(0, int(recording_end - time.time()))
            status_text = f"REC [{DISPLAY_MAP[current_label]}] | {remaining}s left | Frames: {len(memory_buffer)}"
            color = (0, 0, 255)
        else:
            status_text = "STATUS: READY (Press 0-4)"
            color = (0, 255, 0)

        cv2.putText(annotated_frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.imshow("Data Collector", annotated_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()