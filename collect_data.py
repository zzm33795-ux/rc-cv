import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO

# 1. 后台数据映射表（保持不变，用于终端提示和数字保存）
ACTION_MAP = {
    '0': "大字站",
    '1': "弓箭步",
    '2': "举双手",
    '3': "蹲下",
    '4': "其他_正常走动弯腰"
}

# 2. 专门给 OpenCV 屏幕显示用的英文表（彻底解决问号乱码）
DISPLAY_MAP = {
    '0': "STAR STAND",
    '1': "LUNGE",
    '2': "RAISE HANDS",
    '3': "SQUAT",
    '4': "OTHERS / NOISE"
}

CSV_FILE = "pose_data.csv"


def init_csv():
    """初始化 CSV 文件，写入表头"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = []
            for i in range(17):
                headers.extend([f"x_{i}", f"y_{i}"])
            headers.append("label")
            writer.writerow(headers)
        print(f"✅ 数据集文件已初始化: {CSV_FILE}")


def main():
    init_csv()

    print("⏳ 正在加载超轻量级 yolo26n-pose 模型权重...")
    model = YOLO('yolo26n-pose.pt')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 状态控制
    is_recording = False
    current_label = None
    memory_buffer = []  # 内存缓冲区

    print("\n" + "=" * 40)
    print("🚀 姿态数据采集系统（手动开关版）已启动！")
    print("💡 操作指南：")
    print("  1. 摆好姿势，按一下数字键 [0-4] 开始录制")
    print("  2. 录制结束时，按一下 [空格键] 保存并停止")
    print("  3. [按下 q 键] -> 退出程序")
    print("=" * 40 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 硬件加速推理
        try:
            results = model(frame, verbose=False, classes=0, device=0, half=True)
        except Exception:
            results = model(frame, verbose=False, classes=0)

        annotated_frame = results[0].plot()

        # 按键检测
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # 核心控制状态机
        if not is_recording:
            # 未录制状态下：检测 0-4 开启录制
            key_char = chr(key) if key != 255 else None
            if key_char in ACTION_MAP:
                current_label = key_char
                is_recording = True
                memory_buffer = []
                print(f"🎬 开始录制【{ACTION_MAP[current_label]}】，完成时请按 [空格键] 结束...")
        else:
            # 录制状态下：检测空格键结束并保存
            if key == ord(' '):
                is_recording = False
                if len(memory_buffer) > 0:
                    print(f"💾 正在将 {len(memory_buffer)} 帧数据写入 CSV...")
                    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerows(memory_buffer)
                    print(f"✅ 【{ACTION_MAP[current_label]}】保存成功！\n")
                memory_buffer = []

        # 录制时抓取骨骼点存入内存
        if is_recording and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            if len(kpts) == 17:
                # 归一化计算
                hip_center_x = (kpts[11][0] + kpts[12][0]) / 2
                hip_center_y = (kpts[11][1] + kpts[12][1]) / 2
                sh_center_y = (kpts[5][1] + kpts[6][1]) / 2
                torso_len = max(10.0, abs(hip_center_y - sh_center_y))

                features = []
                for x, y in kpts:
                    rel_x = (x - hip_center_x) / torso_len
                    rel_y = (y - hip_center_y) / torso_len
                    features.extend([rel_x, rel_y])

                features.append(int(current_label))
                memory_buffer.append(features)

        # 界面 UI 渲染（全英文，规避乱码）
        if is_recording:
            status_text = f"RECORDING [{DISPLAY_MAP[current_label]}] | Frames: {len(memory_buffer)}"
            text_color = (0, 0, 255)  # 红色
        else:
            status_text = "STATUS: READY (Press 0-4 to start)"
            text_color = (0, 255, 0)  # 绿色

        cv2.putText(annotated_frame, status_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        cv2.imshow("Data Collector (Click to Start / Space to Stop)", annotated_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()