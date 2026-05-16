import cv2
import math
import numpy as np
from ultralytics import YOLO
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


def calculate_angle(a, b, c):
    """计算角 a-b-c，b 是顶点。返回角度值(0-180)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


class VisionThread(QThread):
    frame_signal = Signal(QImage)
    result_signal = Signal(list)
    msg_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera_id = 0
        self.model = None

    def set_camera(self, index):
        self.camera_id = index

    def run(self):
        self.running = True

        if self.model is None:
            self.msg_signal.emit("⏳ 正在加载 YOLO26m-Pose 大模型权重...")
            self.model = YOLO('yolo26m-pose.pt')
            self.msg_signal.emit("✅ 模型加载完毕！")

        self.msg_signal.emit(f"正在通过 DirectShow 启动设备 {self.camera_id} ...")

        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        # 保持你原有的 720p 分辨率设置不变
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            self.msg_signal.emit(f"❌ 连接失败：请检查设备 {self.camera_id} 是否被其他程序占用")
            self.running = False
            return

        self.msg_signal.emit("🟢 视觉引擎全速运转中 (已尝试启用 GPU 加速)...")

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ================= 核心修改区：仅强制启用 GPU =================
            try:
                # 尝试强制使用独立显卡(device=0)和半精度(half=True)
                results = self.model(frame, verbose=False, classes=0, device=0, half=True)
            except Exception as e:
                # 如果当前环境没有装 PyTorch-CUDA，防止系统崩溃，自动退回 CPU 模式
                results = self.model(frame, verbose=False, classes=0)
            # ==============================================================

            annotated_frame = results[0].plot()
            table_data = []

            # 提取坐标与动作判定逻辑
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                for i, person_kpts in enumerate(results[0].keypoints.xy):
                    kpts = person_kpts.cpu().numpy()
                    if len(kpts) == 17:
                        l_sh, r_sh = kpts[5], kpts[6]
                        l_wr, r_wr = kpts[9], kpts[10]
                        l_hip, r_hip = kpts[11], kpts[12]
                        l_knee, r_knee = kpts[13], kpts[14]
                        l_ank, r_ank = kpts[15], kpts[16]

                        action_name = "未识别"

                        # 动态参照物：计算躯干大致长度，替代写死的像素值
                        # 这样无论人离摄像头多远，比例关系都是恒定的
                        torso_length = max(10.0, ((l_hip[1] + r_hip[1]) / 2) - ((l_sh[1] + r_sh[1]) / 2))

                        # 计算双腿的夹角 (Hip-Knee-Ankle)
                        l_leg_angle = calculate_angle(l_hip, l_knee, l_ank)
                        r_leg_angle = calculate_angle(r_hip, r_knee, r_ank)

                        # 1. 举双手 (Raise Hands)
                        if l_wr[1] < l_sh[1] - torso_length * 0.2 and r_wr[1] < r_sh[1] - torso_length * 0.2:
                            action_name = "举双手"

                        # 2. 蹲下 (Squat)
                        elif l_leg_angle < 120 and r_leg_angle < 120 and \
                             ((l_hip[1] > l_knee[1] - torso_length * 0.3) or (r_hip[1] > r_knee[1] - torso_length * 0.3)) and \
                             (l_sh[1] < l_hip[1] and r_sh[1] < r_hip[1]):
                            action_name = "蹲下"

                        # 3. 弓箭步 (Lunge)
                        elif (l_leg_angle < 120 and r_leg_angle > 150) or (r_leg_angle < 120 and l_leg_angle > 150):
                            action_name = "弓箭步"

                        # 4. 大字站 (Star Stand)
                        elif l_leg_angle > 140 and r_leg_angle > 140 and \
                             (abs(l_ank[0] - r_ank[0]) > abs(l_hip[0] - r_hip[0]) * 1.5) and \
                             (abs(l_wr[0] - r_wr[0]) > abs(l_sh[0] - r_sh[0]) * 1.5) and \
                             (l_wr[1] > l_sh[1] - torso_length * 0.5):
                            action_name = "大字站"

                        conf = float(results[0].boxes.conf[i].cpu().numpy()) if results[0].boxes else 0.0
                        table_data.append([f"ID-{i + 1}", "人体姿态", f"{conf:.2f}", action_name])

            self.result_signal.emit(table_data)

            # 图像转换与推流
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_signal.emit(qt_img)

        # 安全释放硬件资源
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait(2000)