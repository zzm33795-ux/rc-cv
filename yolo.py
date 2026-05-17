import cv2
import math
import numpy as np
import os
import joblib  # 引入 joblib 用于加载机器学习模型
from ultralytics import YOLO
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


def calculate_angle(a, b, c):
    """计算角 a-b-c，b 是顶点。返回角度值(0-180)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    if np.all(a == 0) or np.all(b == 0) or np.all(c == 0):
        return 0.0
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
        self.clf_model = None  # 动作分类器模型

        # 动作名称映射表
        self.action_map = {
            0: "大字站",
            1: "弓箭步",
            2: "举双手",
            3: "蹲下",
            4: "其他"
        }

    def set_camera(self, index):
        self.camera_id = index

    def run(self):
        self.running = True

        # 1. 加载 YOLO 姿态大模型
        if self.model is None:
            self.msg_signal.emit("⏳ 正在加载 YOLO26m-Pose 大模型权重...")
            self.model = YOLO('yolo26m-pose.pt')
            self.msg_signal.emit("✅ YOLO 模型加载完毕！")

        # 2. 加载机器学习动作分类大脑
        if self.clf_model is None:
            if os.path.exists("pose_classifier.pkl"):
                self.msg_signal.emit("⏳ 正在加载动作分类大脑...")
                self.clf_model = joblib.load("pose_classifier.pkl")
                self.msg_signal.emit("✅ 随机森林分类模型加载成功！")
            else:
                self.msg_signal.emit("⚠️ 警告: 找不到 pose_classifier.pkl，请先采集数据并训练模型！")

        self.msg_signal.emit(f"正在通过 DirectShow 启动设备 {self.camera_id} ...")
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            self.msg_signal.emit(f"❌ 连接失败：请检查设备 {self.camera_id} 是否被其他程序占用")
            self.running = False
            return

        self.msg_signal.emit("🟢 视觉引擎全速运转中 (已启用 AI 动作推断)...")

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # YOLO 硬件加速推理
            try:
                results = self.model(frame, verbose=False, classes=0, device=0, half=True)
            except Exception:
                results = self.model(frame, verbose=False, classes=0)

            annotated_frame = results[0].plot()
            table_data = []

            # ================= 核心重构区：AI 模型推断 =================
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                for i, person_kpts in enumerate(results[0].keypoints.xy):
                    kpts = person_kpts.cpu().numpy()

                    if len(kpts) == 17:
                        action_name = "未识别"

                        # 1. 计算归一化基准：髋关节中心点 和 躯干长度
                        hip_center_x = (kpts[11][0] + kpts[12][0]) / 2
                        hip_center_y = (kpts[11][1] + kpts[12][1]) / 2
                        sh_center_y = (kpts[5][1] + kpts[6][1]) / 2
                        torso_len = max(10.0, abs(hip_center_y - sh_center_y))

                        features = []

                        # 2. 提取 34 维基础坐标特征 (完全复刻采集脚本)
                        for x, y in kpts:
                            rel_x = (x - hip_center_x) / torso_len if x > 0 else 0
                            rel_y = (y - hip_center_y) / torso_len if y > 0 else 0
                            features.extend([rel_x, rel_y])

                        # 3. 提取 8 维关键角度特征
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

                        # 4. 喂给机器学习模型进行预测
                        # 4. 喂给机器学习模型进行预测
                        if self.clf_model is not None:
                            try:
                                # 获取所有类别的概率分布
                                prob_distributions = self.clf_model.predict_proba([features])[0]
                                pred_class = np.argmax(prob_distributions)
                                action_conf = float(prob_distributions[pred_class])  # 这才是真正的动作置信度

                                if pred_class == 4:
                                    action_name = "正常活动"
                                else:
                                    action_name = self.action_map.get(pred_class, "未知动作")

                                # 将真正的 action_conf 传给 UI
                                table_data.append([f"ID-{i + 1}", "人体姿态", f"{action_conf:.2f}", action_name])

                                # 【加分项】直接在视频画面的人头顶上画出动作名字
                                if action_name != "正常活动" and action_conf > 0.45:
                                    x1, y1 = int(results[0].boxes.xyxy[i][0]), int(results[0].boxes.xyxy[i][1])
                                    cv2.putText(annotated_frame, f"{action_name} {action_conf:.2f}",
                                                (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            except Exception as e:
                                print(f"动作预测出错: {e}")

            # ==========================================================

            self.result_signal.emit(table_data)

            # 图像转换与推流
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.frame_signal.emit(qt_img)

        # 安全释放硬件资源
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait(2000)