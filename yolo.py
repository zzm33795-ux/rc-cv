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

            # ================= 核心重构区：AI 模型推断与多重物理锁互锁 =================
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                # 【加分项优化：主目标锁定机制】防止身后的裁判或路人乱入干扰记分板
                boxes = results[0].boxes.xyxy.cpu().numpy()
                target_idx = 0
                if len(boxes) > 1:
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    target_idx = np.argmax(areas)

                kpts = results[0].keypoints.xy[target_idx].cpu().numpy()

                # 获取关键点的置信度，用来判断腿到底有没有露出来
                confs = results[0].keypoints.conf[target_idx].cpu().numpy() if results[
                                                                                   0].keypoints.conf is not None else np.ones(
                    17)

                if len(kpts) == 17:
                    action_name = "未识别"

                    # 1. 计算归一化基准：髋关节中心点 和 躯干长度
                    hip_center_x = (kpts[11][0] + kpts[12][0]) / 2
                    hip_center_y = (kpts[11][1] + kpts[12][1]) / 2
                    sh_center_y = (kpts[5][1] + kpts[6][1]) / 2
                    torso_len = max(10.0, abs(hip_center_y - sh_center_y))

                    features = []

                    # 2. 提取 34 维基础坐标特征
                    for x, y in kpts:
                        rel_x = (x - hip_center_x) / torso_len if x > 0 else 0
                        rel_y = (y - hip_center_y) / torso_len if y > 0 else 0
                        features.extend([rel_x, rel_y])

                    # 3. 提取 8 维关键角度特征
                    angles = [
                        calculate_angle(kpts[5], kpts[7], kpts[9]),  # 左肘 [0]
                        calculate_angle(kpts[6], kpts[8], kpts[10]),  # 右肘 [1]
                        calculate_angle(kpts[11], kpts[5], kpts[7]),  # 左肩 [2]
                        calculate_angle(kpts[12], kpts[6], kpts[8]),  # 右肩 [3]
                        calculate_angle(kpts[5], kpts[11], kpts[13]),  # 左髋 [4]
                        calculate_angle(kpts[6], kpts[12], kpts[14]),  # 右髋 [5]
                        calculate_angle(kpts[11], kpts[13], kpts[15]),  # 左膝 [6]
                        calculate_angle(kpts[12], kpts[14], kpts[16])  # 右膝 [7]
                    ]
                    features.extend(angles)

                    # 4. 【核心防误触】多重物理限位锁检查
                    # 互锁关卡 A：检查下半身核心关键点（左膝13, 右膝14, 左踝15, 右踝16）是否丢失或置信度太低
                    legs_invisible = (
                            np.all(kpts[13] == 0) or np.all(kpts[14] == 0) or
                            np.all(kpts[15] == 0) or np.all(kpts[16] == 0) or
                            np.mean(confs[11:17]) < 0.4  # 平均置信度低于阈值说明下半身被遮挡或未入画
                    )

                    # 5. 喂给机器学习模型进行预测
                    if self.clf_model is not None:
                        try:
                            if legs_invisible:
                                # 如果触发了关卡 A（腿没露全），直接熔断拦截，不让 AI 盲猜
                                action_name = "正常活动"
                                action_conf = 1.0
                            else:
                                # 只有腿部健全在画面内，才允许神经网络大脑介入推断
                                prob_distributions = self.clf_model.predict_proba([features])[0]
                                pred_class = np.argmax(prob_distributions)
                                action_conf = float(prob_distributions[pred_class])  # 真正的动作置信度

                                if pred_class == 4:
                                    action_name = "正常活动"
                                else:
                                    action_name = self.action_map.get(pred_class, "未知动作")

                                # 互锁关卡 B：后置几何夹角锁
                                # 如果 AI 觉得是“蹲下”，但测出的膝盖还挺直着（角度 > 130°），必然是误触！
                                if action_name == "蹲下":
                                    left_knee_angle = angles[6]
                                    right_knee_angle = angles[7]
                                    if left_knee_angle > 130.0 or right_knee_angle > 130.0:
                                        action_name = "正常活动"
                                        action_conf = 0.0

                            # 将真正的 action_conf 传给 UI
                            table_data.append([f"ID-{target_idx + 1}", "人体姿态", f"{action_conf:.2f}", action_name])

                            # 在视频画面上实时渲染动作和置信度
                            if action_name != "正常活动" and action_conf > 0.45:
                                x1, y1 = int(results[0].boxes.xyxy[target_idx][0]), int(
                                    results[0].boxes.xyxy[target_idx][1])
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