import cv2
import math
import numpy as np
import os
import joblib
from ultralytics import YOLO
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


def calculate_angle(a, b, c):
    """计算夹角 (b为顶点)，返回 0-180 度"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    if np.all(a == 0) or np.all(b == 0) or np.all(c == 0):
        return 0.0
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


class VisionThread(QThread):
    frame_signal = Signal(QImage)
    result_signal = Signal(list)
    msg_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera_id = 0
        self.model = None
        self.clf_model = None

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

        # 1. 加载推断模型
        if self.model is None:
            self.msg_signal.emit("⏳ 正在加载 YOLO 姿态推断模型...")
            self.model = YOLO('yolo26m-pose.pt')
            self.msg_signal.emit("✅ 姿态模型就绪。")

        # 2. 加载分类器
        if self.clf_model is None:
            if os.path.exists("pose_classifier.pkl"):
                self.msg_signal.emit("⏳ 正在加载动作分类器...")
                self.clf_model = joblib.load("pose_classifier.pkl")
                self.msg_signal.emit("✅ 分类器加载成功。")
            else:
                self.msg_signal.emit("⚠️ 警告: 未找到 pose_classifier.pkl，请先训练模型。")

        self.msg_signal.emit(f"正在连接视频设备 {self.camera_id} ...")
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            self.msg_signal.emit(f"❌ 连接失败：设备 {self.camera_id} 可能被占用。")
            self.running = False
            return

        self.msg_signal.emit("🟢 视觉模块已启动，正在进行实时推断...")

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # 硬件加速推断
            try:
                results = self.model(frame, verbose=False, classes=0, device=0, half=True)
            except Exception:
                results = self.model(frame, verbose=False, classes=0)

            annotated_frame = results[0].plot()
            table_data = []

            # ================= 核心推断与物理限位 =================
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                # 目标锁定：仅处理画面中面积最大的人体
                boxes = results[0].boxes.xyxy.cpu().numpy()
                target_idx = np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) if len(boxes) > 1 else 0

                kpts = results[0].keypoints.xy[target_idx].cpu().numpy()
                confs = results[0].keypoints.conf[target_idx].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)

                if len(kpts) == 17:
                    action_name = "未识别"

                    # 1. 提取归一化坐标特征 (34维)
                    hip_center_x, hip_center_y = (kpts[11][0] + kpts[12][0]) / 2, (kpts[11][1] + kpts[12][1]) / 2
                    sh_center_y = (kpts[5][1] + kpts[6][1]) / 2
                    torso_len = max(10.0, abs(hip_center_y - sh_center_y))

                    features = []
                    for x, y in kpts:
                        rel_x = (x - hip_center_x) / torso_len if x > 0 else 0
                        rel_y = (y - hip_center_y) / torso_len if y > 0 else 0
                        features.extend([rel_x, rel_y])

                    # 2. 提取核心夹角特征 (8维)
                    angles = [
                        calculate_angle(kpts[5], kpts[7], kpts[9]),    # L-Elbow
                        calculate_angle(kpts[6], kpts[8], kpts[10]),   # R-Elbow
                        calculate_angle(kpts[11], kpts[5], kpts[7]),   # L-Shoulder
                        calculate_angle(kpts[12], kpts[6], kpts[8]),   # R-Shoulder
                        calculate_angle(kpts[5], kpts[11], kpts[13]),  # L-Hip
                        calculate_angle(kpts[6], kpts[12], kpts[14]),  # R-Hip
                        calculate_angle(kpts[11], kpts[13], kpts[15]), # L-Knee
                        calculate_angle(kpts[12], kpts[14], kpts[16])  # R-Knee
                    ]
                    features.extend(angles)

                    # 3. 数据完整性熔断：下半身关键点丢失时直接拦截
                    legs_invisible = (
                        np.all(kpts[13] == 0) or np.all(kpts[14] == 0) or
                        np.all(kpts[15] == 0) or np.all(kpts[16] == 0) or
                        np.mean(confs[11:17]) < 0.4
                    )

                    # 4. 分类器推断
                    if self.clf_model is not None:
                        try:
                            if legs_invisible:
                                action_name, action_conf = "正常活动", 1.0
                            else:
                                prob_distributions = self.clf_model.predict_proba([features])[0]
                                pred_class = np.argmax(prob_distributions)
                                action_conf = float(prob_distributions[pred_class])

                                action_name = "正常活动" if pred_class == 4 else self.action_map.get(pred_class, "未知动作")

                                # 几何逻辑限位：拦截非标准的“蹲下”误触
                                if action_name == "蹲下" and (angles[6] > 130.0 or angles[7] > 130.0):
                                    action_name, action_conf = "正常活动", 0.0

                            table_data.append([f"ID-{target_idx + 1}", "人体姿态", f"{action_conf:.2f}", action_name])

                            # 画面内联渲染
                            if action_name != "正常活动" and action_conf > 0.45:
                                x1, y1 = int(results[0].boxes.xyxy[target_idx][0]), int(results[0].boxes.xyxy[target_idx][1])
                                cv2.putText(annotated_frame, f"{action_name} {action_conf:.2f}",
                                            (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        except Exception as e:
                            print(f"推断异常: {e}")

            self.result_signal.emit(table_data)

            # 格式转换推流
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frame_signal.emit(qt_img)

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait(2000)