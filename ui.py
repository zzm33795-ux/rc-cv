# ui.py
import sys
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                               QVBoxLayout, QHBoxLayout, QWidget, QFrame,
                               QComboBox, QSlider, QGroupBox, QGridLayout, QListWidget)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

# 导入自研核心模块
from yolo import VisionThread
from audio import ShazamAlgorithm, AudioThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机器人视觉与音频识别系统")
        self.resize(1150, 780)

        # 简洁专业的深色 QSS 样式表
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { font-family: 'Microsoft YaHei', 'Segoe UI'; color: #e0e0e0; }
            QFrame#Sidebar { background-color: #1e1e1e; border-right: 1px solid #333; }
            QLabel#AppTitle { color: #ffffff; font-size: 20px; font-weight: bold; padding: 15px 0px; border-bottom: 1px solid #333; }
            QLabel#StatusLabel { color: #aaaaaa; font-size: 14px; }
            QLabel#FpsLabel { color: #f39c12; font-size: 14px; font-weight: bold; padding: 5px; background: #2c3e50; border-radius: 4px; }
            QLabel#StatLabel { color: #2ecc71; font-size: 14px; font-weight: bold; }
            QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 15px; padding-top: 15px; font-size: 14px; color: #ccc; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #fff; }
            QComboBox { background-color: #2b2b2b; border: 1px solid #444; border-radius: 4px; padding: 5px; color: #fff; }
            QSlider::groove:horizontal { border-radius: 2px; height: 4px; background: #444; }
            QSlider::handle:horizontal { background: #3498db; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
            QPushButton { background-color: #34495e; border: none; border-radius: 4px; padding: 10px; font-weight: bold; color: #fff; font-size: 14px;}
            QPushButton:hover { background-color: #415b76; }
            QPushButton#BtnStart { background-color: #27ae60; font-size: 15px;}
            QPushButton#BtnStart:hover { background-color: #2ecc71; }
            QPushButton#BtnAction { background-color: #2980b9; font-size: 15px;}
            QPushButton#BtnAction:hover { background-color: #3498db; }
            QPushButton#BtnStop { background-color: #c0392b; }
            QPushButton#BtnStop:hover { background-color: #e74c3c; }
            QPushButton#BtnReset { background-color: #f39c12; color: #fff; }
            QPushButton#BtnReset:hover { background-color: #f1c40f; }

            QLabel.ScoreItem { font-size: 14px; background-color: #2b2b2b; padding: 8px; border-radius: 4px; border: 1px solid #444; }
            QLabel.ScoreItemReady { color: #aaaaaa; }
            QLabel.ScoreItemSuccess { color: #2ecc71; border: 1px solid #2ecc71; font-weight: bold; }

            QListWidget { background-color: #1a1a1a; border: 1px solid #444; border-radius: 4px; color: #2ecc71; padding: 5px; font-size: 13px;}
        """)

        self.vision_thread = VisionThread()
        self.shazam = ShazamAlgorithm()

        if not self.shazam.load_database("fingerprints.pkl"):
            print("提示：未检测到 fingerprints.pkl 音频特征库，请先运行特征提取。")

        self.audio_thread = AudioThread(self.shazam)

        self.fps_frames = 0
        self.fps_start_time = time.time()

        # --- 核心状态追踪变量 ---
        self.is_action_evaluating = False  # 动作考核状态锁
        self.success_count = 0
        self.target_actions = ["举双手", "大字站", "蹲下", "弓箭步"]
        self.locked_actions = {action: False for action in self.target_actions}

        # 【新增】防抖计数器：记录每个动作连续被识别到的帧数
        self.action_counters = {action: 0 for action in self.target_actions}

        self.audio_history = []  # 已识别的音频历史记录

        self.init_ui()
        self.bind_logic()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ================= 左侧：控制面板 =================
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(290)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setSpacing(12)

        title = QLabel("系统控制面板")
        title.setObjectName("AppTitle")
        title.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(title)

        self.fps_label = QLabel("实时 FPS: 0.0")
        self.fps_label.setObjectName("FpsLabel")
        self.fps_label.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(self.fps_label)

        cam_group = QGroupBox("输入源与参数")
        cam_layout = QVBoxLayout(cam_group)
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["摄像头 0 (默认)", "摄像头 1 (外接)"])
        cam_layout.addWidget(QLabel("选择摄像头:"))
        cam_layout.addWidget(self.cam_combo)

        cam_layout.addWidget(QLabel("动作置信度阈值:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(20, 90)
        self.conf_slider.setValue(45)
        cam_layout.addWidget(self.conf_slider)
        side_layout.addWidget(cam_group)

        self.status_label = QLabel("系统状态: 等待启动")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)
        side_layout.addWidget(self.status_label)

        side_layout.addStretch()

        # --- 全新设计的竞赛化按钮布局 ---
        self.btn_run = QPushButton("▶ 1. 启动系统传感器")
        self.btn_run.setObjectName("BtnStart")
        self.btn_run.setFixedHeight(45)
        side_layout.addWidget(self.btn_run)

        self.btn_start_action = QPushButton("🎯 2. 开始动作考核")
        self.btn_start_action.setObjectName("BtnAction")
        self.btn_start_action.setFixedHeight(45)
        self.btn_start_action.setEnabled(False)  # 未开机前禁用
        side_layout.addWidget(self.btn_start_action)

        self.btn_reset = QPushButton("↺ 重置所有记分板")
        self.btn_reset.setObjectName("BtnReset")
        side_layout.addWidget(self.btn_reset)

        self.btn_stop = QPushButton("⏹ 停止运行")
        self.btn_stop.setObjectName("BtnStop")
        side_layout.addWidget(self.btn_stop)

        # ================= 右侧：画面与结果 =================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)

        self.video_label = QLabel("无视频信号")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 8px; font-size: 16px; color: #666;")
        self.video_label.setMinimumHeight(440)
        right_layout.addWidget(self.video_label, 6)

        # 底部记分板
        score_board_layout = QHBoxLayout()

        # 1. 姿态任务面板
        vision_group = QGroupBox("任务一: 姿态识别 (40分)")
        vision_layout = QVBoxLayout(vision_group)

        self.stat_label = QLabel("识别成功统计: 0 / 4  (当前未开启考核)")
        self.stat_label.setObjectName("StatLabel")
        vision_layout.addWidget(self.stat_label)

        action_grid = QGridLayout()
        self.action_ui_elements = {}
        for i, action in enumerate(self.target_actions):
            lbl = QLabel(f"等待识别 | {action}")
            lbl.setProperty("class", "ScoreItem ScoreItemReady")
            lbl.setAlignment(Qt.AlignCenter)
            self.action_ui_elements[action] = lbl
            action_grid.addWidget(lbl, i // 2, i % 2)
        vision_layout.addLayout(action_grid)

        # 2. 音频任务面板 (分离了实时分布和历史记录)
        audio_group = QGroupBox("任务二: 音乐匹配 (30分)")
        audio_layout = QVBoxLayout(audio_group)

        lbl_live = QLabel("▶ 实时概率分布 (15秒窗口):")
        lbl_live.setStyleSheet("color: #ccc; font-weight: bold;")
        audio_layout.addWidget(lbl_live)

        self.audio_ui_elements = []
        for i in range(5):  # 这里压缩了间距，为下面历史记录腾出空间
            lbl = QLabel(f"{i + 1}. 暂无结果")
            lbl.setProperty("class", "ScoreItem ScoreItemReady")
            lbl.setContentsMargins(0, 0, 0, 0)
            self.audio_ui_elements.append(lbl)
            audio_layout.addWidget(lbl)

        lbl_hist = QLabel("💾 已识别记录:")
        lbl_hist.setStyleSheet("color: #ccc; font-weight: bold; margin-top: 5px;")
        audio_layout.addWidget(lbl_hist)

        self.audio_history_list = QListWidget()
        audio_layout.addWidget(self.audio_history_list)

        score_board_layout.addWidget(vision_group, 5)
        score_board_layout.addWidget(audio_group, 5)
        right_layout.addLayout(score_board_layout, 4)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(right_panel)

    def bind_logic(self):
        self.btn_run.clicked.connect(self.start_process)
        self.btn_stop.clicked.connect(self.stop_process)
        self.btn_reset.clicked.connect(self.reset_results)
        self.btn_start_action.clicked.connect(self.toggle_action_eval)

        self.vision_thread.frame_signal.connect(self.update_frame)
        self.vision_thread.result_signal.connect(self.update_vision_score)
        self.vision_thread.msg_signal.connect(self.update_status)

        self.audio_thread.msg_signal.connect(self.update_status)
        self.audio_thread.result_signal.connect(self.update_audio_score)

    def start_process(self):
        self.reset_results()
        self.fps_frames = 0
        self.fps_start_time = time.time()

        self.vision_thread.set_camera(self.cam_combo.currentIndex())
        self.vision_thread.start()
        self.audio_thread.start()

        self.btn_start_action.setEnabled(True)  # 系统启动后，允许开启考核
        self.update_status("系统传感器已启动，随时可以开始考核。")

    def toggle_action_eval(self):
        """控制是否正式开始抓取动作算分"""
        self.is_action_evaluating = not self.is_action_evaluating
        if self.is_action_evaluating:
            self.btn_start_action.setText("⏸ 暂停动作考核")
            self.btn_start_action.setStyleSheet("background-color: #e67e22; color: #fff;")
            self.stat_label.setText(f"识别成功统计: {self.success_count} / 4  (🔴 正在考核中)")
            self.update_status("考核已开启，请做出指定动作！")
        else:
            self.btn_start_action.setText("🎯 2. 开始动作考核")
            self.btn_start_action.setStyleSheet("background-color: #2980b9; color: #fff;")
            self.stat_label.setText(f"识别成功统计: {self.success_count} / 4  (⏸ 已暂停)")
            self.update_status("考核已暂停。")

    def stop_process(self):
        self.vision_thread.stop()
        self.audio_thread.stop()
        self.video_label.clear()
        self.video_label.setText("无视频信号")
        self.btn_start_action.setEnabled(False)
        self.is_action_evaluating = False
        self.btn_start_action.setText("🎯 2. 开始动作考核")
        self.btn_start_action.setStyleSheet("")
        self.update_status("系统已停止")
        self.fps_label.setText("实时 FPS: 0.0")

    def reset_results(self):
        """全面重置：应对裁判要求的两次机会"""
        self.success_count = 0
        self.is_action_evaluating = False
        self.btn_start_action.setText("🎯 2. 开始动作考核")
        self.btn_start_action.setStyleSheet("")

        self.stat_label.setText("识别成功统计: 0 / 4  (等待开始)")

        # 【修改】除了锁定状态，防抖计数器也要一并清零
        self.locked_actions = {action: False for action in self.target_actions}
        self.action_counters = {action: 0 for action in self.target_actions}

        for action, lbl in self.action_ui_elements.items():
            lbl.setText(f"等待识别 | {action}")
            lbl.setStyleSheet("")

        for i, lbl in enumerate(self.audio_ui_elements):
            lbl.setText(f"{i + 1}. 暂无结果")
            lbl.setStyleSheet("")

        # 清空音频历史
        self.audio_history.clear()
        self.audio_history_list.clear()

    def update_frame(self, qt_img):
        pix = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

        self.fps_frames += 1
        if self.fps_frames % 10 == 0:
            current_time = time.time()
            fps = 10.0 / (current_time - self.fps_start_time)
            self.fps_label.setText(f"实时 FPS: {fps:.1f}")
            self.fps_start_time = current_time

    def update_vision_score(self, data_list):
        # 仅当点击了“开始动作考核”后，才进行动作判断和锁定
        if not self.is_action_evaluating:
            return

        threshold = self.conf_slider.value() / 100.0

        # 【新增】用来记录当前帧识别到了哪些目标动作
        detected_this_frame = set()

        for data in data_list:
            if len(data) == 4:
                _, _, conf_str, action_name = data
                clean_action = action_name.replace("【", "").replace("】", "")
                conf = float(conf_str)

                if clean_action in self.target_actions and conf > threshold:
                    detected_this_frame.add(clean_action)  # 记录本帧看到的动作

                    if not self.locked_actions[clean_action]:
                        self.action_counters[clean_action] += 1  # 连续检测到，计数器+1

                        # 【核心防抖】必须连续稳定保持该动作 5 帧以上，才算真正考核通过！
                        if self.action_counters[clean_action] >= 5:
                            self.locked_actions[clean_action] = True
                            self.success_count += 1
                            self.stat_label.setText(f"识别成功统计: {self.success_count} / 4  (🔴 正在考核中)")

                            lbl = self.action_ui_elements[clean_action]
                            lbl.setText(f"已识别 | {clean_action} (Conf: {conf:.2f})")
                            lbl.setStyleSheet("color: #2ecc71; border: 1px solid #2ecc71; font-weight: bold;")

        # 【新增】如果这帧没看到这个动作，意味着动作中断或变形了，计数器必须清零
        for action in self.target_actions:
            if action not in detected_this_frame:
                self.action_counters[action] = 0

    def update_audio_score(self, prob_dist):
        # 如果没有听到有效音乐，恢复待机状态
        if not prob_dist:
            for i in range(5):
                self.audio_ui_elements[i].setText(f"{i + 1}. 暂无结果")
                self.audio_ui_elements[i].setStyleSheet("color: #aaaaaa;")
            return

        # 只要听到了音乐，强制渲染出前 5 名的所有概率
        for i in range(5):
            lbl = self.audio_ui_elements[i]
            if i < len(prob_dist):
                song_name, prob = prob_dist[i]

                # 哪怕是 0.0%，也强制显示出来，完美契合赛题的“分布”要求
                lbl.setText(f"{i + 1}. {song_name} - {prob:.1f}%")

                if i == 0 and prob > 50:
                    lbl.setStyleSheet("color: #2ecc71; border: 1px solid #2ecc71; font-weight: bold;")

                    # 自动存入历史记录
                    import time
                    if song_name not in self.audio_history:
                        self.audio_history.append(song_name)
                        time_str = time.strftime('%H:%M:%S')
                        self.audio_history_list.addItem(f"[{time_str}] 成功识别: {song_name}")
                else:
                    # 其他落选歌曲正常显示白色字体
                    lbl.setStyleSheet("color: #e0e0e0;")

    def update_status(self, msg):
        self.status_label.setText(f"系统状态: {msg}")

    def closeEvent(self, event):
        self.vision_thread.stop()
        self.audio_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())