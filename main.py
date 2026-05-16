# main.py
import sys
from PySide6.QtWidgets import QApplication

# 导入我们模块化分离出来的 UI 类
from ui import MainWindow


def main():
    # 实例化 Qt 应用
    app = QApplication(sys.argv)

    # 实例化主窗口并显示
    window = MainWindow()
    window.show()

    # 进入应用主循环
    sys.exit(app.exec())


if __name__ == '__main__':
    main()