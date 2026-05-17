import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 定义和采集脚本完全一致的标签映射，用于打印结果
ACTION_MAP = {
    0: "大字站",
    1: "弓箭步",
    2: "举双手",
    3: "蹲下",
    4: "其他_正常走动弯腰"
}

CSV_FILE = "pose_data_optimized.csv"
MODEL_FILE = "pose_classifier.pkl"


def main():
    # 1. 检查数据集是否存在
    if not os.path.exists(CSV_FILE):
        print(f"❌ 错误：找不到 {CSV_FILE}！请先运行 collect_data.py 采集一些姿态数据。")
        return

    print("📅 正在加载数据集...")
    df = pd.read_csv(CSV_FILE)

    # 2. 拆分特征(X)和标签(y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print(f"📊 数据集加载成功！总样本数: {len(df)} 行")
    for label, name in ACTION_MAP.items():
        count = np.sum(y == label)
        print(f"  - 【{name}】样本数: {count} 帧")

    if len(df) < 100:
        print("⚠️ 警告：数据量太少，模型可能无法准确泛化，建议每个动作至少录制 200 帧以上。")

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🤖 正在使用随机森林（Random Forest）算法训练分类器...")
    # 使用随机森林分类器，鲁棒性极强，防误触效果最好
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("✅ 模型训练完成！")

    # 4. 评估模型准确率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 核心评估指标 —— 模型在测试集上的准确率: {accuracy * 100:.2f}%")

    print("\n详细分类报告:")
    # ================= 修复区域 =================
    # 动态获取当前数据集里实际存在的标签（比如只录了0,1,2,3，就把这四个挑出来）
    unique_labels = np.unique(y)
    actual_target_names = [ACTION_MAP[i] for i in unique_labels]

    # 将实际存在的标签和名字传给报告函数，完美避免 ValueError
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=actual_target_names))
    # ============================================

    # 5. 保存模型为 .pkl 文件
    joblib.dump(model, MODEL_FILE)
    print(f"💾 核心大脑已导出！模型文件成功保存至: {MODEL_FILE}")


if __name__ == '__main__':
    main()