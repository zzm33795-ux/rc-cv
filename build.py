# build_db.py
import os
from audio import ShazamAlgorithm


def main():
    print(" 开始构建离线音频指纹库...")

    # 初始化我们重构好的核心引擎
    engine = ShazamAlgorithm()

    # 这里是你 music 文件夹里的文件名，以及对应的赛题歌曲名
    # 如果你的 mp3 名字叫其他名字，把前面的路径改掉即可
    songs = [
        ("music/1（Shed A Light）.mp3", "指定曲目一"),
        ("music/2（Stronger）.mp3", "指定曲目二"),
        ("music/3（新的心跳）.mp3", "指定曲目三"),
        ("music/4（追梦赤子心）.mp3", "指定曲目四"),
        ("music/5（By Your Side）.mp3", "指定曲目五")
    ]

    if not os.path.exists("music"):
        print("❌ 没找到 music 文件夹！请先建一个 music 文件夹并把 mp3 放进去。")
        return

    # 循环遍历并提取特征
    success_count = 0
    for file_path, song_name in songs:
        if os.path.exists(file_path):
            # register_song 内部会自动用 librosa 读取 mp3，转成 22050Hz 单声道并生成指纹
            engine.register_song(file_path, song_name)
            success_count += 1
        else:
            print(f"⚠️ 找不到音频文件: {file_path}，请检查名字拼写！")

    # 只要有成功提取的歌曲，就打包生成序列化文件
    if success_count > 0:
        engine.save_database("fingerprints.pkl")
        print("🎉 指纹库 fingerprints.pkl 构建完毕！你可以去启动 main.py 跑全系统了！")
    else:
        print("❌ 没有提取到任何歌曲，未生成指纹库。")


if __name__ == "__main__":
    main()