# audio.py
import os
import pickle
import numpy as np
import librosa
import sounddevice as sd
import hashlib
from scipy.ndimage import maximum_filter
from PySide6.QtCore import QThread, Signal


class ShazamAlgorithm:
    """沙赞音频指纹算法"""

    def __init__(self):
        self.song_database = {}
        self.known_songs = []

    def extract_peaks(self, audio_data, sr=22050):
        stft = np.abs(librosa.stft(audio_data, n_fft=2048, hop_length=512))
        stft_db = librosa.amplitude_to_db(stft, ref=np.max)

        filter_size = 15
        local_max = maximum_filter(stft_db, size=filter_size) == stft_db

        background_threshold = -40
        peaks = (stft_db > background_threshold) & local_max
        freq_idx, time_idx = np.where(peaks)
        return list(zip(freq_idx, time_idx))

    def generate_hashes(self, peaks):
        peaks.sort(key=lambda x: x[1])
        hashes = []
        for i in range(len(peaks)):
            anchor = peaks[i]
            for j in range(1, min(5, len(peaks) - i)):
                target = peaks[i + j]
                time_diff = target[1] - anchor[1]
                if 0 < time_diff < 100:
                    hash_str = f"{anchor[0]}|{target[0]}|{time_diff}"
                    hash_val = hashlib.sha1(hash_str.encode('utf-8')).hexdigest()[:12]
                    hashes.append((hash_val, anchor[1]))
        return hashes

    def register_song(self, file_path, song_name):
        """【离线工具集】用于赛前将音乐提取为特征"""
        print(f"🎵 正在提取特征入库: {song_name}...")
        try:
            y, sr = librosa.load(file_path, sr=22050, mono=True)
            peaks = self.extract_peaks(y, sr)
            hashes = self.generate_hashes(peaks)
            for h, t in hashes:
                self.song_database[h] = (song_name, t)
            if song_name not in self.known_songs:
                self.known_songs.append(song_name)
            print(f"✅ {song_name} 提取成功，生成指纹 {len(hashes)} 个")
        except Exception as e:
            print(f"❌ {song_name} 提取失败，检查文件: {e}")

    def save_database(self, filename="fingerprints.pkl"):
        """【离线工具集】固化特征，避免赛场启动卡顿"""
        with open(filename, 'wb') as f:
            pickle.dump((self.song_database, self.known_songs), f)
        print(f"💾 指纹库已成功固化到本地: {filename}")

    def load_database(self, filename="fingerprints.pkl"):
        """【实战引擎】启动时直接秒级加载内存库"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.song_database, self.known_songs = pickle.load(f)
            print(f"⚡ 成功加载离线指纹库，包含曲目: {self.known_songs}")
            return True
        return False

    def identify_audio(self, audio_data, sr=22050):
        """【核心引擎】时间偏移一致性抗噪 + 完整概率分布输出"""
        peaks = self.extract_peaks(audio_data, sr)
        hashes = self.generate_hashes(peaks)

        matches = {song: {} for song in self.known_songs}

        for h, t_sample in hashes:
            if h in self.song_database:
                song_name, t_origin = self.song_database[h]
                offset = t_origin - t_sample
                offset_bin = round(offset / 2.0)
                matches[song_name][offset_bin] = matches[song_name].get(offset_bin, 0) + 1

        song_scores = {}
        total_score = 0
        valid_signal = False  # 新增：用来判断是否真的有音乐在放

        for song_name, offset_counts in matches.items():
            if not offset_counts:
                song_scores[song_name] = 0
                continue

            max_coherent_matches = max(offset_counts.values())

            # 不再强制归零，保留底噪碰撞分数，以便算出 0.x% 的真实分布
            song_scores[song_name] = max_coherent_matches
            total_score += max_coherent_matches

            # 只要有一首歌的对齐特征大于 5，说明当前环境中确实在放音乐，而不是纯纯的风扇噪音
            if max_coherent_matches >= 5:
                valid_signal = True

        # 如果全是白噪音（没有任何一首歌匹配度达标），直接返回空，让UI显示"暂无结果"
        if not valid_signal:
            return []

        # 将保留了底噪的绝对得分转化为【概率分布百分比】
        probability_distribution = []
        for song, score in song_scores.items():
            prob = (score / total_score * 100) if total_score > 0 else 0
            probability_distribution.append((song, prob))

        return sorted(probability_distribution, key=lambda x: x[1], reverse=True)


class AudioThread(QThread):
    """麦克风动态智能拾音线程 (最长15秒，识别到即刻停止)"""
    result_signal = Signal(list)
    msg_signal = Signal(str)
    finished_signal = Signal()  # 新增：告诉 UI 录音已结束，可以恢复按钮

    def __init__(self, shazam_engine):
        super().__init__()
        self.engine = shazam_engine
        self.sample_rate = 22050
        self.chunk_duration = 3  # 采样切片：每次监听 3 秒
        self.max_duration = 15  # 赛题最大限制：15 秒
        self.running = False

    def run(self):
        self.running = True
        try:
            self.msg_signal.emit("🎤 正在智能监听音频 (最长15秒，识别即停)...")
            cumulative_audio = np.array([], dtype='float32')
            max_iters = int(self.max_duration / self.chunk_duration)

            match_found = False

            for i in range(max_iters):
                if not self.running:
                    break

                # 采集一个 3 秒的音频切片
                recording = sd.rec(int(self.chunk_duration * self.sample_rate),
                                   samplerate=self.sample_rate,
                                   channels=1, dtype='float32')
                sd.wait()

                if not self.running:
                    break

                chunk_data = recording.flatten()
                # 将新切片拼接到累积的音频带中
                cumulative_audio = np.concatenate((cumulative_audio, chunk_data))

                # 立即用沙赞算法分析当前累积的音频
                matches = self.engine.identify_audio(cumulative_audio, self.sample_rate)

                # 如果有音乐特征，且排名第一的匹配概率达到 50% 以上，触发提早熔断锁分！
                if matches and matches[0][1] >= 50.0:
                    top_song = matches[0][0]
                    dist_str = " | ".join([f"{song}: {prob:.1f}%" for song, prob in matches if prob > 0])
                    self.msg_signal.emit(f"🎵 锁定: {top_song} (用时 {(i + 1) * self.chunk_duration}秒)")
                    self.result_signal.emit(matches)
                    match_found = True
                    break
                else:
                    self.msg_signal.emit(f"⏳ 正在分析旋律中... (已听 {(i + 1) * self.chunk_duration}/15秒)")
                    # 发送中间态的低置信度数据供UI实时显示概率跳动，但不会触发记分板
                    self.result_signal.emit(matches)

            # 如果 15 秒跑满了还没过 50% 的置信度
            if not match_found and self.running:
                if cumulative_audio.size > 0:
                    final_matches = self.engine.identify_audio(cumulative_audio, self.sample_rate)
                    if final_matches and final_matches[0][1] > 0:
                        self.msg_signal.emit("⚠️ 15秒用尽，给出最高概率推测。")
                        self.result_signal.emit(final_matches)
                    else:
                        self.msg_signal.emit("🔇 15秒内未识别到有效音乐，请重试。")
                        self.result_signal.emit([])
                else:
                    self.msg_signal.emit("🔇 收音异常，未获取到有效数据。")
                    self.result_signal.emit([])

        except Exception as e:
            self.msg_signal.emit(f"❌ 音频模块报错: {str(e)}")
            self.result_signal.emit([])

        self.running = False
        self.finished_signal.emit()  # 释放 UI 按钮

    def stop(self):
        self.running = False
        import sounddevice as sd
        sd.stop()
        self.quit()
        self.wait()