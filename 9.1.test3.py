import cv2
import numpy as np
from deepface import DeepFace
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class FixedCameraEmotionAnalyzer:
    def __init__(self):
        self.emotion_stats = defaultdict(int)
        self.frame_count = 0
        self.start_time = time.time()
        self.emotion_history = []
        self.cap = None

    def initialize_camera(self):
        """尝试不同的摄像头后端"""
        backends = [
            cv2.CAP_DSHOW,  # DirectShow (Windows)
            cv2.CAP_MSMF,  # Microsoft Media Foundation
            cv2.CAP_ANY,  # 自动选择
            cv2.CAP_V4L2  # Linux Video4Linux2
        ]

        for backend in backends:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    print(f"成功使用后端: {backend}")
                    # 设置摄像头参数
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    return cap
                cap.release()
            except:
                continue

        # 如果所有后端都失败，尝试默认
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("使用默认后端")
                return cap
        except:
            pass

        return None

    def analyze_frame(self, frame):
        try:
            # 使用DeepFace进行分析
            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True  # 减少输出
            )

            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                dominant_emotion = result['dominant_emotion']
                emotion_scores = result['emotion']

                # 更新统计
                self.emotion_stats[dominant_emotion] += 1
                self.frame_count += 1
                self.emotion_history.append({
                    'emotion': dominant_emotion,
                    'scores': emotion_scores,
                    'timestamp': time.time()
                })

                # 绘制结果
                for face in results:
                    if 'region' in face:
                        region = face['region']
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"{dominant_emotion} ({max(emotion_scores.values()):.2f})"
                        cv2.putText(frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                return frame, results

        except Exception as e:
            print(f"分析错误: {e}")

        return frame, None

    def generate_report(self):
        if self.frame_count == 0:
            return "没有检测到情绪数据", None

        duration = time.time() - self.start_time
        total_frames = self.frame_count

        emotion_percentages = {
            emotion: (count / total_frames) * 100
            for emotion, count in self.emotion_stats.items()
        }

        report = f"微表情分析报告\n"
        report += f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"分析时长: {duration:.2f} 秒\n"
        report += f"总帧数: {total_frames}\n\n"
        report += "情绪统计:\n"

        for emotion, percentage in emotion_percentages.items():
            report += f"{emotion}: {percentage:.2f}%\n"

        chart_base64 = self._create_chart(emotion_percentages)

        return report, chart_base64

    def _create_chart(self, emotion_percentages):
        plt.figure(figsize=(10, 6))
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
        plt.pie(emotion_percentages.values(), labels=emotion_percentages.keys(),
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.axis('equal')
        plt.title('情绪分布')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        return img_str


def main():
    print("初始化摄像头...")
    analyzer = FixedCameraEmotionAnalyzer()

    # 初始化摄像头
    cap = analyzer.initialize_camera()
    if cap is None:
        print("错误: 无法打开摄像头")
        print("请检查:")
        print("1. 摄像头是否连接")
        print("2. 其他程序是否占用摄像头")
        print("3. 摄像头权限是否开启")
        return

    print("按 'q' 键退出并生成报告")
    print("按 'r' 键重置统计")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取帧失败，尝试重新初始化摄像头...")
                cap.release()
                cap = analyzer.initialize_camera()
                if cap is None:
                    break
                continue

            # 水平翻转图像（镜像效果）
            frame = cv2.flip(frame, 1)

            # 分析帧
            analyzed_frame, results = analyzer.analyze_frame(frame)

            # 显示实时统计
            if analyzer.frame_count > 0:
                stats_text = f"帧数: {analyzer.frame_count}"
                cv2.putText(analyzed_frame, stats_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 显示结果
            cv2.imshow('微表情识别 - 按q退出, 按r重置', analyzed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                analyzer = FixedCameraEmotionAnalyzer()
                print("统计已重置")

    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()

        # 生成报告
        report, chart_base64 = analyzer.generate_report()

        print("\n" + "=" * 50)
        print(report)
        print("=" * 50)

        if chart_base64:
            with open('emotion_report.png', 'wb') as f:
                f.write(base64.b64decode(chart_base64))
            print("图表已保存为 emotion_report.png")


if __name__ == "__main__":
    main()