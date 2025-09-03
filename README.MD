# 微表情识别摄像头工具（Emotion Recognition Camera）

![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.79-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

实时面部情绪识别工具，基于 OpenCV 摄像头捕捉和 DeepFace 情绪分析，支持实时统计、结果可视化和报告生成。


## 📌 功能特点
- 🔍 **实时识别**：支持 7 种情绪（开心、悲伤、愤怒、惊讶、恐惧、厌恶、中性）。
- 📊 **数据统计**：自动记录情绪出现频率，生成占比报告。
- 🎨 **可视化**：摄像头画面实时标注面部框和情绪标签，退出后生成情绪分布饼图。
- 🔄 **灵活操作**：按 `r` 键重置统计，按 `q` 键退出并保存报告。
- 💻 **多系统兼容**：支持 Windows、macOS、Linux，自动适配摄像头后端。


## 🚀 快速开始

### 1. 环境准备
#### 方式1：直接运行（非技术人员，仅 Windows）
- 下载 [Release](https://github.com/你的GitHub用户名/Emotion-Recognition-Camera/releases) 中的 `Emotion-Recognition.exe`（后续可上传可执行文件到 Release，见步骤4）。
- 双击运行，首次启动会自动加载模型（需联网，约100MB）。

#### 方式2：源码运行（推荐，全系统）
- **依赖**：Python 3.8~3.10 + 摄像头（内置/外置）。
- **安装依赖库**：
  ```bash
  # 克隆仓库到本地（或直接下载 ZIP 压缩包）
  git clone https://github.com/你的GitHub用户名/Emotion-Recognition-Camera.git
  cd Emotion-Recognition-Camera

  # 安装依赖（建议创建虚拟环境，避免冲突）
  pip install -r requirements.txt