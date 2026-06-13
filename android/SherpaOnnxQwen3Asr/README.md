# SherpaOnnx Qwen3-ASR Demo

基于 sherpa-onnx 框架的 Qwen3-ASR 0.6B 模型 **Android 最小验证 Demo**。

## 功能

- 从设备文件系统加载 Qwen3-ASR 0.6B int8 量化模型 (~940MB)
- 麦克风录音 → 离线语音识别
- 显示识别结果和性能指标（RTF、耗时）

## 准备工作

### 1. 构建 JNI 库

```bash
# 在项目根目录
./build-android-arm64-v8a.sh

# 复制到 demo 的 jniLibs 目录
cp build-android-arm64-v8a/lib/libsherpa-onnx-jni.so \
   android/SherpaOnnxQwen3Asr/app/src/main/jniLibs/arm64-v8a/
```

### 2. 下载模型文件

从 HuggingFace 下载：
https://huggingface.co/csukuangfj/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25

```bash
# 推送到设备（推荐 /data/local/tmp/，全局可读）
adb push sherpa-onnx-qwen3-asr-0.6b-int8-2026-03-25 /data/local/tmp/
```

### 3. 编译安装

```bash
cd android/SherpaOnnxQwen3Asr
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 使用方法

1. 打开 APP，确认模型路径（默认为 `/data/local/tmp/qwen3-asr-0.6b-int8-2026-03-25`）
2. 点击 **"加载模型"**，等待模型加载完成（10-30 秒）
3. 点击 **"开始录音"**，说话
4. 点击 **"停止录音"**，自动开始识别
5. 查看识别结果和 RTF 等性能指标

## 性能

| 指标 | 预期值 (Mate 40e) |
|------|-------------------|
| 模型大小 | ~940MB |
| 模型加载时间 | 10-30s |
| RTF (5s 音频) | ~0.5-1.0x |
| 峰值内存 | ~1.1-1.3GB |
