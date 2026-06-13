#!/usr/bin/env bash
# 下载 Qwen3-ASR 0.6B 模型文件
# 建议在网络良好的环境下运行（如使用代理）
set -ex

MODEL_DIR="${1:-/tmp/sherpa-onnx-qwen3-asr-0.6b-int8-2026-03-25}"
MODEL_ARCHIVE="sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2"
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_ARCHIVE}"

# 也可使用 GitHub API 绕过慢速重定向:
# MODEL_ASSET_ID=390698077
# curl -L -H "Accept: application/octet-stream" \
#   "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/assets/${MODEL_ASSET_ID}" \
#   -o "${MODEL_ARCHIVE}"

if [ -f "$MODEL_ARCHIVE" ]; then
    echo "Archive already downloaded: $MODEL_ARCHIVE"
else
    echo "Downloading model archive (~838MB)..."
    if command -v wget &> /dev/null; then
        wget -c "$MODEL_URL"
    else
        curl -L -C - -o "$MODEL_ARCHIVE" "$MODEL_URL"
    fi
fi

echo "Extracting to $MODEL_DIR..."
rm -rf "$MODEL_DIR"
mkdir -p "$MODEL_DIR"
tar xvf "$MODEL_ARCHIVE" -C "$(dirname "$MODEL_DIR")"

echo "Model extracted to: $MODEL_DIR"
ls -lh "$MODEL_DIR"

echo ""
echo "=== 推送到 Android 设备 ==="
echo "adb push '$MODEL_DIR' /sdcard/sherpa-onnx/"
echo ""
echo "或者在 APP 中将模型路径改为: $MODEL_DIR"
