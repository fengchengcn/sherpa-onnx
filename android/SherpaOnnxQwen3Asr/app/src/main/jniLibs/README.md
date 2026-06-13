# JNI Libraries

This directory should contain the sherpa-onnx JNI library and ONNX Runtime.

## Required files per ABI

```
jniLibs/
├── arm64-v8a/
│   ├── libsherpa-onnx-jni.so
│   └── libonnxruntime.so
├── armeabi-v7a/
│   ├── libsherpa-onnx-jni.so
│   └── libonnxruntime.so
├── x86_64/
│   ├── libsherpa-onnx-jni.so
│   └── libonnxruntime.so
└── x86/
    ├── libsherpa-onnx-jni.so
    └── libonnxruntime.so
```

## Build libsherpa-onnx-jni.so from source

From the project root, run:

```bash
# For ARM64 (most modern Android phones)
./build-android-arm64-v8a.sh

# For ARMv7 (older 32-bit devices)
./build-android-armv7-eabi.sh

# For x86_64 (emulator)
./build-android-x86-64.sh

# For x86 (emulator)
./build-android-x86.sh
```

Then copy `libsherpa-onnx-jni.so` to the corresponding ABI directory:

```bash
cp build-android-arm64-v8a/lib/libsherpa-onnx-jni.so jniLibs/arm64-v8a/
```

## ONNX Runtime

Download the ONNX Runtime Android package from:
https://github.com/microsoft/onnxruntime/releases

Extract and copy `libonnxruntime.so` from `onnxruntime-android-<version>.aar` (rename to `.zip` and unzip) or from the JNI package distribution.

## Pre-built libraries

Pre-built libraries are also available from the [sherpa-onnx releases page](https://github.com/k2-fsa/sherpa-onnx/releases).
