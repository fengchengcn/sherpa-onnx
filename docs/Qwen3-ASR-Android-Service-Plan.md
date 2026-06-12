# Qwen3-ASR-0.6B 通用 ASR 服务 —— Android 10 实现方案

> 方案制定日期: 2026-06-12
> 目标平台: Android 10 (API 29) +
> 框架: sherpa-onnx v1.13.2
> 模型: Qwen3-ASR-0.6B (int8 量化)

---

## 一、背景与现状

### 1.1 目标

在 Android 10 手机上，用 sherpa-onnx 框架运行 Qwen3-ASR-0.6B 模型，实现一个**通用 ASR 服务**，供多个第三方 APP（输入法、录音机、会议记录等）调用。

### 1.2 现状确认

经过对代码库的全面分析，确认以下事实:

**✅ sherpa-onnx 已完整支持 Qwen3-ASR-0.6B**:

| 层级 | 状态 | 关键文件 |
|------|------|---------|
| C++ 核心 | 完整实现 | `sherpa-onnx/csrc/offline-qwen3-asr-model.cc` (~860行), `offline-recognizer-qwen3-asr-impl.cc` (~1124行) |
| JNI 桥接 | 完整支持 | `sherpa-onnx/jni/offline-recognizer.cc` (lines 349-376) |
| Kotlin API | 完整封装 | `sherpa-onnx/kotlin-api/OfflineRecognizer.kt` (`OfflineQwen3AsrModelConfig`, model index 61) |
| 预置模型 | 已发布 | `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` |

**Qwen3-ASR-0.6B 模型架构**:

```
音频输入 (16kHz mono)
    ↓
Conv Frontend (3层 Conv2D, ~42MB)  →  audio tokens
    ↓
Audio Encoder (18层 Transformer, ~310M params, ~174MB int8)  →  hidden representations
    ↓
LLM Decoder (28层 Qwen3, ~470M params, ~722MB int8, KV-cache 自回归)
    ↓
文本输出 (最多 128 tokens, temperature/top-p 采样)
```

- 总参数量: ~782M
- int8 量化后模型文件: conv_frontend (42MB) + encoder (174MB) + decoder (722MB) + tokenizer/
- 峰值内存: ~1.1-1.3GB
- 推理速度 (PC): RTF ~0.17x
- 移动端预估: 旗舰机 ~3-5s/5s音频, 中端机 ~8-15s, 低端机 ~15-30s

**⚠️ 现有 Android 架构无法跨 APP 调用**:

- 所有 16 个 Demo APP 都是单进程内嵌式
- `SherpaOnnxJavaDemo/SpeechSherpaRecognitionService` — `onBind()` 返回 null, `exported=false`
- 代码库中 **零个 AIDL 文件**, 无任何 IPC 机制
- 需要从零构建跨 APP 服务架构

---

## 二、整体架构设计

> **目标设备**: Huawei Mate 40e (8GB RAM, 麒麟 900E, Android 10 / HarmonyOS)
> **MVP 策略**: 纯 AIDL + 短句一次性识别；HTTP/WebSocket/VAD 流式作为 v2 规划，优先保证稳定性与可落地性

```
                   ┌── 第三方 APP ──────────────────────────────┐
                   │                                              │
  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
  │ APP A    │  │ APP B    │  │ APP C    │                      │
  │ (输入法)  │  │ (录音机)  │  │ (会议)   │                      │
  │          │  │          │  │          │                      │
  │ Client   │  │ Client   │  │ Client   │                      │
  │ SDK .aar │  │ SDK .aar │  │ SDK .aar │                      │
  └────┬─────┘  └────┬─────┘  └────┬─────┘                      │
       │              │              │                            │
       │          AIDL IPC (Binder) │                            │
       │      ParcelFileDescriptor  │                            │
       └──────────────┼─────────────┘                            │
                      │                                          │
└─────────────────────┼──────────────────────────────────────────┘
                      │
┌─────────────────────┼──────────────────────────────────────────┐
│                     ▼                                          │
│   SherpaOnnxAsrService (Foreground Service, :asr_service)     │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    AIDL Stub                              │ │
│  │         ISherpaAsrService.aidl                            │ │
│  │         AudioDataReader (ParcelFileDescriptor → FloatArray)│ │
│  └────────────────────────┬─────────────────────────────────┘ │
│                           │ 所有请求入队                       │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              AsrRequestProcessor                          │ │
│  │   PriorityBlockingQueue                                   │ │
│  │   短音频(<10s)优先 + 同优先级FIFO                           │ │
│  │   单线程消费 (模型不支持并行)                               │ │
│  └────────────────────────┬─────────────────────────────────┘ │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                   ModelManager                            │ │
│  │              OfflineRecognizer                            │ │
│  │         Qwen3-ASR 0.6B int8 (~1.1GB)                     │ │
│  │         空闲超时自动卸载 (默认 60s)                         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │   Model Downloader (国内 CDN/OSS, 断点续传, ~1.9GB tar.bz2)│ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| IPC 机制 | **纯 AIDL** (MVP) | 原生 APP 间通信核心路径；v2 加本地 HTTP/WS 支持自有 WebView (外部 H5 不可用) |
| 音频传输 | **ParcelFileDescriptor** | 避免 Binder 1MB 事务限制；支持任意长度音频 |
| 进程模型 | **独立进程** (`:asr_service`) | 1.1GB 模型内存隔离；客户端 OOM 不影响服务；LMK 独立击杀域 |
| 服务类型 | **前台 Service** | Android 10+ 长时运行服务必须前台化；`onCreate()` 立即 `startForeground()` |
| 推理调度 | **统一队列** (PriorityBlockingQueue) | 所有入口 (AIDL submit) 强制入队；短音频优先；单线程消费防止锁竞争 |
| 模型生命周期 | **默认空闲超时卸载 (60s)** | 释放 1.1GB 给其他 APP；常驻模式仅作为用户手动开启的高级选项 |
| 模型分发 | **国内 CDN/OSS 下载** | GitHub Releases 国内不稳定；断点续传 + SHA256 校验 |
| 客户端集成 | **轻量 .aar SDK** | 屏蔽 AIDL/Binder/DeathRecipient 细节；处理华为 ROM 后台限制 |
| 识别模式 (MVP) | **短句一次性提交** | MVP 仅短句；v2 通过客户端 VAD 自动断句实现流式体验（详见 §2.3.2） |

### MVP 范围与二期规划

| 功能 | MVP (v1) | 二期 (v2) | 说明 |
|------|----------|-----------|------|
| AIDL IPC | ✅ 实现 | 保持 | 原生 APP 间通信核心通道 |
| HTTP/WebSocket Server | ❌ 不做 | ✅ 实现 | v1 仅 AIDL；v2 加本地 HTTP/WS 支持自有 WebView 调用 |
| VAD 断句流式识别 | ❌ 不做 | ✅ 实现 | v1 仅短句一次性提交；v2 加 VAD 自动切分 + 逐句识别 |
| 默认常驻保活 | ❌ 不做 | 用户可选 | v1 默认 60s 空闲卸载；常驻模式作为设置中的高级选项 |
| 模型下载 (CDN) | ✅ 实现 | 保持 | 国内 CDN/OSS + 断点续传 |
| 短音频优先队列 | ✅ 实现 | 保持 | 防止长音频阻塞短请求 |

### Phase 2 扩展架构 (v2 规划，非 MVP)

```
v1 MVP (AIDL only):
  第三方 APP → [SDK .aar] → AIDL → [AsrRequestProcessor] → InferenceEngine

v2 扩展 (add HTTP/WS + VAD streaming):
                                      ┌─── LocalHttpServer (127.0.0.1:8765)
  第三方 APP → [SDK .aar] → AIDL ────┤
  自有 WebView → fetch/WS ───────────┤→ [AsrRequestProcessor] → InferenceEngine
                                      │        ↑
                                      └─── VAD 断句模块 ─┘
                                           (长音频自动切短句)
```

> v2 的 HTTP/WebSocket 详细设计见下文 §2.3。MVP 阶段不实现，但架构预留扩展点。

---

### 2.2 服务生命周期与保活机制（关键设计）

#### 为什么必须用前台 Service？

Android 有三种 Service，存活能力差异巨大:

| 类型 | 存活能力 | 系统可见性 | 被杀风险 | 适用场景 |
|------|---------|-----------|---------|---------|
| **前台 Service** | ⭐⭐⭐ 最高 | 通知栏常驻图标 | 极低，仅极端低内存时 | 音乐播放、导航、**ASR 引擎** |
| 后台 Service | ⭐⭐ 中 | 不可见 | Android 8+ 几分钟内被杀 | 已基本废弃 |
| 绑定 Service | ⭐ 随客户端 | 不可见 | 客户端解绑即销毁 | 同进程内通信 |

前台 Service 会在通知栏显示一个持久通知（类似音乐播放器的播放控件），Android 系统据此判断"用户知道这个进程在运行，不要杀它":

```
┌────────────────────────────────────┐
│ 🔴 Sherpa ASR 服务运行中            │
│ Qwen3-ASR-0.6B 模型已就绪           │
└────────────────────────────────────┘
```

#### 自动启动机制：客户端无需手动启动服务

这是整个方案最核心的用户体验设计——**第三方 APP 调用 ASR 能力时，完全不需要关心服务是否已经启动**:

```kotlin
// ========== 第三方 APP 的代码（例如输入法） ==========
val client = SherpaSpeechClient(context)

// 只需要一行 connect(), SDK 内部自动处理:
//   情况 A: ASR 服务已在运行 → 直接绑定，即刻可用
//   情况 B: ASR 服务未运行   → 先 startForegroundService() 推入前台，
//                              再 bindService() 获取 Binder
client.connect()

// 服务就绪后即可识别
client.recognize(audioData, 16000)
```

底层原理——SDK 内部双管齐下启动策略:

> **关键设计**: Android 10 要求前台 Service 必须在 `onCreate()` 后 **5 秒内** 调用 `startForeground()`。
> 如果仅通过 `bindService()` 隐式拉起，Service 的 `onCreate()` 中加载模型可能超过 5 秒，
> 系统会抛出 `ForegroundServiceDidNotStartInTimeException` 导致崩溃。
>
> 因此 SDK 的 `connect()` 采用 **先启动、后绑定** 的策略:

```kotlin
// client_sdk/src/.../ServiceConnectionManager.kt
fun connect() {
    val intent = Intent().setComponent(
        ComponentName("com.k2fsa.sherpa.onnx.asr.service",
                       "com.k2fsa.sherpa.onnx.asr.service.SherpaOnnxAsrService")
    )
    // 第 1 步: 先确保服务启动并在 5 秒内推入前台
    context.startForegroundService(intent)
    // 第 2 步: 再绑定获取 Binder（服务已在 onCreate 中调用了 startForeground）
    context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
}
```

完整时序流程:

```
第三方 APP 进程                     Android 系统                        ASR 服务进程
     │                                  │                                  │
     │── client.connect()              │                                  │
     │── startForegroundService() ─────→│                                  │
     │                                  │── 创建进程 ──────────────────────→│
     │                                  │                                  │── onCreate()
     │                                  │                                  │   ├ 立即 startForeground()
     │                                  │                                  │   │   (通知: "初始化中...")
     │                                  │                                  │   ├ 加载模型 (~10s)
     │                                  │                                  │   └ 更新通知: "模型已就绪 — AIDL 可接入"
     │                                  │                                  │
     │── bindService(intent) ──────────→│                                  │
     │                                  │ 检查: 服务已在运行                │
     │                                  │ (已通过 startForegroundService  │
     │                                  │  确保服务为前台进程)              │
     │                                  │                                  │
     │                        ┌─────────┴──────────┐                       │
     │                        │ 情况A: 模型已加载   │  情况B: 模型加载中    │
     │                        │ → 直接绑定         │  → 绑定成功，等待      │
     │                        │   (热启动, <200ms) │    onServiceReady()   │
     │                        └─────────┬──────────┘                       │
     │                                  │                                  │
     │                                  │ Binder IPC 连接建立              │
     │←── onServiceConnected() ────────│←───────────────────────────────── │
     │←── onServiceReady() (回调) ─────│←───────────────────────────────── │
     │                                  │                                  │
     │── client.recognize(audio) ──────→│─────────────────────────────────→│
     │                                  │                                  │
```

**关键时间指标**:

| 场景 | 延迟 | 说明 |
|------|------|------|
| 热启动 (服务已驻留) | < 200ms | 直接 Binder 绑定，无需加载模型 |
| 冷启动 (服务未运行) | 8-15 秒 | Mate 40e 实测：进程创建 + ONNX Runtime 加载 940MB 权重 + 预热推理 |

#### ⚠️ 华为/HarmonyOS 后台启动限制

华为系 ROM (EMUI/HarmonyOS) 对后台启动 Service 的限制比原生 Android 更严格。
`startForegroundService()` 在某些场景下仍可能被拦截（如 APP 在后台、省电模式等）。

**降级策略 — 主 APP + 第三方 APP 分工**:

```
                  ┌── 主 APP (ASR Service 宿主) ──┐
                  │  负责：startForegroundService  │
                  │  提供：ASR Service 本身        │
                  │  包含：设置界面、模型管理       │
                  └────────────┬───────────────────┘
                               │ AIDL (Binder IPC)
                  ┌────────────┼───────────────────┐
                  │            ▼                   │
                  │  第三方 APP (如输入法)          │
                  │  负责：bindService() 仅绑定     │
                  │  不负责启动服务                  │
                  └────────────────────────────────┘
```

```kotlin
// ServiceConnectionManager.kt — 华为 ROM 安全启动策略
fun connect() {
    val intent = Intent().setComponent(
        ComponentName("com.k2fsa.sherpa.onnx.asr.service",
                       "com.k2fsa.sherpa.onnx.asr.service.SherpaOnnxAsrService")
    )

    try {
        // 尝试 1: startForegroundService (前台 APP 大多数情况成功)
        context.startForegroundService(intent)
        context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
    } catch (e: SecurityException) {
        // 尝试 2: 华为 ROM 后台限制触发 — 仅 bind (服务可能已在运行)
        Log.w(TAG, "startForegroundService 被华为 ROM 拦截，尝试仅 bindService")
        try {
            val bound = context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
            if (!bound) {
                // 最终 fallback: 引导用户打开主 APP
                showFallbackDialog("请打开「Sherpa ASR」应用以启动语音识别服务")
            }
        } catch (e2: Exception) {
            showFallbackDialog("无法连接 ASR 服务，请确认已安装「Sherpa ASR」应用")
        }
    }
}
```

> **对第三方 APP 开发者的建议**: 在 SDK Quick Start 文档中明确说明:
> 1. 用户设备上需安装主 APK（Sherpa ASR Service）
> 2. 首次使用前建议用户手动打开一次主 APP（确保服务进程创建）
> 3. 之后第三方 APP 通过 `bindService()` 即可热启动（<200ms）

#### 长期驻留策略：两种模式可选

服务启动一次后，如何处理空闲期？提供两种策略:

**策略 A: 空闲超时（默认推荐）**

```
时间线:
──────────────────────────────────────────────────────────→
APP A 调用     APP A 解绑    空闲倒计时           APP C 调用
    │              │        60s...                 │
    ▼              ▼           │                   ▼
[服务运行] ──── [模型保持] ───[⏰]── [自动停止]    [重新冷启动]
                ▲                                  ▲
                0 秒延迟                          8-15 秒延迟
```

```kotlin
class SherpaOnnxAsrService : Service() {
    private val boundClients = AtomicInteger(0)
    private val shutdownHandler = Handler(Looper.getMainLooper())
    private val shutdownRunnable = Runnable {
        Log.i(TAG, "所有客户端已解绑超过 60 秒，自动停止服务并卸载模型")
        stopSelf()
    }

    override fun onBind(intent: Intent): IBinder {
        boundClients.incrementAndGet()
        shutdownHandler.removeCallbacks(shutdownRunnable)  // 有新客户端，取消倒计时
        return stub.asBinder()
    }

    override fun onUnbind(intent: Intent): Boolean {
        if (boundClients.decrementAndGet() == 0) {
            // 所有客户端都解绑了，启动 60 秒倒计时
            shutdownHandler.postDelayed(shutdownRunnable, 60_000)
        }
        return true  // 返回 true 允许后续 onRebind()
    }

    override fun onRebind(intent: Intent?) {
        boundClients.incrementAndGet()
        shutdownHandler.removeCallbacks(shutdownRunnable)  // 倒计时内有人重连
    }
}
```

优点: 省电、释放 1.1GB 内存给其他 APP
缺点: 首次/再次调用有 8-15 秒冷启动 (Mate 40e)

**策略 B: 常驻保活（高级选项，默认关闭）**

> **⚠️ 仅适合 Mate 40e 且用户明确知晓内存代价时使用。**
> 常驻 1.1GB 在多任务场景（微信 + 浏览器 + 视频会议）下极大概率被 LMK 击杀，
> 与其被系统不确定地杀掉，不如主动卸载让出内存。

服务启动后永不自动停止，始终占用 1.1GB 内存。仅在以下条件同时满足时考虑:
- 用户每天使用 ASR 频率 >20 次
- 用户能接受通知栏常驻图标
- 用户理解 1.1GB 内存占用对多任务的影响

实现方式:

```kotlin
// 仅通过 ASR 服务设置界面开启（用户主动操作）
// SettingsActivity.kt 中修改 SharedPreferences
class SherpaOnnxAsrService : Service() {
    companion object {
        var persistentMode = false  // 默认 false，由 SettingsActivity 控制
    }

    override fun onUnbind(intent: Intent): Boolean {
        if (!persistentMode && boundClients.decrementAndGet() == 0) {
            shutdownHandler.postDelayed(shutdownRunnable, 60_000)
        }
        return true
    }
}
```

> **不做开机自启**: 即使开启常驻模式，也不注册 `BOOT_COMPLETED` 广播。
> 1.1GB 模型在开机后自动加载会严重影响开机体验，且华为 ROM 对自启动管控严格。

**策略对比**:

| 维度 | 策略 A: 空闲超时 (默认) | 策略 B: 常驻保活 (高级) |
|------|------------------------|------------------------|
| 首次调用延迟 | 8-15 秒 (Mate 40e 冷启动) | 首次 8-15 秒，之后 0 延迟 |
| 日常调用延迟 | 大多热启动 <200ms | 始终 0 延迟 |
| 内存占用 | 空闲 60s 后释放 1.1GB | 始终占用 1.1GB |
| LMK 风险 | 低 (空闲时不存在) | 高 (多任务场景下优先击杀目标) |
| 电量影响 | 几乎无 | 进程存在但不耗电 |
| 适用场景 | 偶尔使用 ASR (默认) | 输入法等极高频场景 |
| 推荐度 | ⭐⭐⭐ 默认 | ⭐ 谨慎开启 |

**推荐方案**: 默认策略 A (60s 空闲超时)。策略 B 仅作为 ASR 服务设置界面中的高级选项，由用户主动开启。

#### 完整生命周期状态机

```
                    ┌──────────┐
                    │  未安装   │
                    └────┬─────┘
                         │ 安装 APK
                         ▼
                    ┌──────────┐
         ┌─────────│  已停止   │─────────┐
         │         └────┬─────┘         │
         │              │               │
         │   startForegroundService()   │ 用户手动开启
         │   + bindService()            │ 常驻模式
         │   (任何客户端)              │
         │              │               │
         │              ▼               │
         │         ┌──────────┐         │
         │         │ 初始化中  │◄────────┘
         │         │ (8-15s)  │
         │         └────┬─────┘
         │              │ 模型加载完成
         │              ▼
         │         ┌──────────┐
         │         │  就绪中   │◄──────────────┐
         │         │ (可接受   │               │
         │         │  请求)    │               │
         │         └──┬───┬───┘               │
         │            │   │                   │
         │   客户端绑定 │   │ 客户端解绑        │
         │            │   │                   │
         │            ▼   ▼                   │
         │   ┌──────────┐ ┌──────────────┐    │
         │   │  工作中   │ │  空闲中(60s) │    │
         │   │ (推理中)  │ │  倒计时中    │────┘
         │   └────┬─────┘ └──────┬───────┘ 有新客户端绑定
         │        │              │           (取消倒计时)
         │        │ 推理完成      │ 60s 倒计时到期
         │        ▼              │ (且 persistentMode=false)
         │   回到"就绪中"         ▼
         │               ┌──────────┐
         └───────────────│  已停止   │
                         └──────────┘
```

---

### 2.3 Phase 2: 多协议接入 + VAD 流式识别 (v2 规划)

> **本节为 v2 扩展方案，MVP 阶段不实现。** 设计保留于此，确保架构预留扩展点。

#### 2.3.1 HTTP/WebSocket Server (给自有 WebView)

**适用场景**: 自有 APP 内嵌 WebView 的 H5 页面需要调用 ASR 能力（非第三方 WebView）。

**前置条件**:
1. WebView 配置 `mixedContentMode = MIXED_CONTENT_ALWAYS_ALLOW`
2. ASR Service 的 `AndroidManifest.xml` 添加 `usesCleartextTraffic="true"` 或 `network_security_config.xml` 放行 `127.0.0.1`
3. LocalHttpServer 添加 OPTIONS 预检 + CORS 头响应

**⚠️ 不可用场景** (死局，无法绕过):
- 微信公众号/外部 HTTPS H5 → Mixed Content 拦截
- Chrome/Safari 等外部浏览器 → `127.0.0.1` 指向手机自身

**技术选型**: NanoHTTPD (~50KB, 纯 Java, 无额外依赖)

```
AIDL Service 扩展 (v2 onCreate 增加):
  httpServer = LocalHttpServer(127.0.0.1:8765)   ← CORS + OPTIONS
  wsServer   = LocalWebSocketServer(127.0.0.1:8766) ← 流式音频帧

API 端点 (与 AIDL 接口对等):
  GET  /status                  → 服务状态
  POST /recognize               → 同步识别 (octet-stream → JSON)
  POST /recognize/async         → 异步识别 (返回 requestId)
  GET  /result/{requestId}      → 查询异步结果
  DELETE /cancel/{requestId}    → 取消任务
  WS   /ws/recognize            → 流式音频帧 → JSON 部分结果
```

#### 2.3.2 VAD 断句流式识别

**问题**: MVP 仅支持"一次性提交完整短句"，无法处理连续对话场景。

**v2 方案 — 客户端 VAD + 逐句提交**:

```
长音频输入 (连续说话):
  │
  ▼
┌─────────────────┐
│  VAD 断句模块    │  ← 客户端 SDK 内置，非服务端
│  (Silero VAD    │
│   或 WebRTC VAD) │
└────────┬────────┘
         │ 检测到静音 > 800ms → 切句
         ▼
  [句1: "你好"] → AIDL submit → 识别 → 回调1
  [句2: "今天天气"] → AIDL submit → 识别 → 回调2
  [句3: "怎么样"] → AIDL submit → 识别 → 回调3
```

> **为什么是客户端 VAD 而非服务端**: 音频通过 AIDL 传输已是完整数据。
> 在客户端 SDK 中做 VAD 切句，每个短句独立 `submit()`，自然利用现有队列的短音频优先策略。
> 服务端不感知"流式"——它收到的始终是 VAD 切好的短句。

**VAD 选型**:
- **Silero VAD** (ONNX 模型，<1MB): 精度最高，适合语音助手场景
- **WebRTC VAD** (纯 DSP): 零模型依赖，适合输入法/录音机场景

**客户端 API 扩展 (v2)**:
```kotlin
// v2 流式 API (客户端 VAD 自动断句)
client.startStreamingRecognition(
    vadConfig = VadConfig(
        model = VadModel.SILERO,  // 或 WEBRTC
        silenceTimeoutMs = 800,   // 静音 800ms 判定为句尾
        minSpeechMs = 200,        // 最小有效语音长度
    ),
    callback = object : StreamingRecognitionCallback {
        override fun onPartialSentence(text: String, isFinal: Boolean) {
            // isFinal=true: VAD 检测到句尾，本句识别完成
            // isFinal=false: 可能是超长句的中间结果 (可选)
        }
    }
)
// 持续喂入音频
client.feedAudio(floatArray)
client.feedAudio(floatArray)
// ...
client.stopStreaming()
```

> **v2 不做服务端伪流式**: 不在服务端每 2s 重跑全量识别——VAD 断句后在客户端侧自然形成短句，
> 每句独立走队列，CPU/电量开销与 MVP 逐句提交一致。

---

## 三、项目结构

在 `android/` 下新建 `SherpaOnnxAsrService/`，包含三个 Gradle 模块:

```
android/SherpaOnnxAsrService/
├── settings.gradle.kts
├── build.gradle.kts
│
├── asr_service/                            # ASR 服务 APK 模块
│   ├── build.gradle.kts
│   └── main/
│       ├── AndroidManifest.xml
│       ├── aidl/com/k2fsa/sherpa/onnx/asr/service/
│       │   ├── ISherpaAsrService.aidl       # 主服务接口
│       │   ├── ISherpaAsrCallback.aidl      # 异步识别回调 (oneway)
│       │   ├── AsrRequest.aidl              # 请求 (Parcelable)
│       │   ├── AsrResult.aidl               # 结果 (Parcelable)
│       │   ├── ServiceStatus.aidl           # 服务状态 (Parcelable)
│       │   └── ServiceEvent.aidl            # 服务事件 (Parcelable)
│       ├── java/com/k2fsa/sherpa/onnx/asr/service/
│       │   ├── SherpaOnnxAsrService.kt      # 前台 Service (MVP 纯 AIDL, v2 +HTTP/WS)
│       │   ├── ModelManager.kt              # 模型加载/卸载/生命周期
│       │   ├── InferenceEngine.kt           # OfflineRecognizer 线程安全封装
│       │   ├── AudioDataReader.kt           # 从 ParcelFileDescriptor 读取音频数据
│       │   ├── AsrRequestProcessor.kt       # 优先级队列 + 串行调度
│       │   ├── ModelDownloadManager.kt      # 从国内 CDN/OSS 下载模型 (断点续传)
│       │   ├── ServiceStatusProvider.kt     # 状态跟踪
│       │   └── AudioBufferPool.kt           # 预分配 native buffer
│       └── libs/
│
├── client_sdk/                              # 客户端 SDK (.aar) 模块
│   ├── build.gradle.kts
│   └── src/main/java/com/k2fsa/sherpa/onnx/asr/client/
│       ├── SherpaSpeechClient.kt            # 主 API 入口
│       ├── RecognitionCallback.kt           # 结果回调接口
│       ├── ServiceConnectionManager.kt      # Binder 连接/重连/DeathRecipient 管理
│       ├── AudioDataProvider.kt             # 音频数据写入 pipe，传递 ParcelFileDescriptor
│       ├── VadSentenceSplitter.kt           # [v2] VAD 断句模块 (Silero/WebRTC)
│       ├── RecognitionRequest.kt            # 请求 POJO
│       └── RecognitionResult.kt             # 结果 POJO
│
├── asr_service/v2/                           # [v2 扩展] Phase 2 多协议接入
│   └── src/main/java/.../asr/service/
│       ├── LocalHttpServer.kt               # [v2] 嵌入式 HTTP Server (自有 WebView)
│       └── LocalWebSocketServer.kt          # [v2] 嵌入式 WebSocket Server (流式)
│
└── demo_app/                                # 演示 APP（验证用）
    ├── build.gradle.kts
    └── src/main/java/.../MainActivity.kt
```

---

## 四、AIDL 接口定义

### ISherpaAsrService.aidl — 核心服务接口

```java
interface ISherpaAsrService {
    // 提交音频识别任务，立即返回 requestId
    String submit(in AsrRequest request, ISherpaAsrCallback callback);

    // 取消指定任务
    void cancel(String requestId);

    // 查询服务状态
    ServiceStatus getStatus();

    // 模型是否已加载就绪
    boolean isReady();

    // 注册/取消服务级事件监听
    void registerListener(ISherpaAsrCallback listener);
    void unregisterListener(ISherpaAsrCallback listener);
}
```

### ISherpaAsrCallback.aidl — 异步回调

```java
oneway interface ISherpaAsrCallback {
    void onResult(String requestId, in AsrResult result);    // 识别完成
    void onServiceEvent(in ServiceEvent event);               // 服务事件
}
```

> `oneway` 关键: 防止服务端因慢/崩溃客户端而阻塞

### 数据类

**AsrRequest** (输入) — ⚠️ 关键设计: Binder IPC 缓冲区限制:

> **为什么不用 `float[] audioData`?**
> Android Binder 机制有严格的事务缓冲区大小限制（通常 **1MB**，且是当前进程所有进行中的 Binder 事务共享）。
> 16kHz float32 音频每秒 = 64KB，15 秒 = 960KB，30 秒 = 1.92MB。
> 一旦通过 AIDL 传递超过 ~15 秒的 `float[]`，系统将抛出 `TransactionTooLargeException`，客户端直接崩溃。
>
> **解决方案**: 使用 `ParcelFileDescriptor` 传递音频数据。客户端将音频写入管道 (pipe)，服务端通过文件描述符流式读取。
> 这彻底消除了 Binder 传输大小限制，支持任意长度的音频。

```java
// AsrRequest.aidl
parcelable AsrRequest {
    ParcelFileDescriptor audioFd;   // 音频数据管道 (替代 float[])
    long audioDataLength;           // float32 采样点数
    int sampleRate;                 // 采样率
    String language;                // 可选语言提示 (zh/en/ja/...)
    String hotwords;                // 可选热词 (逗号分隔)
    int priority;                   // 优先级 (0=低/1=普通/2=高)
    long maxWaitMs;                 // 队列等待超时
}
```

> **备选方案**: API 27+ 可使用 `SharedMemory` (ASharedMemory) 实现零拷贝共享内存传输。
> `SharedMemory` 通过 AIDL 传递 `ParcelFileDescriptor` 底层支持的 `ashmem` 句柄，
> 适合需要极低延迟的场景。本文档默认采用 `ParcelFileDescriptor` (兼容所有 API 级别)。

**服务端读取 ParcelFileDescriptor**:

```kotlin
// AudioDataReader.kt — 从 ParcelFileDescriptor 读取音频数据
object AudioDataReader {
    fun readFloats(fd: ParcelFileDescriptor, floatCount: Long): FloatArray {
        val floats = FloatArray(floatCount.toInt())
        try {
            FileInputStream(fd.fileDescriptor).use { fis ->
                val bytes = ByteArray(floatCount.toInt() * 4)
                var read = 0
                while (read < bytes.size) {
                    val n = fis.read(bytes, read, bytes.size - read)
                    if (n < 0) break
                    read += n
                }
                ByteBuffer.wrap(bytes)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .get(floats)
            }
        } finally {
            // ⚠️ 必须显式关闭 FD：Binder 跨进程时 dup() 了一份新 FD 给服务端，
            //    不关闭会导致每次识别泄漏一个 FD。Linux 单进程上限 1024。
            try { fd.close() } catch (e: IOException) { /* 已关闭或异常 */ }
        }
        return floats
    }
}
```

> **重要**: `ParcelFileDescriptor` 在 AIDL 中必须标记方向 (`in`)，传输后 `fd` 的 ownership 转移。
> 服务端读取完毕后应关闭 fd；客户端写入完毕后也应关闭自己持有的端。

**AsrResult** (输出):
- `String text` — 识别文本
- `String[] tokens` — token 列表
- `String lang` — 检测语言
- `int status` — 0=成功/1=超时/2=错误/3=取消
- `String errorMessage`
- `long inferenceTimeMs` — 推理耗时
- `long queueWaitTimeMs` — 排队耗时

---

## 五、关键实现要点

### 5.1 SherpaOnnxAsrService.kt

参考现有 `SherpaOnnxJavaDemo/SpeechSherpaRecognitionService.java`，关键改造:

| 现有实现 | 改造为 |
|---------|--------|
| `onBind()` → `return null` | `onBind()` → `return stub.asBinder()` |
| `exported="false"` | `exported="true"` + 自定义权限 |
| AppViewModel + LiveData 通信 | AIDL callback + `oneway` 回调 |
| 硬编码模型路径 | ModelManager 动态查找/下载 |
| 单一线程处理 | 任务队列 + 优先级调度 |
| 无客户端管理 | `linkToDeath` 检测客户端退出, 自动清理 |
| 仅同进程通信 | MVP: 纯 AIDL 跨进程；[v2]: 新增 HTTP/WS Server 支持自有 WebView |

`onCreate()` 生命周期:

> **关键**: Android 前台 Service 必须在 `onCreate()` 后 **5 秒内** 调用 `startForeground()`。
> 因此 `startForeground()` 必须是 `onCreate()` 中**第一个**操作（模型加载可能耗时 10+ 秒，
> 但在 `startForeground()` 之后系统已将该进程标记为前台，不会触发 ANR）。

```kotlin
override fun onCreate() {
    super.onCreate()
    createNotificationChannel()

    // ⚠️ 第 0 步 (必须在 5 秒内完成): 立即推入前台
    // 此时模型尚未加载，通知显示"初始化中..."
    startForeground(NOTIFICATION_ID, buildNotification("初始化中..."))

    // 1. 加载模型 (~10s, 此时服务已在前台, 系统不会杀)
    modelManager = ModelManager(applicationContext).also { it.initialize() }
    inferenceEngine = InferenceEngine(modelManager.getRecognizer())

    // 2. 启动任务队列 (唯一推理入口)
    requestProcessor = AsrRequestProcessor(inferenceEngine)

    updateNotification("模型已就绪 — AIDL 可接入")
}

override fun onDestroy() {
    requestProcessor.shutdown()
    inferenceEngine.release()
    modelManager.release()
    super.onDestroy()
}

// 系统内存压力回调 — LMK 前最后防线
override fun onTrimMemory(level: Int) {
    super.onTrimMemory(level)
    if (level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) {
        Log.w(TAG, "系统内存严重不足 (level=$level)，即将被 LMK 击杀")
        // 通知所有客户端服务即将不可用
        notifyServiceDegraded("memory_pressure")
        // 不主动释放模型 — ONNX Runtime 的 native 内存在 Java 堆外，
        // 主动释放可能导致崩溃。让 LMK 直接杀进程更安全。
    }
}
```

> **配置前提 (MVP)**: 确保 `AndroidManifest.xml` 中设置了:
> - `android:largeHeap="true"` (为 1.1GB 模型内存)
> - `android:process=":asr_service"` (独立进程隔离)
> - `android:exported="true"` + 自定义权限 (跨 APP 调用)
>
> **v2 额外配置**: `android:usesCleartextTraffic="true"`（仅在启用 HTTP/WS 时需要）

### 5.2 ModelManager.kt

```kotlin
// 模型文件布局 (filesDir/models/)
// 从国内 CDN/OSS 下载 (非 GitHub Releases)
sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/
├── conv_frontend.onnx       (~42 MB)
├── encoder.int8.onnx        (~174 MB)
├── decoder.int8.onnx        (~722 MB)
├── tokenizer/
│   ├── tokenizer.json
│   ├── vocab.json
│   └── merges.txt
└── manifest.json            # SHA256 + version + download_url

// 初始化
val config = OfflineRecognizerConfig(
    modelConfig = OfflineModelConfig(
        qwen3Asr = OfflineQwen3AsrModelConfig(
            convFrontend = "$modelDir/conv_frontend.onnx",
            encoder = "$modelDir/encoder.int8.onnx",
            decoder = "$modelDir/decoder.int8.onnx",
            tokenizer = "$modelDir/tokenizer",
            maxTotalLen = 512,
            maxNewTokens = 128,
            temperature = 1e-6f,
            topP = 0.8f,
            seed = 42,
        ),
        numThreads = 3,
        provider = "cpu",
    )
)
recognizer = OfflineRecognizer(assetManager = null, config = config)
```

### 5.3 InferenceEngine.kt

> **关键设计**: `InferenceEngine` **不对外暴露直接调用接口**。所有识别请求必须通过
> `AsrRequestProcessor` (任务队列) 串行调度，防止并发访问 ONNX Runtime session。

```kotlin
// InferenceEngine.kt — 仅由 AsrRequestProcessor 内部调用
// 标记为 internal，第三方代码不可直接访问
internal class InferenceEngine(private val recognizer: OfflineRecognizer) {

    @Synchronized  // ONNX Runtime session 非线程安全
    fun recognize(
        audioData: FloatArray,
        sampleRate: Int,
        hotwords: String? = null,
        language: String? = null,
    ): AsrResult {
        val stream = recognizer.createStream(hotwords ?: "")
        if (language != null) {
            stream.setOption("language", language)
        }
        stream.acceptWaveform(audioData, sampleRate)
        recognizer.decode(stream)
        val result = recognizer.getResult(stream)
        stream.release()
        return mapToAsrResult(result)
    }

    fun release() {
        recognizer.release()
    }
}
```

**唯一调用链** (所有请求统一入队):

```
客户端进程                             服务端进程 (:asr_service)
    │                                       │
    │── submit(AsrRequest{audioFd, ...}) ─→│
    │                                       │── AIDL Stub
    │                                       │   ├ AudioDataReader.readFloats(fd)
    │                                       │   └ 构建 QueueEntry{
    │                                       │       audioData, priority(短/长),
    │                                       │       language, hotwords, callback
    │                                       │     }
    │                                       │── requestProcessor.enqueue(entry)
    │                                       │── [排队中...]
    │                                       │── [轮到本请求]
    │                                       │── InferenceEngine.recognize(...)
    │                                       │── callback.onResult(...)
    │←──────────── callback.onResult ──────│
    │                                       │
```

> 绝不允许任何路径绕过队列直接调用 `InferenceEngine.recognize()`。
> 这确保了: (1) ONNX Runtime 线程安全 (2) 短音频不被长音频阻塞 (3) 可取消/可超时。

### 5.4 AsrRequestProcessor.kt — 任务调度

> **短音频优先策略**: 避免 30 秒会议录音阻塞 2 秒输入法请求。
> 短音频 (<10s) 获得队列优先级提升，长音频不阻塞短请求。

```kotlin
// 优先级队列: 短音频优先 > 显式优先 > 同优先级 FIFO
private val queue = PriorityBlockingQueue<QueueEntry>(11) { a, b ->
    // 第 1 层: 短音频优先 (duration < 10s)
    val aShort = a.audioDurationMs < 10_000
    val bShort = b.audioDurationMs < 10_000
    if (aShort != bShort) return@PriorityBlockingQueue if (aShort) -1 else 1

    // 第 2 层: 显式优先级 (0=低/1=普通/2=高)
    val cmp = b.request.priority - a.request.priority
    if (cmp != 0) return@PriorityBlockingQueue cmp

    // 第 3 层: FIFO
    a.enqueueTime.compareTo(b.enqueueTime)
}

// 单线程消费 loop (模型不支持并行)
private val executor = Executors.newSingleThreadExecutor { r ->
    Thread(r, "InferenceWorker").apply { priority = Thread.MAX_PRIORITY }
}

init {
    executor.execute {
        while (!Thread.currentThread().isInterrupted) {
            val entry = queue.take()  // 阻塞等待

            // ⚠️ 竞态保护: 出队后先检查客户端是否还活着
            // 场景: 队列有 5 个请求，第 3 个客户端被用户杀后台。
            // 如果不检查，会白白加载音频并执行耗时推理 (~几秒)，
            // 直到 callback 时才发现 DeadObjectException，浪费算力。
            if (entry.callback != null && !entry.callback.asBinder().pingBinder()) {
                Log.w(TAG, "客户端已死亡，跳过 requestId=${entry.request.requestId}")
                entry.request.audioFd?.close()  // 清理泄漏的 FD
                continue
            }

            try {
                val fd = entry.request.audioFd
                val audioData = AudioDataReader.readFloats(fd, entry.request.audioDataLength)
                val result = inferenceEngine.recognize(
                    audioData, entry.request.sampleRate,
                    entry.request.hotwords, entry.request.language
                )
                entry.callback?.onResult(entry.request.requestId, result)
            } catch (e: DeadObjectException) {
                Log.w(TAG, "回调时客户端已死亡: ${e.message}")
            } catch (e: Exception) {
                Log.e(TAG, "推理异常: ${e.message}", e)
                entry.callback?.onServiceEvent(ServiceEvent.error(e.message))
            }
        }
    }
}

// 入队时自动计算音频时长 (基于采样点数)
data class QueueEntry(
    val request: AsrRequest,
    val callback: ISherpaAsrCallback?,
    val enqueueTime: Long = System.currentTimeMillis(),
) {
    // 16kHz 采样率下: durationMs = audioDataLength / 16
    val audioDurationMs: Long = request.audioDataLength * 1000 / request.sampleRate
}
```

**示例 — 队列调度效果**:

```
入队顺序: [30s会议录音(低), 2s输入法(普通), 15s语音消息(高)]
FIFO 处理: 会议 → 输入法 → 语音消息  (输入法等 30s+)
短优处理: 输入法 → 语音消息 → 会议  (输入法立即处理)

入队顺序: [2s输入法(普通), 2s语音搜索(普通)]
短优处理: 先入先出 (同为短音频且同优先级)

竞态示例:
队列: [请求1✅, 请求2✅, 请求3❌(已死), 请求4✅, 请求5✅]
Worker 消费到请求3 → pingBinder()=false → 跳过 → 立即处理请求4 (节省数秒)
```

### 5.5 客户端 SDK API

```kotlin
// ======== 用法示例（第三方 APP 中） ========
val client = SherpaSpeechClient(context)

client.setRecognitionCallback(object : RecognitionCallback {
    override fun onResult(result: RecognitionResult) {
        textView.text = result.text  // "你好世界"
    }
    override fun onError(error: RecognitionError) { /* ... */ }
    override fun onServiceDied() {
        // 服务被 LMK 击杀，SDK 内部自动重连
        showToast("ASR 服务重连中...")
    }
    override fun onServiceReconnected() {
        showToast("ASR 服务已恢复")
    }
})

// connect() 内部自动处理 startForegroundService + bindService
client.connect()
client.recognize(audioData, sampleRate = 16000, hotwords = "sherpa", language = "zh")
```

```kotlin
// ======== ServiceConnectionManager.kt — 核心连接/重连管理 ========
class ServiceConnectionManager(private val context: Context) {
    private var service: ISherpaAsrService? = null
    private var deathRecipient: IBinder.DeathRecipient? = null
    private var reconnectAttempt = 0
    private val maxReconnectDelay = 60_000L  // 最大重连间隔 60s

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = ISherpaAsrService.Stub.asInterface(binder)
            reconnectAttempt = 0

            // 注册 DeathRecipient: 服务进程被杀时客户端不会崩溃
            deathRecipient = IBinder.DeathRecipient {
                Log.w(TAG, "ASR 服务被 LMK 击杀, Binder 已死亡")
                service = null
                onServiceDied()
                scheduleReconnect()
            }.also { binder?.linkToDeath(it, 0) }

            onServiceReady()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
        }
    }

    fun connect() {
        val intent = Intent().setComponent(
            ComponentName("com.k2fsa.sherpa.onnx.asr.service",
                           "com.k2fsa.sherpa.onnx.asr.service.SherpaOnnxAsrService")
        )
        // 第 1 步: 确保服务推入前台 (onCreate 中会立即 startForeground)
        context.startForegroundService(intent)
        // 第 2 步: 绑定获取 Binder
        context.bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    // 指数退避重连: 1s → 2s → 4s → 8s → ... → 60s
    private fun scheduleReconnect() {
        val delay = (1000L * (1 shl reconnectAttempt)).coerceAtMost(maxReconnectDelay)
        reconnectAttempt++
        Log.i(TAG, "将在 ${delay}ms 后尝试重连 (第 $reconnectAttempt 次)")
        handler.postDelayed({
            connect()  // 重新 startForegroundService + bindService
        }, delay)
    }

    fun disconnect() {
        deathRecipient?.let { service?.asBinder()?.unlinkToDeath(it, 0) }
        context.unbindService(serviceConnection)
    }
}
```

```kotlin
// ======== AudioDataProvider.kt — 音频写入 Pipe / SharedMemory ========
// 客户端SDK内部使用，将 FloatArray 通过 ParcelFileDescriptor 传递给服务端
class AudioDataProvider {
    companion object {
        fun toParcelFileDescriptor(floatArray: FloatArray): Pair<ParcelFileDescriptor, () -> Unit> {
            val pipe = ParcelFileDescriptor.createPipe()  // pipe[0]=read, pipe[1]=write
            val readFd = pipe[0]   // 传给服务端
            val writeFd = pipe[1]  // 客户端写入

            thread {
                FileOutputStream(writeFd.fileDescriptor).use { fos ->
                    val byteBuffer = ByteBuffer.allocate(floatArray.size * 4)
                        .order(ByteOrder.LITTLE_ENDIAN)
                    byteBuffer.asFloatBuffer().put(floatArray)
                    fos.write(byteBuffer.array())
                    fos.flush()
                }
                writeFd.close()
            }

            return Pair(readFd) { readFd.close() }
        }
    }
}
```

```kotlin
// ======== SherpaSpeechClient.kt — 主 API 入口 ========
class SherpaSpeechClient(private val context: Context) {
    private val connectionManager = ServiceConnectionManager(context)

    fun connect() = connectionManager.connect()

    fun recognize(audioData: FloatArray, sampleRate: Int = 16000,
                  hotwords: String? = null, language: String? = null): String {
        val service = connectionManager.getService()
            ?: throw IllegalStateException("服务未连接")

        // 1. 将音频数据写入 pipe
        val (audioFd, cleanup) = AudioDataProvider.toParcelFileDescriptor(audioData)
        try {
            // 2. 构建 Parcelable 请求 (传递 FD 而非大数组)
            val request = AsrRequest(
                audioFd = audioFd,
                audioDataLength = audioData.size.toLong(),
                sampleRate = sampleRate,
                language = language ?: "",
                hotwords = hotwords ?: "",
                priority = 1,
                maxWaitMs = 30_000
            )
            // 3. 通过 AIDL 同步调用 (Binder 仅传输 FD 句柄，无大小限制)
            val requestId = service.submit(request, null)
            // 4. 轮询等待结果 (简化示例，生产环境用 Callback)
            return awaitResult(requestId)
        } finally {
            cleanup()
        }
    }
}
```

---

## 六、性能评估

### 6.1 内存预算

| 组件 | 估计大小 |
|------|---------|
| ONNX 模型权重 (native heap) | ~940 MB |
| KV Cache (max_total_len=512) | ~50 MB |
| ONNX Runtime 内部 buffers | ~50 MB |
| JVM 堆 (ART + 服务代码) | ~50 MB |
| 音频 feature buffer | ~10 MB |
| **总进程内存** | **~1.1 GB** |

> Android 10 单进程上限通常 1.5-2GB（因 OEM 而异）。**需要 `android:largeHeap="true"`**。
>
> **⚠️ LMK (Low Memory Killer) 风险**: 即使配置了 `largeHeap`，当系统内存紧张时（如用户玩大型游戏），
> 占用 1.1GB 的 ASR 服务进程仍在 LMK 的高优先级击杀名单上。
> 客户端 SDK 必须实现 **DeathRecipient + 指数退避重连** 机制，确保服务被击杀后能优雅恢复，
> 决不能因为对端 Binder 死亡导致宿主 APP 崩溃。详见 5.5 节。

### 6.2 推理速度预估 (Mate 40e 实测)

> **目标设备**: Huawei Mate 40e, 麒麟 900E (1×Cortex-A77@3.13GHz + 3×A77@2.54GHz + 4×A55@2.05GHz), 8GB RAM

| 音频长度 | 预估推理时间 | RTF | 说明 |
|---------|------------|-----|------|
| 2s (输入法短语) | ~1.5-3s | 0.75-1.5x | 短句 token 少，decoder 步数少 |
| 5s (语音搜索) | ~3-7s | 0.6-1.4x | 典型使用场景 |
| 10s (语音消息) | ~6-15s | 0.6-1.5x | 推荐上限 |
| 30s (会议录音) | ~20-45s | 0.67-1.5x | 不推荐 (队列阻塞风险) |

> **瓶颈**: LLM Decoder 自回归生成 — 每个输出 token 需经过 28 层 Qwen3 Transformer。
> 128 tokens × 28 层 = 3584 次 decoder 前向传播。`numThreads=3` (匹配 A77 大核数)。
>
> **结论**: Mate 40e **能做**，但 30s 以上的长音频不推荐。
> 建议客户端 SDK 在 UI 层面引导用户限制单次录音 ≤15s。

### 6.3 内存与 LMK 策略 (Mate 40e 专项)

| 场景 | ASR 进程状态 | 可用 RAM (给其他 APP) | LMK 风险 |
|------|------------|---------------------|---------|
| ASR 空闲 (模型已卸载) | 仅空进程 (~50MB) | ~7.95GB | 几乎为零 |
| ASR 工作中 (~1.1GB) | 前台 Service | ~6.9GB | 低 (前台进程受保护) |
| ASR 常驻模式 (1.1GB) | 前台 Service 空闲 | ~6.9GB | **高** — 多任务时 LMK 优先击杀 |
| 游戏 + 微信 + ASR | 前台 Service 工作中 | ~5GB | 中 — 可能触发 LMK |

**推荐策略**: 默认空闲 60s 卸载模型 (策略 A)，常驻模式仅在用户主动开启后生效。

---

## 七、实现步骤（分阶段）

### Phase 1: 工程骨架 (1-2天)
1. 创建 `android/SherpaOnnxAsrService/` 目录 + Gradle 多模块结构
2. 编写全部 AIDL 文件 + Parcelable 类（AsrRequest 使用 ParcelFileDescriptor）
3. 验证 AIDL 编译通过，确认 `client_sdk` 模块能引用生成的 Stub

### Phase 2: 核心服务 (3-4天)
4. 实现 `SherpaOnnxAsrService.kt`: 前台服务 (onCreate 立即 startForeground) + AIDL Stub + onTrimMemory
5. 实现 `ModelManager.kt`: OfflineRecognizer 加载 + 空闲超时卸载 (60s) + 常驻模式开关
6. 实现 `InferenceEngine.kt`: @Synchronized 线程安全封装 (internal, 仅 AsrRequestProcessor 调用)
7. 实现 `AudioDataReader.kt`: 从 ParcelFileDescriptor 读取 FloatArray
8. 实现 `AsrRequestProcessor.kt`: PriorityBlockingQueue + 短音频优先 + 单线程消费
9. 实现 `ServiceStatusProvider.kt`: 模型状态/队列深度/客户端数 追踪

### Phase 3: 模型下载 (2-3天)
10. 实现 `ModelDownloadManager.kt`: 国内 CDN/OSS 下载 + 断点续传 + SHA256 校验 + tar.bz2 解压
11. 下载进度通知栏显示 + 仅 WiFi 选项

### Phase 4: 客户端 SDK (2-3天)
12. 创建 `client_sdk/` 模块
13. 实现 `AudioDataProvider.kt`: FloatArray → ParcelFileDescriptor pipe
14. 实现 `ServiceConnectionManager.kt`: startForegroundService + bindService + DeathRecipient + 指数退避重连 + 华为 ROM fallback
15. 实现 `SherpaSpeechClient.kt`: 对外 API (connect/recognize/cancel/disconnect)
16. 构建 `.aar` 包

### Phase 5: 测试验证 (2-3天, Mate 40e 真机)
17. 单次短音频 (<10s) 端到端识别测试
18. 多客户端并发 + 短音频优先调度验证
19. 华为 ROM 后台启动限制场景测试
20. 服务被 LMK 击杀后 DeathRecipient 重连测试
21. Android Studio Profiler 内存检查 (峰值 <1.3GB)
22. 空闲 60s 模型卸载 + 再次调用冷启动测试

### Phase 6: v2 扩展 (3-5天, MVP 稳定后)
23. 实现 `LocalHttpServer.kt`: NanoHTTPD 嵌入式 HTTP Server (127.0.0.1:8765) + CORS + OPTIONS
24. 实现 `LocalWebSocketServer.kt`: WebSocket Server (127.0.0.1:8766)
25. 实现 `VadSentenceSplitter.kt`: 客户端 SDK 内置 Silero/WebRTC VAD 断句
26. 客户端 `StreamingRecognitionCallback` API + `feedAudio/stopStreaming`
27. 自有 WebView 场景端到端测试 (MixedContentMode 放行 + CleartextTraffic)

---

## 八、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **低端手机内存不足** (4GB RAM) | 模型加载失败/OOM | `largeHeap=true`; 加载前检测可用内存 < 2GB 时提示不支持 |
| **LMK (Low Memory Killer)** 击杀 ASR 服务 | 客户端 Binder 死亡，识别中断 | SDK 注册 `DeathRecipient` + 指数退避重连；宿主 APP 不因对端死亡而崩溃 |
| **Binder 事务超限** (`TransactionTooLargeException`) | >15s 音频通过 AIDL 传输导致客户端崩溃 | 使用 `ParcelFileDescriptor` 传音频流替代 `float[]`（详见第四节） |
| **推理速度慢** (低端设备 >20s) | 用户体验差 | 限制最大音频 30s; 明确告知离线模型非实时; 未来可选 QNN DSP 加速 |
| **模型下载 1.9GB** | 流量消耗大; GitHub Releases 国内不稳定 | 国内 CDN/OSS + 仅 WiFi + 断点续传 + SHA256 校验 |
| **多 APP 并发请求** | 排队等待长 | 队列化 + 可取消 + 告知队列深度 |
| **Qwen3-ASR 仅离线** | 不能实时出字; 不支持增量 decode | MVP 仅短句一次性提交; v2 通过客户端 VAD 断句实现流式体验 |
| **华为 ROM 后台启动限制** | `startForegroundService()` 被拦截 | 主 APP 负责启动; 第三方仅 bindService; SecurityException fallback 引导用户打开主 APP |
| **长音频阻塞队列** | 30s 会议录音阻塞 2s 输入法请求 | 短音频优先 (<10s) 调度策略; 建议客户端限制录音 ≤15s |
| **FD 泄漏** (ParcelFileDescriptor) | 每次请求泄漏 1 个 FD；进程上限 1024 → 数百次后崩溃 | `finally { fd.close() }` 强制关闭（§4 AudioDataReader.kt） |
| **死客户端浪费推理** | 客户端已杀但 Worker 仍执行耗时推理（数秒） | 出队后 `pingBinder()` 检查客户端存活，已死则跳过（§5.4） |
| **ONNX Runtime 兼容性** | 麒麟 900E 特定指令集兼容性 | 使用 sherpa-onnx 内置 onnxruntime 构建（ARMv8.2 CPU target）; Mate 40e 真机验证 |

---

## 九、文件变更清单

全部为**新增文件**，不修改任何现有代码:

| 文件 | 说明 |
|------|------|
| `android/SherpaOnnxAsrService/settings.gradle.kts` | 多模块工程配置 |
| `android/SherpaOnnxAsrService/build.gradle.kts` | 根构建脚本 |
| `android/SherpaOnnxAsrService/asr_service/build.gradle.kts` | 服务模块 (minSdk 29, largeHeap) |
| `android/SherpaOnnxAsrService/asr_service/src/main/AndroidManifest.xml` | 权限 + Service 声明 (largeHeap, :asr_service) |
| `android/SherpaOnnxAsrService/asr_service/src/main/aidl/.../ISherpaAsrService.aidl` | 主 AIDL 接口 |
| `android/SherpaOnnxAsrService/asr_service/src/main/aidl/.../ISherpaAsrCallback.aidl` | 回调 AIDL (oneway) |
| `android/SherpaOnnxAsrService/asr_service/src/main/aidl/.../AsrRequest.aidl` | 请求 Parcelable (ParcelFileDescriptor + audioDataLength) |
| `android/SherpaOnnxAsrService/asr_service/src/main/aidl/.../AsrResult.aidl` | 结果 Parcelable |
| `android/SherpaOnnxAsrService/asr_service/src/main/aidl/.../ServiceStatus.aidl` | 状态 Parcelable |
| `android/SherpaOnnxAsrService/asr_service/src/main/aidl/.../ServiceEvent.aidl` | 事件 Parcelable |
| `android/SherpaOnnxAsrService/asr_service/src/main/java/.../SherpaOnnxAsrService.kt` | 前台 Service + 立即 startForeground + onTrimMemory + 华为 ROM 适配 |
| `android/SherpaOnnxAsrService/asr_service/src/main/java/.../ModelManager.kt` | 模型加载 + 空闲 60s 超时卸载 + 常驻模式开关 |
| `android/SherpaOnnxAsrService/asr_service/src/main/java/.../InferenceEngine.kt` | 推理引擎 (internal, 仅通过队列调用) |
| `android/SherpaOnnxAsrService/asr_service/src/main/java/.../AudioDataReader.kt` | 从 ParcelFileDescriptor 流式读取 FloatArray |
| `android/SherpaOnnxAsrService/asr_service/src/main/java/.../AsrRequestProcessor.kt` | PriorityBlockingQueue + 短音频优先 + 单线程消费 |
| `android/SherpaOnnxAsrService/asr_service/src/main/java/.../ModelDownloadManager.kt` | 国内 CDN/OSS 下载 + 断点续传 + SHA256 |
| `android/SherpaOnnxAsrService/asr_service/src/main/java/.../ServiceStatusProvider.kt` | 模型状态/队列深度/客户端数 追踪 |
| `android/SherpaOnnxAsrService/client_sdk/build.gradle.kts` | SDK 构建 |
| `android/SherpaOnnxAsrService/client_sdk/src/main/java/.../SherpaSpeechClient.kt` | 客户端主 API |
| `android/SherpaOnnxAsrService/client_sdk/src/main/java/.../RecognitionCallback.kt` | 回调接口 (onResult/onError/onServiceDied/onReconnected) |
| `android/SherpaOnnxAsrService/client_sdk/src/main/java/.../ServiceConnectionManager.kt` | 连接/DeathRecipient/指数退避重连/华为 ROM fallback |
| `android/SherpaOnnxAsrService/client_sdk/src/main/java/.../AudioDataProvider.kt` | 音频写入 pipe, 传递 ParcelFileDescriptor |
| `android/SherpaOnnxAsrService/client_sdk/src/main/java/.../VadSentenceSplitter.kt` | [v2] VAD 断句模块 (Silero/WebRTC) |
| `android/SherpaOnnxAsrService/asr_service/src/main/java/.../LocalHttpServer.kt` | [v2] 嵌入式 HTTP Server (自有 WebView, CORS + OPTIONS) |
| `android/SherpaOnnxAsrService/asr_service/src/main/java/.../LocalWebSocketServer.kt` | [v2] 嵌入式 WebSocket Server (流式音频帧) |
| `android/SherpaOnnxAsrService/demo_app/build.gradle.kts` | Demo 构建 |
| `android/SherpaOnnxAsrService/demo_app/src/main/java/.../MainActivity.kt` | Demo 入口 |

---

## 十、可复用的现有代码

无需从零编写，可参考以下现有实现:

| 现有代码 | 用于 |
|---------|------|
| `SherpaOnnxJavaDemo/SpeechSherpaRecognitionService.java` | 前台服务模式、AudioRecord 配置、通知创建 |
| `SherpaOnnxJavaDemo/MainActivity.java` | 服务绑定/启动模式 |
| `sherpa-onnx/kotlin-api/OfflineRecognizer.kt` (model 61) | Qwen3-ASR 配置模板 (conv_frontend + encoder + decoder + tokenizer) |
| `SherpaOnnxAar/sherpa_onnx/build.gradle.kts` | AAR 构建配置参考 |
| `sherpa-onnx/jni/offline-recognizer.cc` (lines 349-376) | JNI 参数映射参考 (createStream/acceptWaveform/decode/getResult) |
| 国内对象存储 (阿里云 OSS / 七牛) | 模型文件分发 CDN，替代 GitHub Releases |
| Silero VAD (`snakers4/silero-vad`) | [v2] 客户端 VAD 断句模型 (ONNX, <1MB) |
| WebRTC VAD (`webrtc-audio-processing`) | [v2] 客户端 VAD 备选方案 (纯 DSP, 零模型依赖) |
| NanoHTTPD (`org.nanohttpd:nanohttpd:2.3.1`) | [v2] 嵌入式 HTTP Server (自有 WebView 接入) |
