package com.k2fsa.sherpa.onnx.qwen3asr

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineQwen3AsrModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.getFeatureConfig
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

private const val TAG = "Qwen3-ASR-Demo"
private const val REQUEST_RECORD_AUDIO_PERMISSION = 200

class MainActivity : AppCompatActivity() {

    // UI components
    private lateinit var editModelPath: EditText
    private lateinit var recordButton: Button
    private lateinit var loadButton: Button
    private lateinit var textView: TextView
    private lateinit var statusText: TextView

    // Audio recorder
    private val audioRecorder = AudioRecorder()

    // ASR recognizer
    private var recognizer: OfflineRecognizer? = null
    private var isModelLoaded = false

    // Default model path
    private val defaultModelPath = "/data/local/tmp/qwen3-asr-0.6b-int8-2026-03-25"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            REQUEST_RECORD_AUDIO_PERMISSION
        )

        // Bind views
        editModelPath = findViewById(R.id.edit_model_path)
        recordButton = findViewById(R.id.record_button)
        loadButton = findViewById(R.id.load_button)
        textView = findViewById(R.id.my_text)
        statusText = findViewById(R.id.status_text)

        textView.movementMethod = ScrollingMovementMethod()

        // Set default model path
        editModelPath.setText(defaultModelPath)

        // Load button
        loadButton.setOnClickListener { onLoadModel() }

        // Record button
        recordButton.setOnClickListener { onRecordToggle() }
    }

    /**
     * 获取用户输入的模型路径
     */
    private fun getModelPath(): String {
        return editModelPath.text.toString().trim().ifBlank { defaultModelPath }
    }

    /**
     * 验证模型目录是否包含所需文件
     */
    private fun validateModelDir(modelDir: String): String? {
        val dir = File(modelDir)
        if (!dir.exists() || !dir.isDirectory) {
            return "模型目录不存在: $modelDir"
        }

        val requiredFiles = listOf("conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx")
        val tokenizerDir = File(modelDir, "tokenizer")

        for (file in requiredFiles) {
            val f = File(modelDir, file)
            if (!f.exists()) {
                return "缺少模型文件: ${f.absolutePath}"
            }
        }

        if (!tokenizerDir.exists() || !tokenizerDir.isDirectory) {
            return "缺少 tokenizer 目录: ${tokenizerDir.absolutePath}"
        }

        // 检查 tokenizer 目录中至少有一个文件
        val tokenizerFiles = tokenizerDir.list()
        if (tokenizerFiles.isNullOrEmpty()) {
            return "tokenizer 目录为空: ${tokenizerDir.absolutePath}"
        }

        return null // 验证通过
    }

    /**
     * 加载 Qwen3-ASR 模型
     */
    private fun onLoadModel() {
        val modelDir = getModelPath()

        // 先验证模型文件
        val error = validateModelDir(modelDir)
        if (error != null) {
            Toast.makeText(this, error, Toast.LENGTH_LONG).show()
            textView.text = "模型加载失败:\n$error\n\n" +
                    "请确保模型文件已放置在指定路径，或修改上方路径。\n" +
                    "下载地址: https://huggingface.co/csukuangfj/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
            return
        }

        loadButton.isEnabled = false
        loadButton.text = "加载中..."
        recordButton.isEnabled = false
        textView.text = "正在加载模型，请稍候...\n这可能需要 10-30 秒。"

        lifecycleScope.launch(Dispatchers.IO) {
            val startTime = System.currentTimeMillis()
            try {
                Log.i(TAG, "Loading model from: $modelDir")

                val config = OfflineRecognizerConfig(
                    featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
                    modelConfig = OfflineModelConfig(
                        qwen3Asr = OfflineQwen3AsrModelConfig(
                            convFrontend = "$modelDir/conv_frontend.onnx",
                            encoder = "$modelDir/encoder.int8.onnx",
                            decoder = "$modelDir/decoder.int8.onnx",
                            tokenizer = "$modelDir/tokenizer",
                        ),
                        numThreads = 3,
                        debug = false,
                    ),
                )

                // assetManager = null → 从文件系统路径加载
                recognizer = OfflineRecognizer(
                    assetManager = null,
                    config = config,
                )

                val elapsed = System.currentTimeMillis() - startTime
                isModelLoaded = true

                withContext(Dispatchers.Main) {
                    loadButton.text = "模型已加载 ✓"
                    recordButton.isEnabled = true
                    statusText.visibility = android.view.View.VISIBLE

                    val modelSize = formatModelSize(modelDir)
                    statusText.text = "模型已就绪 | 加载耗时: ${elapsed}ms | 模型大小: $modelSize"
                    textView.text = "✅ Qwen3-ASR 0.6B 模型加载成功！\n\n" +
                            "加载耗时: ${elapsed}ms (${elapsed / 1000.0}s)\n" +
                            "模型路径: $modelDir\n" +
                            "模型大小: $modelSize\n\n" +
                            "点击\"开始录音\"进行语音识别测试。"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model loading failed", e)
                withContext(Dispatchers.Main) {
                    loadButton.text = "加载模型"
                    loadButton.isEnabled = true
                    textView.text = "❌ 模型加载失败:\n${e.message}\n\n" +
                            "请检查模型文件是否完整。"
                    Toast.makeText(this@MainActivity, "模型加载失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    /**
     * 录音按钮切换
     */
    private fun onRecordToggle() {
        if (!isModelLoaded) {
            Toast.makeText(this, "请先加载模型", Toast.LENGTH_SHORT).show()
            return
        }

        if (!audioRecorder.isRecording()) {
            // 开始录音
            if (!hasRecordPermission()) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.RECORD_AUDIO),
                    REQUEST_RECORD_AUDIO_PERMISSION
                )
                return
            }

            val success = audioRecorder.start()
            if (success) {
                recordButton.text = getString(R.string.stop)
                textView.text = "🔴 正在录音..."
            } else {
                Toast.makeText(this, "录音启动失败，请检查麦克风权限", Toast.LENGTH_SHORT).show()
            }
        } else {
            // 停止录音 → 开始识别
            recordButton.isEnabled = false
            recordButton.text = "识别中..."
            statusText.text = "正在识别..."

            lifecycleScope.launch(Dispatchers.IO) {
                val audioSamples = audioRecorder.stop()
                val audioDuration = audioSamples.size.toDouble() / 16000.0

                withContext(Dispatchers.Main) {
                    textView.text = "⏳ 识别中...\n音频时长: ${String.format("%.2f", audioDuration)}s\n样本数: ${audioSamples.size}"
                }

                val startTime = System.currentTimeMillis()
                try {
                    val result = runRecognition(audioSamples, 16000)
                    val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                    val rtf = elapsed / audioDuration

                    withContext(Dispatchers.Main) {
                        statusText.text = buildMetricsString(audioDuration, elapsed, rtf)
                        textView.text = buildResultText(
                            audioDuration = audioDuration,
                            elapsed = elapsed,
                            rtf = rtf,
                            text = result,
                        )
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Recognition failed", e)
                    withContext(Dispatchers.Main) {
                        textView.text = "❌ 识别失败:\n${e.message}\n\n${e.stackTraceToString()}"
                        statusText.text = "识别出错"
                    }
                } finally {
                    withContext(Dispatchers.Main) {
                        recordButton.isEnabled = true
                        recordButton.text = "开始录音"
                    }
                }
            }
        }
    }

    /**
     * 执行一次离线识别
     */
    private fun runRecognition(samples: FloatArray, sampleRate: Int): String {
        val rec = recognizer ?: throw IllegalStateException("Recognizer not initialized")

        val stream = rec.createStream()
        try {
            stream.acceptWaveform(samples, sampleRate)
            rec.decode(stream)
            val result = rec.getResult(stream)
            return result.text
        } finally {
            stream.release()
        }
    }

    private fun hasRecordPermission(): Boolean {
        return ActivityCompat.checkSelfPermission(
            this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun buildMetricsString(audioDuration: Double, elapsed: Double, rtf: Double): String {
        return String.format(
            Locale.US,
            "音频: %.2fs | 识别耗时: %.2fs | RTF: %.3f",
            audioDuration, elapsed, rtf
        )
    }

    private fun buildResultText(
        audioDuration: Double,
        elapsed: Double,
        rtf: Double,
        text: String,
    ): String {
        val rtfQuality = when {
            rtf < 0.3 -> "🟢 优秀"
            rtf < 1.0 -> "🟡 可用"
            rtf < 2.0 -> "🟠 较慢"
            else -> "🔴 很慢"
        }

        return buildString {
            appendLine("══════ Qwen3-ASR 0.6B 识别结果 ══════")
            appendLine()
            appendLine("【识别文本】")
            appendLine(if (text.isNotBlank()) text else "(空 — 可能没有检测到有效语音)")
            appendLine()
            appendLine("【性能指标】")
            appendLine("  音频时长: ${String.format("%.2f", audioDuration)}s")
            appendLine("  识别耗时: ${String.format("%.2f", elapsed)}s")
            appendLine("  RTF: ${String.format("%.3f", rtf)}")
            appendLine("  实时率评价: $rtfQuality")
            appendLine()
            appendLine("════════════════════════════════════════════")
        }
    }

    private fun formatModelSize(modelDir: String): String {
        var totalSize = 0L
        File(modelDir).walkTopDown().forEach { file ->
            if (file.isFile) totalSize += file.length()
        }
        return when {
            totalSize >= 1024 * 1024 * 1024 -> String.format("%.1f GB", totalSize / (1024.0 * 1024 * 1024))
            totalSize >= 1024 * 1024 -> String.format("%.0f MB", totalSize / (1024.0 * 1024))
            else -> String.format("%.0f KB", totalSize / 1024.0)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        audioRecorder.release()
        recognizer?.release()
    }
}
