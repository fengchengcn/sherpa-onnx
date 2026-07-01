package com.k2fsa.sherpa.onnx.qwen3asr

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/**
 * 封装麦克风录音逻辑。
 * 16kHz, mono, 16-bit PCM。
 * 录音在独立线程中运行，样本累积到内存 buffer 中。
 */
class AudioRecorder(
    private val sampleRate: Int = 16000,
) {
    companion object {
        private const val TAG = "AudioRecorder"
    }

    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    private val isRecording = AtomicBoolean(false)

    // 累积的 PCM 样本 (16-bit short)
    private val audioBuffer = mutableListOf<Short>()

    /**
     * 开始录音。返回 true 表示成功。
     */
    fun start(): Boolean {
        if (isRecording.get()) {
            Log.w(TAG, "Already recording")
            return false
        }

        audioBuffer.clear()

        val minBufSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBufSize == AudioRecord.ERROR || minBufSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(TAG, "Failed to get min buffer size")
            return false
        }

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                minBufSize * 2
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                audioRecord?.release()
                audioRecord = null
                return false
            }

            audioRecord?.startRecording()
            isRecording.set(true)

            recordingThread = Thread {
                val buffer = ShortArray(minBufSize)
                while (isRecording.get()) {
                    val readSize = audioRecord?.read(buffer, 0, buffer.size) ?: -1
                    if (readSize > 0) {
                        synchronized(audioBuffer) {
                            for (i in 0 until readSize) {
                                audioBuffer.add(buffer[i])
                            }
                        }
                    }
                }
            }
            recordingThread?.start()
            Log.i(TAG, "Recording started")
            return true
        } catch (e: SecurityException) {
            Log.e(TAG, "No RECORD_AUDIO permission", e)
            return false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording", e)
            return false
        }
    }

    /**
     * 停止录音，返回累积的 PCM 样本 (FloatArray, 归一化到 [-1, 1])。
     */
    fun stop(): FloatArray {
        isRecording.set(false)

        recordingThread?.join(1000)
        recordingThread = null

        audioRecord?.apply {
            stop()
            release()
        }
        audioRecord = null

        val samples: List<Short>
        synchronized(audioBuffer) {
            samples = audioBuffer.toList()
            audioBuffer.clear()
        }

        Log.i(TAG, "Recording stopped. Total samples: ${samples.size}, " +
                "duration: ${samples.size.toDouble() / sampleRate}s")

        // RMS diagnostic log — matching MNN's Omni Waveform Stats format
        val floats = FloatArray(samples.size) { i -> samples[i] / 32768.0f }
        var sumSq = 0.0
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        var absSum = 0.0
        for (s in floats) {
            sumSq += (s * s).toDouble()
            if (s < minVal) minVal = s
            if (s > maxVal) maxVal = s
            absSum += kotlin.math.abs(s.toDouble())
        }
        val rms = sqrt(sumSq / floats.size)
        val rmsDB = 20.0 * kotlin.math.log10(rms)
        val avgAbs = absSum / floats.size
        Log.i(TAG, "Waveform Stats: samples=${floats.size}, min=%.4f, max=%.4f, avg_abs=%.4f"
            .format(minVal, maxVal, avgAbs))
        Log.i(TAG, "Audio RMS: %.2f dBFS (raw passthrough, no gain)"
            .format(rmsDB))

        return floats
    }

    fun isRecording(): Boolean = isRecording.get()

    fun release() {
        stop()
    }
}
