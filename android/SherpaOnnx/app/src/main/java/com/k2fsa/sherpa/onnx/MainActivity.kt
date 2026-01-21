package com.k2fsa.sherpa.onnx

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.getSystemService
import java.util.Locale
import kotlin.concurrent.thread
import kotlin.math.min

private const val TAG = "sherpa-onnx"
private const val REQUEST_RECORD_AUDIO_PERMISSION = 200

class MainActivity : AppCompatActivity() {
    private val permissions: Array<String> = arrayOf(Manifest.permission.RECORD_AUDIO)

    private lateinit var recognizer: OnlineRecognizer
    private var audioRecord: AudioRecord? = null
    private lateinit var recordButton: Button
    private lateinit var decodeWavButton: Button
    private lateinit var textView: TextView
    private lateinit var statusTextView: TextView
    private lateinit var wavSpinner: Spinner
    private var recordingThread: Thread? = null

    private val sampleRateInHz = 16000
    private var idx: Int = 0
    private var lastText: String = ""

    @Volatile
    private var isRecording: Boolean = false

    @Volatile
    private var isDecoding: Boolean = false

    private var totalTokens = 0
    private var startTime = 0L
    private var audioDuration = 0.0

    private val handler = Handler(Looper.getMainLooper())
    private val statusRunnable = object : Runnable {
        override fun run() {
            statusTextView.text = getSystemStatus()
            handler.postDelayed(this, 500)
        }
    }

    private val modelDir = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
    private val wavFiles = arrayOf("0.wav", "1.wav", "2.wav", "3.wav")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION)

        statusTextView = findViewById(R.id.status_text)
        textView = findViewById(R.id.my_text)
        textView.movementMethod = ScrollingMovementMethod()

        wavSpinner = findViewById(R.id.wav_spinner)
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, wavFiles)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        wavSpinner.adapter = adapter

        recordButton = findViewById(R.id.record_button)
        recordButton.setOnClickListener { onclick() }

        decodeWavButton = findViewById(R.id.decode_wav_button)
        decodeWavButton.setOnClickListener { decodeWav() }

        initModel()
        startStatusTimer()
    }

    private fun startStatusTimer() {
        handler.removeCallbacks(statusRunnable)
        handler.post(statusRunnable)
    }

    private fun getSystemStatus(): String {
        val memInfo = ActivityManager.MemoryInfo()
        val activityManager = getSystemService<ActivityManager>()
        activityManager?.getMemoryInfo(memInfo)
        val availableMem = memInfo.availMem / 1024 / 1024

        val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
        val tps = if ((isRecording || isDecoding) && elapsed > 0) {
            String.format(Locale.US, "%.1f", totalTokens / elapsed)
        } else "0.0"

        val rtf = if (isDecoding && audioDuration > 0 && elapsed > 0) {
            String.format(Locale.US, "%.3f", elapsed / audioDuration)
        } else "N/A"

        // 使用资源占位符消除警告
        return getString(R.string.status_format, availableMem, tps, rtf)
    }

    private fun decodeWav() {
        if (isRecording || isDecoding) return

        val selectedFile = wavSpinner.selectedItem as String
        val wavPath = "$modelDir/test_wavs/$selectedFile"

        thread(true) {
            isDecoding = true
            try {
                val waveData = WaveReader.readWave(application.assets, wavPath)
                audioDuration = waveData.samples.size.toDouble() / waveData.sampleRate

                val stream = recognizer.createStream()
                totalTokens = 0
                startTime = System.currentTimeMillis()

                val chunkSize = (0.02 * waveData.sampleRate).toInt()
                var offset = 0

                while (offset < waveData.samples.size) {
                    val end = min(offset + chunkSize, waveData.samples.size)
                    val chunk = waveData.samples.sliceArray(offset until end)

                    stream.acceptWaveform(chunk, sampleRate = waveData.sampleRate)
                    while (recognizer.isReady(stream)) {
                        recognizer.decode(stream)
                    }

                    val result = recognizer.getResult(stream)
                    totalTokens = result.tokens.size

                    runOnUiThread {
                        // 使用资源占位符
                        textView.text = getString(R.string.decoding_format, selectedFile, result.text)
                    }
                    offset += chunkSize
                }

                val tailPaddings = FloatArray((0.8 * waveData.sampleRate).toInt())
                stream.acceptWaveform(tailPaddings, sampleRate = waveData.sampleRate)
                stream.inputFinished()
                while (recognizer.isReady(stream)) {
                    recognizer.decode(stream)
                }

                val finalResult = recognizer.getResult(stream)
                runOnUiThread {
                    // 使用资源占位符
                    textView.text = getString(R.string.finished_format, selectedFile, finalResult.text)
                }
                stream.release()
            } catch (e: Exception) {
                Log.e(TAG, "Decode failed", e)
            } finally {
                isDecoding = false
            }
        }
    }

    private fun onclick() {
        if (!isRecording) {
            val ret = initMicrophone()
            if (!ret) return
            audioRecord!!.startRecording()
            recordButton.setText(R.string.stop)
            isRecording = true
            textView.text = ""
            lastText = ""
            idx = 0
            totalTokens = 0
            startTime = System.currentTimeMillis()
            recordingThread = thread(true) { processSamples() }
        } else {
            isRecording = false
            audioRecord!!.stop()
            audioRecord!!.release()
            audioRecord = null
            recordButton.setText(R.string.start)
            startTime = 0
        }
    }

    private fun processSamples() {
        val stream = recognizer.createStream()
        val buffer = ShortArray((0.1 * sampleRateInHz).toInt())
        while (isRecording) {
            val ret = audioRecord?.read(buffer, 0, buffer.size)
            if (ret != null && ret > 0) {
                val samples = FloatArray(ret) { buffer[it] / 32768.0f }
                stream.acceptWaveform(samples, sampleRate = sampleRateInHz)
                while (recognizer.isReady(stream)) {
                    recognizer.decode(stream)
                }
                val result = recognizer.getResult(stream)
                totalTokens = result.tokens.size
                var textToDisplay = lastText
                if (result.text.isNotBlank()) {
                    textToDisplay = if (lastText.isBlank()) {
                        getString(R.string.recognized_format, idx, result.text)
                    } else {
                        getString(R.string.recognized_concat_format, lastText, idx, result.text)
                    }
                }
                if (recognizer.isEndpoint(stream)) {
                    recognizer.reset(stream)
                    if (result.text.isNotBlank()) {
                        lastText = getString(R.string.recognized_concat_format, lastText, idx, result.text)
                        textToDisplay = lastText
                        idx++
                    }
                }
                runOnUiThread { textView.text = textToDisplay }
            }
        }
        stream.release()
    }

    private fun initMicrophone(): Boolean {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return false
        val numBytes = AudioRecord.getMinBufferSize(sampleRateInHz, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRateInHz, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, numBytes * 2)
        return true
    }

    private fun initModel() {
        val config = OnlineRecognizerConfig(
            featConfig = getFeatureConfig(sampleRate = sampleRateInHz, featureDim = 80),
            modelConfig = getModelConfig(type = 0)!!,
            enableEndpoint = true,
        )
        recognizer = OnlineRecognizer(assetManager = application.assets, config = config)
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(statusRunnable)
    }
}