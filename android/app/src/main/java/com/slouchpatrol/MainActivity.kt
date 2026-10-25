package com.slouchpatrol

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.provider.Settings.Global
import android.speech.tts.TextToSpeech
import android.view.OrientationEventListener
import android.view.Surface
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

class MainActivity : ComponentActivity() {

    private val latest = AtomicReference<Bitmap?>(null)
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private lateinit var gemma: GemmaJudge
    private lateinit var harness: Harness
    private var tts: TextToSpeech? = null

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else findViewById<TextView>(R.id.status).text = "Camera permission is required."
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) tts?.language = Locale.US
        }

        gemma = GemmaJudge(this)
        harness = Harness(
            gemma = gemma,
            latestFrame = { latest.get() },
            speak = { text -> tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "slouch") },
            onUpdate = { snap -> runOnUiThread { render(snap) } },
            // The camera starts only after the model is in memory: loading both at once gets the app killed.
            onModelReady = { runOnUiThread { requestCameraThenStart() } },
        )
        // `adb shell am start -n com.slouchpatrol/.MainActivity -e backend GPU|CPU|MIXED` switches and remembers it.
        intent.getStringExtra("backend")?.let {
            getSharedPreferences("settings", MODE_PRIVATE).edit().putString("backend", it.uppercase()).commit()
        }
        // `-e threshold 4` sets the slouch score (0-10) that counts as slouching.
        intent.getStringExtra("threshold")?.toIntOrNull()?.let {
            getSharedPreferences("settings", MODE_PRIVATE).edit().putInt("slouchThreshold", it).commit()
        }
        // `-e votes 2/2` fires when 2 of the last 2 checks say slouching (i.e. 2 in a row).
        intent.getStringExtra("votes")?.split("/")?.mapNotNull { it.trim().toIntOrNull() }?.takeIf { it.size == 2 }?.let {
            getSharedPreferences("settings", MODE_PRIVATE).edit()
                .putInt("voteThreshold", it[0]).putInt("voteWindow", it[1]).commit()
        }
        // `-e steps 2` walks forward exactly 2 times before the hit; `-e steps 0` lets Gemma plan the walk.
        intent.getStringExtra("steps")?.toIntOrNull()?.let {
            getSharedPreferences("settings", MODE_PRIVATE).edit().putInt("fixedSteps", it).commit()
        }
        // `-e target "a bright pink wig"` sets who to watch ("the person wearing ...").
        intent.getStringExtra("target")?.takeIf { it.isNotBlank() }?.let {
            getSharedPreferences("settings", MODE_PRIVATE).edit().putString("target", it).commit()
        }
        // `-e maxfwd 3` caps forward steps in Gemma mode.
        intent.getStringExtra("maxfwd")?.toIntOrNull()?.let {
            getSharedPreferences("settings", MODE_PRIVATE).edit().putInt("maxForward", it).commit()
        }
        // `-e robot 192.168.7.23:5555` sets the SONIC launcher address the same way.
        intent.getStringExtra("robot")?.let {
            getSharedPreferences("settings", MODE_PRIVATE).edit().putString("orinUrl", it).commit()
        }
        harness.settings = loadSettings()
        fillSettingsForm(harness.settings)
        wireButtons()

        harness.start(lifecycleScope)
    }

    private fun requestCameraThenStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            askCamera.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = findViewById<PreviewView>(R.id.preview).surfaceProvider
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            var lastGrab = 0L
            analysis.setAnalyzer(cameraExecutor) { image ->
                val now = System.currentTimeMillis()
                if (now - lastGrab > 200) {
                    lastGrab = now
                    val bitmap = image.toBitmap()
                    val degrees = image.imageInfo.rotationDegrees
                    val upright = if (degrees == 0) bitmap
                    else Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height,
                        Matrix().apply { postRotate(degrees.toFloat()) }, true)
                    latest.set(cropToScreen(upright))
                }
                image.close()
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)

            // The screen is locked to portrait, but the phone may be mounted sideways or upside down on the
            // robot's head. Track how it's physically held so the frames Gemma gets are always upright.
            object : OrientationEventListener(this) {
                override fun onOrientationChanged(degrees: Int) {
                    if (degrees == ORIENTATION_UNKNOWN) return
                    analysis.targetRotation = when (degrees) {
                        in 45 until 135 -> Surface.ROTATION_270
                        in 135 until 225 -> Surface.ROTATION_180
                        in 225 until 315 -> Surface.ROTATION_90
                        else -> Surface.ROTATION_0
                    }
                }
            }.enable()
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * The camera frame is wider than what the preview shows. Crop it to the same centered region,
     * so Gemma sees exactly what's on screen and the person fills more of the image.
     */
    private fun cropToScreen(src: Bitmap): Bitmap {
        val screen = resources.displayMetrics
        val screenAspect = minOf(screen.widthPixels, screen.heightPixels).toFloat() /
            maxOf(screen.widthPixels, screen.heightPixels)
        val portrait = src.height >= src.width
        val (w, h) = if (portrait) {
            minOf(src.width, (src.height * screenAspect).toInt()) to src.height
        } else {
            src.width to minOf(src.height, (src.width * screenAspect).toInt())
        }
        return Bitmap.createBitmap(src, (src.width - w) / 2, (src.height - h) / 2, w, h)
    }

    private fun render(snap: Harness.Snapshot) {
        val airplane = Global.getInt(contentResolver, Global.AIRPLANE_MODE_ON, 0) == 1
        findViewById<TextView>(R.id.status).text = buildString {
            val mode = if (harness.settings.fixedSteps > 0) "fixed ${harness.settings.fixedSteps} steps" else "Gemma walks"
            appendLine("${snap.state}  ·  $mode  ·  strikes ${snap.strikes}  ·  ${gemma.modelName} on ${gemma.backendName}  ·  ${snap.latencyMs} ms")
            appendLine(if (airplane) "Airplane mode ON (on-device only)" else "Airplane mode off")
            appendLine("Target: person wearing ${harness.settings.target}")
            appendLine("Gemma: ${snap.lastRaw.take(120)}")
            append(snap.note)
        }
        findViewById<TextView>(R.id.log).text = snap.log
        snap.sentFrame?.let { findViewById<ImageView>(R.id.sentFrame).setImageBitmap(it) }
        findViewById<Button>(R.id.pause).text = if (snap.state == State.PAUSED) "Resume" else "Pause"
    }

    private fun wireButtons() {
        findViewById<Button>(R.id.reset).setOnClickListener { harness.reset() }
        findViewById<Button>(R.id.pause).setOnClickListener { harness.togglePause() }
        findViewById<Button>(R.id.toggleSettings).setOnClickListener {
            val panel = findViewById<View>(R.id.settingsPanel)
            panel.visibility = if (panel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
        findViewById<Button>(R.id.save).setOnClickListener {
            harness.settings = readSettingsForm()
            saveSettings(harness.settings)
        }
        mapOf(
            R.id.rotateLeft to "rotate_left", R.id.forward to "move_forward", R.id.rotateRight to "rotate_right",
            R.id.tap to "tap", R.id.shake to "shake", R.id.slap to "slap",
        ).forEach { (id, motion) -> findViewById<Button>(id).setOnClickListener { harness.testMotion(motion) } }
        findViewById<Button>(R.id.forgive).setOnClickListener { harness.forgive() }
        findViewById<Button>(R.id.testVoice).setOnClickListener {
            tts?.speak("You need to stop slouching.", TextToSpeech.QUEUE_FLUSH, null, "test")
        }
        findViewById<Button>(R.id.ping).setOnClickListener {
            lifecycleScope.launch {
                val reply = withContext(Dispatchers.IO) { RobotLink.health(harness.settings.orinUrl) }
                findViewById<TextView>(R.id.log).append("\nping: $reply")
            }
        }
    }

    private fun loadSettings(): Settings {
        val p = getSharedPreferences("settings", MODE_PRIVATE)
        val d = Settings()
        return Settings(
            orinUrl = p.getString("orinUrl", d.orinUrl)!!,
            imagesPerCall = p.getInt("imagesPerCall", d.imagesPerCall),
            voteWindow = p.getInt("voteWindow", d.voteWindow),
            voteThreshold = p.getInt("voteThreshold", d.voteThreshold),
            ttsDelayMs = p.getLong("ttsDelayMs", d.ttsDelayMs),
            extraRotation = p.getInt("extraRotation", d.extraRotation),
            backend = p.getString("backend", d.backend)!!,
            slouchThreshold = p.getInt("slouchThreshold", d.slouchThreshold),
            fixedSteps = p.getInt("fixedSteps", d.fixedSteps),
            maxForward = p.getInt("maxForward", d.maxForward),
            target = p.getString("target", d.target)!!,
        )
    }

    private fun saveSettings(s: Settings) {
        getSharedPreferences("settings", MODE_PRIVATE).edit()
            .putString("orinUrl", s.orinUrl)
            .putInt("imagesPerCall", s.imagesPerCall)
            .putInt("voteWindow", s.voteWindow)
            .putInt("voteThreshold", s.voteThreshold)
            .putLong("ttsDelayMs", s.ttsDelayMs)
            .putInt("extraRotation", s.extraRotation)
            .putInt("fixedSteps", s.fixedSteps)
            .apply()
    }

    private fun fillSettingsForm(s: Settings) {
        findViewById<EditText>(R.id.orinUrl).setText(s.orinUrl)
        findViewById<EditText>(R.id.imagesPerCall).setText(s.imagesPerCall.toString())
        findViewById<EditText>(R.id.voteThreshold).setText(s.voteThreshold.toString())
        findViewById<EditText>(R.id.voteWindow).setText(s.voteWindow.toString())
        findViewById<EditText>(R.id.ttsDelay).setText(s.ttsDelayMs.toString())
        findViewById<EditText>(R.id.rotation).setText(s.extraRotation.toString())
        findViewById<EditText>(R.id.fixedSteps).setText(s.fixedSteps.toString())
    }

    private fun readSettingsForm(): Settings {
        fun int(id: Int, fallback: Int) = findViewById<EditText>(id).text.toString().toIntOrNull() ?: fallback
        val d = harness.settings
        return d.copy(
            orinUrl = findViewById<EditText>(R.id.orinUrl).text.toString().ifBlank { d.orinUrl },
            imagesPerCall = int(R.id.imagesPerCall, d.imagesPerCall).coerceIn(1, Harness.MAX_IMAGES),
            voteThreshold = int(R.id.voteThreshold, d.voteThreshold).coerceAtLeast(1),
            voteWindow = int(R.id.voteWindow, d.voteWindow).coerceAtLeast(1),
            ttsDelayMs = int(R.id.ttsDelay, d.ttsDelayMs.toInt()).toLong(),
            extraRotation = int(R.id.rotation, d.extraRotation),
            fixedSteps = int(R.id.fixedSteps, d.fixedSteps).coerceIn(0, 10),
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.shutdown()
        cameraExecutor.shutdown()
        gemma.close()
    }
}
