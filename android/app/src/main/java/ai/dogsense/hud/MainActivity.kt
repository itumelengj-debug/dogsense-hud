package ai.dogsense.hud

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Size
import android.view.KeyEvent
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import kotlin.concurrent.thread
import kotlin.math.atan2

/**
 * Runs directly on the Rokid Glasses (Android-based) - or on the companion phone when the glasses
 * camera is bridged in - capturing camera + mic, streaming them to the GPU server and rendering
 * the returned telemetry on the waveguide HUD.
 *
 * Temple touchpad / D-pad:
 *   tap (CENTER)          toggle the debug skeleton
 *   long-press (CENTER)   enter HUD calibration; in calibration: tap switches offset/scale,
 *                         LEFT/RIGHT/UP/DOWN adjust, long-press saves and exits, BACK cancels.
 */
class MainActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var hud: HudView
    private lateinit var cal: Calibration
    private lateinit var client: DogSenseClient
    private val camExec = Executors.newSingleThreadExecutor()
    @Volatile private var roll = 0f
    @Volatile private var running = true
    private var lastSent = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cal = Calibration(this)
        cal.applyIntent(intent)
        hud = HudView(this, cal)
        setContentView(hud)
        client = DogSenseClient(BuildConfig.SERVER_URL, { t, rtt ->
            if (rtt >= 0) hud.rttMs = if (hud.rttMs < 0) rtt else 0.8f * hud.rttMs + 0.2f * rtt
            hud.telemetry = t
        }, { hud.status = it })
        val perms = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        if (perms.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) start()
        else requestPermissions(perms, 1)
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, r: IntArray) {
        super.onRequestPermissionsResult(rc, p, r)
        if (r.isNotEmpty() && r.all { it == PackageManager.PERMISSION_GRANTED }) start() else hud.status = "PERMISSION DENIED"
    }

    private fun start() {
        client.connect()
        startCamera()
        startAudio()
        (getSystemService(SENSOR_SERVICE) as SensorManager).let { sm ->
            sm.getDefaultSensor(Sensor.TYPE_GRAVITY)?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        }
    }

    // ---- camera: 720p analysis stream -> JPEG @ <= 15 fps -----------------------------------
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(ResolutionSelector.Builder()
                    .setResolutionStrategy(ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                    .build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(camExec) { img -> onImage(img) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onImage(img: ImageProxy) {
        img.use {
            val now = System.currentTimeMillis()
            if (now - lastSent < 66 || !client.connected) return
            val jpeg = yuvToJpeg(it, 75)
            // Sensor-native frames are sent as-is; the server rotates them upright.
            if (client.sendFrame(jpeg, roll, it.imageInfo.rotationDegrees)) lastSent = now
        }
    }

    private fun yuvToJpeg(img: ImageProxy, quality: Int): ByteArray {
        val y = img.planes[0]; val u = img.planes[1]; val v = img.planes[2]
        val w = img.width; val h = img.height
        val nv21 = ByteArray(w * h * 3 / 2)
        var pos = 0
        val yb = y.buffer
        for (row in 0 until h) { yb.position(row * y.rowStride); yb.get(nv21, pos, w); pos += w }
        val ub = u.buffer; val vb = v.buffer
        for (row in 0 until h / 2) for (col in 0 until w / 2) {
            val i = row * u.rowStride + col * u.pixelStride
            nv21[pos++] = vb.get(row * v.rowStride + col * v.pixelStride); nv21[pos++] = ub.get(i)
        }
        val out = ByteArrayOutputStream()
        YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), quality, out)
        return out.toByteArray()
    }

    // ---- audio: 16 kHz mono PCM16, 100 ms chunks ---------------------------------------------
    @SuppressLint("MissingPermission")
    private fun startAudio() = thread(name = "mic") {
        val sr = 16000
        val min = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        // UNPROCESSED keeps the transients (barks) that voice-tuned AGC/NS would squash.
        val rec = AudioRecord(MediaRecorder.AudioSource.UNPROCESSED, sr, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(min, sr / 5 * 2))
        if (rec.state != AudioRecord.STATE_INITIALIZED) { hud.status = "MIC UNAVAILABLE"; return@thread }
        val buf = ShortArray(sr / 10)
        rec.startRecording()
        while (running) {
            val n = rec.read(buf, 0, buf.size)
            if (n > 0) client.sendAudio(buf, n, SpeechGate.speechProb(buf, n))
        }
        rec.stop(); rec.release()
    }

    // ---- IMU: head roll for ego-motion compensation --------------------------------------
    override fun onSensorChanged(e: SensorEvent) {
        // Landscape device: roll is the gravity vector's angle in the display plane.
        roll = atan2(e.values[1], e.values[0]) - (Math.PI / 2).toFloat()
    }

    override fun onAccuracyChanged(s: Sensor?, a: Int) {}

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        if (cal.applyIntent(intent)) hud.invalidate()  // adb calibration while running
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (isCenter(keyCode)) { event.startTracking(); return true }
        val mode = hud.calibrating ?: return super.onKeyDown(keyCode, event)
        val step = if (mode == "offset") 0.01f else 0.02f
        when (keyCode) {
            KeyEvent.KEYCODE_DPAD_LEFT -> if (mode == "offset") cal.ox -= step else cal.sx -= step
            KeyEvent.KEYCODE_DPAD_RIGHT -> if (mode == "offset") cal.ox += step else cal.sx += step
            KeyEvent.KEYCODE_DPAD_UP -> if (mode == "offset") cal.oy -= step else cal.sy += step
            KeyEvent.KEYCODE_DPAD_DOWN -> if (mode == "offset") cal.oy += step else cal.sy -= step
            KeyEvent.KEYCODE_BACK -> { cal.reload(); hud.calibrating = null }  // discard unsaved changes
            else -> return super.onKeyDown(keyCode, event)
        }
        hud.invalidate(); return true
    }

    override fun onKeyLongPress(keyCode: Int, event: KeyEvent): Boolean {
        if (!isCenter(keyCode)) return super.onKeyLongPress(keyCode, event)
        if (hud.calibrating == null) hud.calibrating = "offset"
        else { cal.save(); hud.calibrating = null }
        return true
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        if (!isCenter(keyCode)) return super.onKeyUp(keyCode, event)
        if (event.isCanceled) return true  // was a long-press
        if (hud.calibrating != null) hud.calibrating = if (hud.calibrating == "offset") "scale" else "offset"
        else { hud.showSkeleton = !hud.showSkeleton; hud.invalidate() }
        return true
    }

    private fun isCenter(k: Int) = k == KeyEvent.KEYCODE_DPAD_CENTER || k == KeyEvent.KEYCODE_ENTER

    override fun onDestroy() {
        running = false
        (getSystemService(SENSOR_SERVICE) as SensorManager).unregisterListener(this)
        client.close(); camExec.shutdown()
        super.onDestroy()
    }
}
