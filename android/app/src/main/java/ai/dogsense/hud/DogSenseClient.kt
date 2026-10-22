package ai.dogsense.hud

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * WebSocket link to the DogSense GPU server (see server/app.py for the wire format).
 * Frames are sent only when the previous one has been answered, so latency never piles up.
 */
class DogSenseClient(
    private val url: String,
    private val onTelemetry: (Telemetry, Float) -> Unit,  // telemetry, round-trip ms (-1 for audio replies)
    private val onStatus: (String) -> Unit,
) {
    private val http = OkHttpClient.Builder()
        .pingInterval(5, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    private var ws: WebSocket? = null
    private val frameInFlight = AtomicBoolean(false)
    @Volatile private var sentAtNs = 0L
    @Volatile var connected = false
        private set

    fun connect() {
        ws = http.newWebSocket(Request.Builder().url(url).build(), object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                connected = true; frameInFlight.set(false); onStatus("LINK OK")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                // Only a frame reply carries latency_ms; audio-triggered replies must not
                // release the next frame early.
                var rtt = -1f
                if (text.contains("\"latency_ms\"")) {
                    rtt = (System.nanoTime() - sentAtNs) / 1e6f
                    frameInFlight.set(false)
                }
                runCatching { Telemetry.parse(text) }.onSuccess { onTelemetry(it, rtt) }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                connected = false; onStatus("LINK LOST - retrying")
                Thread.sleep(2000); connect()
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                connected = false; onStatus("LINK CLOSED")
            }
        })
    }

    /**
     * 0x03 | f64 t | f32 roll | u16 rotation | jpeg. The server rotates the image clockwise by
     * [rotationDeg] (CameraX rotationDegrees) so it is upright. Returns false if dropped.
     */
    fun sendFrame(jpeg: ByteArray, headRollRad: Float, rotationDeg: Int): Boolean {
        val s = ws ?: return false
        // Watchdog: a reply lost in transit must not stall the stream forever.
        if (frameInFlight.get() && System.nanoTime() - sentAtNs > 2_000_000_000L) frameInFlight.set(false)
        if (!connected || !frameInFlight.compareAndSet(false, true)) return false
        val buf = ByteBuffer.allocate(15 + jpeg.size).order(ByteOrder.LITTLE_ENDIAN)
        buf.put(0x03).putDouble(System.currentTimeMillis() / 1000.0).putFloat(headRollRad)
            .putShort(rotationDeg.toShort()).put(jpeg)
        sentAtNs = System.nanoTime()
        return s.send(buf.array().toByteString()).also { if (!it) frameInFlight.set(false) }
    }

    /** 0x02 | f32 speechProb | PCM16 LE mono 16 kHz. */
    fun sendAudio(pcm: ShortArray, n: Int, speechProb: Float) {
        val s = ws ?: return
        if (!connected) return
        val buf = ByteBuffer.allocate(5 + n * 2).order(ByteOrder.LITTLE_ENDIAN)
        buf.put(0x02).putFloat(speechProb)
        for (i in 0 until n) buf.putShort(pcm[i])
        s.send(ByteString.of(*buf.array()))
    }

    fun close() {
        ws?.close(1000, "bye"); http.dispatcher.executorService.shutdown()
    }
}
