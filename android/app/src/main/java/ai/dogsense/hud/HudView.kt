package ai.dogsense.hud

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.view.View
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Monochrome-green waveguide HUD. Black pixels are transparent on the Micro-LED optics, so the
 * view only ever draws green at varying weight/intensity. Urgency is carried by geometry
 * (thicker brackets, boxed caption, blinking) instead of colour.
 */
class HudView(context: Context, private val cal: Calibration) : View(context) {
    private val green = Color.rgb(60, 255, 60)
    private val dim = Color.rgb(25, 110, 25)
    private val mono = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = green; style = Paint.Style.STROKE; strokeCap = Paint.Cap.SQUARE }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = green; typeface = mono }
    private val small = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = dim; typeface = mono }
    private val bone = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = dim; strokeWidth = 2f }

    @Volatile var telemetry: Telemetry? = null
        set(v) { field = v; postInvalidateOnAnimation() }
    @Volatile var status: String = "CONNECTING"
        set(v) { field = v; postInvalidateOnAnimation() }
    var showSkeleton = false
    @Volatile var rttMs = -1f
    /** null = normal; "offset" / "scale" = calibration mode adjusting that parameter. */
    var calibrating: String? = null
        set(v) { field = v; postInvalidateOnAnimation() }

    private val edges = intArrayOf(
        0, 1, 1, 2, 6, 7, 7, 8, 3, 4, 4, 5, 9, 10, 10, 11, 2, 22, 8, 22, 5, 12, 11, 12, 22, 12, 12, 13,
        22, 23, 23, 17, 17, 16, 16, 20, 16, 21, 20, 14, 21, 15, 14, 18, 15, 19,
    )

    override fun onDraw(c: Canvas) {
        c.drawColor(Color.BLACK)
        val w = width.toFloat(); val h = height.toFloat()
        val u = h / 480f
        text.textSize = 22 * u; small.textSize = 16 * u
        val t = telemetry
        calibrating?.let { drawCalibration(c, w, h, u, it, t) ; return }
        if (t == null) {
            c.drawText("DOGSENSE  $status", 20 * u, 40 * u, text); return
        }
        val blinkOn = (System.currentTimeMillis() / 300) % 2 == 0L

        // Target reticle: corner brackets only (keeps the dog itself unobstructed).
        t.box?.let { b ->
            val x1 = cal.x(b[0]) * w; val y1 = cal.y(b[1]) * h; val x2 = cal.x(b[2]) * w; val y2 = cal.y(b[3]) * h
            val L = ((x2 - x1) / 6).coerceAtLeast(14 * u)
            line.strokeWidth = (if (t.alert.isNotEmpty()) 6f else 3f) * u
            if (t.alert.isEmpty() || blinkOn) {
                for ((cx, cy, sx, sy) in listOf(
                    floatArrayOf(x1, y1, 1f, 1f), floatArrayOf(x2, y1, -1f, 1f),
                    floatArrayOf(x1, y2, 1f, -1f), floatArrayOf(x2, y2, -1f, -1f))) {
                    c.drawLine(cx, cy, cx + sx * L, cy, line); c.drawLine(cx, cy, cx, cy + sy * L, line)
                }
            }
            if (showSkeleton) t.kpts?.let { k ->
                for (i in edges.indices step 2) {
                    val a = k[edges[i]]; val bb = k[edges[i + 1]]
                    if (a[2] > 0.3f && bb[2] > 0.3f) c.drawLine(cal.x(a[0]) * w, cal.y(a[1]) * h, cal.x(bb[0]) * w, cal.y(bb[1]) * h, bone)
                }
            }
        }

        // Gauges (top-left).
        val pct = (t.arousal * 100).roundToInt()
        c.drawText("AROUSAL ${bar(t.arousal)} ${pct.toString().padStart(3)}%", 20 * u, 36 * u, text)
        c.drawText("VALENCE ${valenceBar(t.valence)}", 20 * u, 64 * u, text)
        if (t.trigger.isNotEmpty()) c.drawText("TRIGGER: ${t.trigger}", 20 * u, 92 * u, small)
        val rtt = if (rttMs >= 0) "${rttMs.roundToInt()}ms rtt" else "--"
        c.drawText("${t.morph}  $rtt  $status", 20 * u, h - 110 * u, small)

        // Caption block (bottom).
        val top = h - 90 * u
        if (t.alert.isNotEmpty()) {
            line.strokeWidth = 3 * u
            c.drawRect(10 * u, top, w - 10 * u, h - 10 * u, line)
            c.drawText("ALERT: ${t.state}", 24 * u, top + 30 * u, text)
            c.drawText(t.alert, 24 * u, top + 56 * u, text)
        } else {
            val d = if (t.detail.isNotEmpty()) " (${t.detail})" else ""
            c.drawText(fit("STATE: ${t.state}$d", w - 40 * u, text), 24 * u, top + 30 * u, text)
        }
        c.drawText(fit(t.cues.joinToString(" | "), w - 40 * u, small), 24 * u, h - 20 * u, small)
        if (t.alert.isNotEmpty()) postInvalidateDelayed(300)
    }

    /**
     * Calibration screen: the outline of the full camera frame, a crosshair at the camera
     * centre and the live dog box, all through the current mapping. Look at a real dog
     * (or any object at the camera centre) and nudge until the brackets sit on it.
     */
    private fun drawCalibration(c: Canvas, w: Float, h: Float, u: Float, mode: String, t: Telemetry?) {
        line.strokeWidth = 2 * u
        c.drawRect(cal.x(0f) * w, cal.y(0f) * h, cal.x(1f) * w, cal.y(1f) * h, line)
        val cx = cal.x(0.5f) * w; val cy = cal.y(0.5f) * h
        c.drawLine(cx - 30 * u, cy, cx + 30 * u, cy, line); c.drawLine(cx, cy - 30 * u, cx, cy + 30 * u, line)
        t?.box?.let { b ->
            line.strokeWidth = 4 * u
            c.drawRect(cal.x(b[0]) * w, cal.y(b[1]) * h, cal.x(b[2]) * w, cal.y(b[3]) * h, line)
        }
        c.drawText("CALIBRATE ${mode.uppercase()}", 20 * u, 36 * u, text)
        c.drawText(cal.toString(), 20 * u, 62 * u, small)
        c.drawText("L/R/U/D adjust · tap: offset/scale · long-press: save", 20 * u, h - 20 * u, small)
    }

    private fun bar(v: Float, n: Int = 10): String {
        val k = (v.coerceIn(0f, 1f) * n).roundToInt()
        return "[" + "|".repeat(k) + ".".repeat(n - k) + "]"
    }

    private fun valenceBar(v: Float, n: Int = 3): String {
        val k = (abs(v.coerceIn(-1f, 1f)) * n).roundToInt()
        val neg = if (v < 0) "- ".repeat(k) else ""
        val pos = if (v > 0) " +".repeat(k) else ""
        return "[" + neg.padStart(2 * n) + "|" + pos.padEnd(2 * n) + "]"
    }

    private fun fit(s: String, maxW: Float, p: Paint): String {
        var out = s
        while (out.length > 4 && p.measureText(out) > maxW) out = out.dropLast(2)
        return if (out.length < s.length) out.dropLast(1) + "â€¦" else out
    }
}
