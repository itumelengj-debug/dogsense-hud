package ai.dogsense.hud

import org.json.JSONObject

/** Mirror of the server's telemetry JSON (dogsense/pipeline.py Session.telemetry). */
data class Telemetry(
    val state: String,
    val detail: String,
    val alert: String,
    val trigger: String,
    val cues: List<String>,
    val arousal: Float,
    val valence: Float,
    val confidence: Float,
    val box: FloatArray?,          // x1, y1, x2, y2 normalised 0..1
    val kpts: Array<FloatArray>?,  // 24 x (x, y, conf) normalised
    val latencyMs: Float,
    val morph: String,
) {
    companion object {
        fun parse(s: String): Telemetry {
            val j = JSONObject(s)
            val box = j.optJSONArray("box")?.let { a -> FloatArray(4) { a.getDouble(it).toFloat() } }
            val kp = j.optJSONArray("kpts")?.let { a ->
                Array(a.length()) { i -> a.getJSONArray(i).let { p -> FloatArray(3) { p.getDouble(it).toFloat() } } }
            }
            val cues = j.optJSONArray("cues")?.let { a -> List(a.length()) { a.getString(it) } } ?: emptyList()
            val m = j.optJSONObject("morph")
            return Telemetry(
                state = j.optString("state"), detail = j.optString("detail"),
                alert = j.optString("alert"), trigger = j.optString("trigger"), cues = cues,
                arousal = j.optDouble("arousal", 0.0).toFloat(), valence = j.optDouble("valence", 0.0).toFloat(),
                confidence = j.optDouble("confidence", 0.0).toFloat(), box = box, kpts = kp,
                latencyMs = j.optDouble("latency_ms", 0.0).toFloat(),
                morph = if (m?.optBoolean("calibrated") == true) "${m.optString("ears")} ears / ${m.optString("tail")} tail" else "calibrating",
            )
        }
    }
}
