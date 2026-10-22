package ai.dogsense.hud

import android.content.Context
import android.content.Intent

/**
 * Maps camera-image coordinates (0..1) onto the waveguide view (0..1).
 *
 * The RV101's camera sees a much wider field than the see-through display covers, and the two
 * are offset, so a box drawn at raw camera coordinates lands beside the dog. The mapping is
 *   view = (camera - 0.5) * scale + 0.5 + offset
 * per axis, tuned once per wearer/fit and stored on the device.
 *
 * Adjust on the glasses (calibration mode, see MainActivity) or over adb:
 *   adb shell am start -n ai.dogsense.hud/.MainActivity --ef sx 2.2 --ef sy 2.2 --ef ox 0.0 --ef oy -0.05
 */
class Calibration(context: Context) {
    private val prefs = context.getSharedPreferences("hud_calibration", Context.MODE_PRIVATE)
    var sx = prefs.getFloat("sx", 1f)
    var sy = prefs.getFloat("sy", 1f)
    var ox = prefs.getFloat("ox", 0f)
    var oy = prefs.getFloat("oy", 0f)

    fun x(cam: Float) = (cam - 0.5f) * sx + 0.5f + ox
    fun y(cam: Float) = (cam - 0.5f) * sy + 0.5f + oy

    fun save() {
        prefs.edit().putFloat("sx", sx).putFloat("sy", sy).putFloat("ox", ox).putFloat("oy", oy).apply()
    }

    /** Applies any of sx/sy/ox/oy passed as float extras; returns true if something changed. */
    fun applyIntent(i: Intent?): Boolean {
        if (i == null) return false
        var changed = false
        for (key in listOf("sx", "sy", "ox", "oy")) {
            if (i.hasExtra(key)) {
                val v = i.getFloatExtra(key, 0f)
                when (key) { "sx" -> sx = v; "sy" -> sy = v; "ox" -> ox = v; "oy" -> oy = v }
                changed = true
            }
        }
        if (changed) save()
        return changed
    }

    fun reload() {
        sx = prefs.getFloat("sx", 1f); sy = prefs.getFloat("sy", 1f)
        ox = prefs.getFloat("ox", 0f); oy = prefs.getFloat("oy", 0f)
    }

    override fun toString() = "scale %.2f×%.2f  offset %+.2f,%+.2f".format(sx, sy, ox, oy)
}
