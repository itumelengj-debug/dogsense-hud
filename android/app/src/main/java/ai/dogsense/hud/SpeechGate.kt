package ai.dogsense.hud

import kotlin.math.sqrt

/**
 * Lightweight on-device "reverse VAD": estimates whether a 100 ms chunk is the wearer talking,
 * so the server can discard it. Swap the body for the Rokid SDK's VAD score if your firmware
 * exposes one - the server only needs a 0..1 probability.
 *
 * Heuristic: the wearer's own voice is loud (mic is centimetres from the mouth) and sustained
 * over several consecutive chunks; barks are short transients.
 */
object SpeechGate {
    private var sustained = 0

    fun speechProb(pcm: ShortArray, n: Int): Float {
        var e = 0.0; var zc = 0
        for (i in 0 until n) {
            val s = pcm[i].toDouble(); e += s * s
            if (i > 0 && (pcm[i] >= 0) != (pcm[i - 1] >= 0)) zc++
        }
        val rms = sqrt(e / n) / 32768.0
        val zcr = zc.toDouble() / n
        // Voiced adult speech: zero-crossing rate ~0.02-0.12 at 16 kHz with high energy.
        val voicedLike = rms > 0.05 && zcr in 0.02..0.12
        sustained = if (voicedLike) sustained + 1 else 0
        return when {
            sustained >= 6 -> 0.9f   // >= 600 ms continuous voicing
            sustained >= 3 -> 0.5f
            else -> 0f
        }
    }
}
