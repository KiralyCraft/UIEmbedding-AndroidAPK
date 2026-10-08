package ro.ubb.uicollector

import org.json.JSONArray
import org.json.JSONObject

/** Paces real capture/resize/inference/spool probes rather than timing the CNN alone. */
class Calibration
{
    private val rates = listOf(0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0)
    private var index = 0
    private var phaseStart = 0L
    private var warmups = 8
    private val timings = mutableListOf<Double>()
    private var fresh = 0
    private var missed = 0
    private var confirming = false
    private var finished = false
    private val results = JSONArray()
    private val passedRates = mutableListOf<Double>()
    var hasSustainableRate = false
        private set
    var selectedFps = 1.0
        private set
    val fps get() = if (confirming) selectedFps else rates[index]
    val description get() = if (confirming) "Sustained calibration at $selectedFps fps" else "Calibration ${index + 1}/${rates.size}: ${rates[index]} fps"

    fun record(startMs: Long, endMs: Long, workMs: Double, wasFresh: Boolean, late: Boolean): Boolean
    {
        if (finished) { return true }
        if (warmups > 0)
        {
            warmups -= 1
            phaseStart = endMs
            return false
        }
        if (phaseStart == 0L) { phaseStart = startMs }
        timings.add(workMs)
        if (wasFresh) { fresh += 1 }
        if (late) { missed += 1 }
        val duration = (endMs - phaseStart).coerceAtLeast(1L)
        val targetDuration = if (confirming) 30000L else 8000L
        if (duration < targetDuration) { return false }
        val p95 = percentile(timings, 0.95)
        val achieved = timings.size * 1000.0 / duration
        val sustainable = p95 <= 750.0 / fps && achieved >= fps * 0.90 && missed.toDouble() / timings.size <= 0.05 && fresh.toDouble() / timings.size >= 0.80
        results.put(JSONObject().put("target_fps", fps).put("duration_ms", duration.toDouble()).put("processed", timings.size).put("fresh_frames", fresh).put("p50_ms", percentile(timings, 0.50)).put("p95_ms", p95).put("achieved_fps", achieved).put("missed_deadlines", missed).put("sustainable", sustainable))
        if (confirming)
        {
            if (sustainable == false)
            {
                val lower = passedRates.lastOrNull { it < selectedFps && it <= 750.0 / p95 }
                if (lower != null)
                {
                    selectedFps = lower
                    timings.clear()
                    fresh = 0
                    missed = 0
                    warmups = 2
                    phaseStart = endMs
                    return false  // The lower rate also needs its own sustained confirmation.
                }
                hasSustainableRate = false
            }
            finished = true
            return true
        }
        if (sustainable) { selectedFps = fps; hasSustainableRate = true; passedRates.add(fps) }
        index += 1
        if (index >= rates.size) { confirming = true }
        timings.clear()
        fresh = 0
        missed = 0
        warmups = 2
        phaseStart = endMs
        return false
    }

    fun report(): JSONArray = results
}
