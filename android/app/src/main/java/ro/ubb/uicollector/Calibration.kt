package ro.ubb.uicollector

import org.json.JSONArray
import org.json.JSONObject

/** Paces real capture/resize/inference/spool probes rather than timing the CNN alone. */
class Calibration(maximumFps: Double = 30.0)
{
    private val rates = (listOf(0.25, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0).filter { it<=maximumFps } + maximumFps).distinct().sorted()
    private var index = 0
    private var phaseStart = 0L
    private var warmups = 2
    private val timings = mutableListOf<Double>()
    private var fresh = 0
    private var missed = 0
    private var confirming = false
    private var finished = false
    data class ProbeResult(val targetFps: Double, val durationMs: Long, val attempts: Int,
        val freshFrames: Int, val p50Ms: Double, val p95Ms: Double, val achievedFps: Double,
        val missedDeadlines: Int, val sustainable: Boolean, val rateLimited: Boolean, val confirmation: Boolean)
    val results = mutableListOf<ProbeResult>()
    private val passedRates = mutableListOf<Double>()
    var hasSustainableRate = false
        private set
    var selectedFps = 1.0
        private set
    val fps get() = if (confirming) selectedFps else rates[index]
    val description get() = if (confirming) "Sustained calibration at $selectedFps fps" else "Calibration ${index + 1}/${rates.size}: ${rates[index]} fps"

    val progress: Float get() = if(finished) 1f else if(confirming) .9f else index * .9f / rates.size

    fun restartWindow() { timings.clear();fresh=0;missed=0;phaseStart=0L;warmups=2 }

    fun record(startMs: Long, endMs: Long, workMs: Double, wasFresh: Boolean, late: Boolean, rateLimited: Boolean = false): Boolean
    {
        if (finished) { return true }
        if (warmups > 0 && !rateLimited)
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
        if (duration < targetDuration && !rateLimited) { return false }
        val p95 = percentile(timings, 0.95)
        val achieved = fresh * 1000.0 / duration
        val sustainable = !rateLimited && p95 <= 750.0 / fps && achieved >= fps * 0.90 && missed.toDouble() / timings.size <= 0.05 && fresh.toDouble() / timings.size >= 0.80
        results.add(ProbeResult(fps,duration,timings.size,fresh,percentile(timings,0.50),p95,achieved,missed,sustainable,rateLimited,confirming))
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
        if (!sustainable || index >= rates.size) {
            // Do not keep requesting faster screenshots once this device rejects a rate.
            if (!hasSustainableRate) { finished=true;return true }
            confirming = true
        }
        timings.clear()
        fresh = 0
        missed = 0
        warmups = 2
        phaseStart = endMs
        return false
    }

    fun report(): JSONArray = JSONArray().apply {
        results.forEach { row -> put(JSONObject().put("target_fps",row.targetFps).put("duration_ms",row.durationMs)
            .put("attempts",row.attempts).put("processed",row.freshFrames).put("fresh_frames",row.freshFrames)
            .put("p50_ms",row.p50Ms).put("p95_ms",row.p95Ms).put("achieved_fps",row.achievedFps)
            .put("missed_deadlines",row.missedDeadlines).put("sustainable",row.sustainable)
            .put("rate_limited",row.rateLimited).put("confirmation",row.confirmation)) }
    }
}
