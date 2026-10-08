package ro.ubb.uicollector

import kotlin.math.min

enum class RecordingMode(val title: String, val limit: Double, val fraction: Double, val description: String) {
    MAXIMUM_DETAIL("Maximum Detail", 30.0, 1.0, "Highest calibrated sustainable rate"),
    BALANCED("Balanced", 5.0, 0.5, "Half the calibrated rate, up to 5/s"),
    BATTERY_SAVER("Battery Saver", 1.0, 0.25, "Quarter of the calibrated rate, up to 1/s; batched uploads");
    fun ceiling(calibrated: Double): Double = min(limit, calibrated * fraction)
}

enum class CapturePolicy { ALL_EXCEPT_EXCLUDED, LEGACY_SELECTED }
enum class CaptureSource(val title: String) {
    ACCESSIBILITY("Continuous · resumes after unlock"),
    MEDIA_PROJECTION("Fast session · new consent after lock")
}

data class RecordingSettings(
    val mode: RecordingMode = RecordingMode.MAXIMUM_DETAIL,
    val policy: CapturePolicy = CapturePolicy.ALL_EXCEPT_EXCLUDED,
    val excluded: Set<String> = emptySet(),
    val legacySelected: Set<String> = emptySet(),
    val quotaMb: Long = 512
) {
    fun allows(packageName: String, collectorPackage: String): Boolean =
        packageName !in setOf(collectorPackage, "android", "com.android.systemui") &&
            packageName !in excluded && (policy != CapturePolicy.LEGACY_SELECTED || packageName in legacySelected)
}

enum class GrantState(val label: String) {
    GRANTED("Granted"), MISSING("Missing"), DENIED("Denied"), NEEDS_SETTINGS("Needs settings")
}

data class PermissionState(
    val notifications: GrantState = GrantState.MISSING,
    val accessibility: GrantState = GrantState.MISSING,
    val usageAccess: GrantState = GrantState.MISSING
) {
    val ready: Boolean get() = notifications == GrantState.GRANTED && accessibility == GrantState.GRANTED && usageAccess == GrantState.GRANTED
    val missingDescription: String get() = listOfNotNull(
        "Notifications".takeIf { notifications != GrantState.GRANTED },
        "Accessibility labels".takeIf { accessibility != GrantState.GRANTED },
        "Usage Access".takeIf { usageAccess != GrantState.GRANTED }
    ).joinToString(", ")
}

data class CalibrationState(val active: Boolean = false, val progress: Float = 0f, val description: String = "Not calibrated", val selectedFps: Double = 0.0)

data class RecordingSnapshot(
    val active: Boolean = false,
    val status: String = "Stopped",
    val packageName: String? = null,
    val elapsedSeconds: Long = 0,
    val committed: Long = 0,
    val targetFps: Double = 0.0,
    val achievedFps: Double = 0.0,
    val preprocessMs: Double = 0.0,
    val inferenceMs: Double = 0.0,
    val headMs: Double = 0.0,
    val processingP95Ms: Double = 0.0,
    val queuedSamples: Long = 0,
    val queuedBytes: Long = 0,
    val uploadStatus: String = "Local recording · server not configured",
    val thermalStatus: Int = 0,
    val batteryPercent: Int = -1,
    val charging: Boolean = false,
    val backend: String = "Not initialized",
    val preprocessing: String = "Not initialized",
    val calibration: CalibrationState = CalibrationState(),
    val history: List<Double> = emptyList()
)

/** Commit throughput uses monotonic time, never compositor refresh or target rate. */
class RecordingMetrics {
    private val commits = ArrayDeque<Long>()
    private val durations = ArrayDeque<Double>()
    private val history = ArrayDeque<Double>()
    var total: Long = 0
        private set
    private var historyAt = 0L

    @Synchronized
    fun reset() { commits.clear(); durations.clear(); history.clear(); total = 0; historyAt = 0 }

    @Synchronized
    fun commit(nowNs: Long, workMs: Double) {
        total++
        commits.addLast(nowNs)
        durations.addLast(workMs)
        if (durations.size > 120) durations.removeFirst()
    }

    @Synchronized
    fun snapshot(nowNs: Long): Triple<Double, Double, List<Double>> {
        while (commits.isNotEmpty() && commits.first() <= nowNs - 10_000_000_000L) commits.removeFirst()
        val fps = commits.size / 10.0
        if (nowNs - historyAt >= 1_000_000_000L) {
            history.addLast(fps)
            if (history.size > 60) history.removeFirst()
            historyAt = nowNs
        }
        return Triple(fps, if (durations.isEmpty()) 0.0 else percentile(durations.toList(), .95), history.toList())
    }
}
