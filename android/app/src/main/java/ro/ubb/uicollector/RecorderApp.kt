package ro.ubb.uicollector

import android.app.Application
import android.content.Context
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import android.os.BatteryManager
import android.os.SystemClock
import java.util.UUID
import java.util.concurrent.TimeUnit

class RecorderApp : Application()
{
    lateinit var vault: Vault
        private set
    lateinit var store: LocalStore
        private set
    lateinit var labels: LabelTracker
        private set
    lateinit var uploader: Uploader
        private set
    @Volatile var captureActive = false
    @Volatile var accessibilityService: LabelAccessibilityService? = null
    @Volatile var calibrating = false
    @Volatile var captureStatus = "Stopped"
    @Volatile var uploadStatus = "Local recording · server not configured"
    @Volatile var currentFps = 0.0
    @Volatile var accountOperation = false
    val metrics = RecordingMetrics()
    private val mutableRecording = MutableStateFlow(RecordingSnapshot())
    val recording = mutableRecording.asStateFlow()
    @Volatile var sessionStartedNs = 0L
    @Volatile var sessionStoppedNs = 0L
    @Volatile var activePackage: String? = null
    @Volatile var latestPreprocessMs = 0.0
    @Volatile var latestInferenceMs = 0.0
    @Volatile var latestHeadMs = 0.0
    @Volatile var thermalStatus = 0
    @Volatile var processingBackend = "Not initialized"
    @Volatile var preprocessingBackend = "Not initialized"
    @Volatile var calibrationState = CalibrationState()
    val localOwner: String get() = "local:$deviceId"
    fun captureOwner(): String = vault.identity()?.let { ownerKey(it) } ?: localOwner
    fun captureSource(): CaptureSource = runCatching { CaptureSource.valueOf(preferences.getString("capture_source",CaptureSource.ACCESSIBILITY.name)!!) }.getOrDefault(CaptureSource.ACCESSIBILITY)
    fun resumeRequestedRecording() {
        if(!preferences.getBoolean("continuous_recording_requested",false) || captureSource()!=CaptureSource.ACCESSIBILITY || captureActive) return
        if(!getSystemService(android.os.UserManager::class.java).isUserUnlocked || accessibilityService==null || !Permissions.inspect(this).ready) return
        if(accessibilityService!!.serviceInfo.capabilities and android.accessibilityservice.AccessibilityServiceInfo.CAPABILITY_CAN_TAKE_SCREENSHOT == 0) {
            captureStatus="Re-enable the collector's Accessibility service to load screen capture support"
            return
        }
        runCatching { startForegroundService(android.content.Intent(this,CaptureService::class.java)) }
            .onFailure { captureStatus="Open the collector to resume: ${it.message}" }
    }

    fun settings(): RecordingSettings {
        val old = preferences.getString("allowed_packages", "").orEmpty().split(',', '\n').map { it.trim() }.filter { it.isNotEmpty() }.toSet()
        return RecordingSettings(
            mode = runCatching { RecordingMode.valueOf(preferences.getString("recording_mode", RecordingMode.MAXIMUM_DETAIL.name)!!) }.getOrDefault(RecordingMode.MAXIMUM_DETAIL),
            policy = runCatching { CapturePolicy.valueOf(preferences.getString("capture_policy", if (old.isEmpty()) CapturePolicy.ALL_EXCEPT_EXCLUDED.name else CapturePolicy.LEGACY_SELECTED.name)!!) }.getOrDefault(CapturePolicy.ALL_EXCEPT_EXCLUDED),
            excluded = preferences.getStringSet("excluded_packages", emptySet()).orEmpty().toSet(),
            legacySelected = old,
            quotaMb = preferences.getLong("queue_quota_mb", 512L)
        )
    }

    fun calibrationSignature(backend: String = preferences.getString("encoder_backend", "litert_opencl_full_encoder")!!): String {
        val manifest = org.json.JSONObject(assets.open("f6_manifest.json").bufferedReader().use { it.readText() })
        return manifest.getString("model_id") + ":" + backend + ":preprocessing_v2:adaptive_capture_v1:" + captureSource().name + ":" + preferences.getString("selected_preprocessor","opencv_cpu") + ":" + preferences.getString("gpu_driver","unknown") + ":" + android.os.Build.FINGERPRINT
    }
    fun needsCalibration(): Boolean = runCatching { preferences.getString("calibrated_signature", "") != calibrationSignature() }.getOrDefault(true)

    fun refreshSnapshot() {
        val now = SystemClock.elapsedRealtimeNanos()
        val (fps, p95, history) = metrics.snapshot(now)
        val queue = store.queueStats()
        val battery = getSystemService(BatteryManager::class.java)
        mutableRecording.value = RecordingSnapshot(captureActive, captureStatus, activePackage,
            if (sessionStartedNs == 0L) 0 else ((if(captureActive || sessionStoppedNs==0L) now else sessionStoppedNs) - sessionStartedNs) / 1_000_000_000L,
            metrics.total, currentFps, fps, latestPreprocessMs, latestInferenceMs, latestHeadMs, p95,
            queue.second, queue.first, uploadStatus, thermalStatus, battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY),
            battery.isCharging, processingBackend, preprocessingBackend, calibrationState, history)
    }

    val preferences by lazy { getSharedPreferences("settings", Context.MODE_PRIVATE) }
    val deviceId: String by lazy {
        preferences.getString("device_id", null) ?: UUID.randomUUID().toString().also { preferences.edit().putString("device_id", it).commit() }
    }

    override fun onCreate()
    {
        super.onCreate()
        Permissions.createChannel(this)
        vault = Vault(this)
        store = LocalStore(this, vault)
        store.recoverInterruptedRuns()
        labels = LabelTracker(this)
        uploader = Uploader(this)
        val constraints = Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()
        val periodic = PeriodicWorkRequestBuilder<UploadWorker>(15, TimeUnit.MINUTES).setConstraints(constraints).build()
        WorkManager.getInstance(this).enqueueUniquePeriodicWork("embedding-periodic-upload", ExistingPeriodicWorkPolicy.KEEP, periodic)
    }

    fun requestUpload()
    {
        val constraints = Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()
        val work = OneTimeWorkRequestBuilder<UploadWorker>().setConstraints(constraints).build()
        WorkManager.getInstance(this).enqueueUniqueWork("embedding-upload-now", ExistingWorkPolicy.KEEP, work)
    }
}
