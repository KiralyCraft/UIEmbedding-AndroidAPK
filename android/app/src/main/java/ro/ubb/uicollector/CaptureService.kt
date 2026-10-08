package ro.ubb.uicollector

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Base64
import android.view.WindowManager
import androidx.core.content.ContextCompat
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.ceil

class CaptureService : Service()
{
    private val app get() = application as RecorderApp
    private lateinit var thread: HandlerThread
    private lateinit var handler: Handler
    private val maintenance = Executors.newScheduledThreadPool(2)
    private var projection: MediaProjection? = null
    private var display: VirtualDisplay? = null
    private var reader: ImageReader? = null
    private var accessibilityCapture: AccessibilityCapture? = null
    private var source = CaptureSource.ACCESSIBILITY
    private var preprocessor: CapturePreprocessor? = null
    private var engine: EmbeddingEngine? = null
    private var calibration: Calibration? = null
    private var rate = RateController(1.0)
    private var width = 0
    private var height = 0
    private var dueNs = 0L
    private var lastNotificationMs = 0L
    private var lastGeneration = -1L
    private var lastImageTimestamp = -1L
    private var pendingFrame = false
    private var pendingAcquisitionNs = 0L
    private var pendingAcquisitionWall = 0L
    private var runId: String? = null
    private var runPackage: String? = null
    private var sequence = 0
    private var owner = ""
    private var captureSession = ""
    private var quotaPaused = false
    private var observedSettings = RecordingSettings()
    private var nextUploadNs = 0L
    private var paused = false
    private var lastPermissionCheckNs = 0L
    @Volatile private var stopping = false

    private val screenOff = object : BroadcastReceiver()
    {
        override fun onReceive(context: Context?, intent: Intent?)
        {
            if (intent?.action == Intent.ACTION_SCREEN_OFF)
            {
                if(source==CaptureSource.ACCESSIBILITY) {
                    handler.post { closeRun("screen_locked");preprocessor?.clear();pendingFrame=false;accessibilityCapture?.reset();calibration?.restartWindow() }
                    app.captureStatus="Screen locked/off · recording resumes after unlock"
                    return
                }
                app.captureStatus = "Screen locked/off. Start capture again after unlocking. Queued data is retained."
                stopSelf()
            }
        }
    }

    override fun onCreate()
    {
        super.onCreate()
        thread = HandlerThread("F6-GPU-capture")
        thread.start()
        handler = Handler(thread.looper)
        getSystemService(NotificationManager::class.java).createNotificationChannel(NotificationChannel("capture", "Continuous embedding capture", NotificationManager.IMPORTANCE_LOW))
        ContextCompat.registerReceiver(this, screenOff, IntentFilter(Intent.ACTION_SCREEN_OFF), ContextCompat.RECEIVER_NOT_EXPORTED)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int
    {
        if (intent?.action == "STOP")
        {
            app.preferences.edit().putBoolean("continuous_recording_requested",false).apply()
            app.captureStatus = "Stopped by user; pending data will continue uploading"
            stopSelf()
            return START_NOT_STICKY
        }
        if (app.captureActive) { return if(source==CaptureSource.ACCESSIBILITY) START_STICKY else START_NOT_STICKY }
        source=app.captureSource()
        if(intent==null && (source!=CaptureSource.ACCESSIBILITY || !app.preferences.getBoolean("continuous_recording_requested",false))) { stopSelf();return START_NOT_STICKY }
        val readiness=Permissions.inspect(app)
        if(!readiness.ready || app.accountOperation) {
            app.captureStatus="Required permissions are incomplete: ${readiness.missingDescription}"
            stopSelf()
            return START_NOT_STICKY
        }
        startForeground(1, notification("Initializing GPU; screen capture is visible and user-controlled"), if(source==CaptureSource.MEDIA_PROJECTION) ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION else if(Build.VERSION.SDK_INT>=34) ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE else 0)
        if(source==CaptureSource.ACCESSIBILITY) app.preferences.edit().putBoolean("continuous_recording_requested",true).apply()
        app.captureStatus = "Initializing GPU and validating the trained model"
        app.calibrating = intent?.getBooleanExtra("calibrate", false) ?: false
        app.captureActive = true
        handler.post {
            try
            {
                check(Permissions.inspect(app).ready && !app.accountOperation) { "Required permissions changed before capture" }
                owner = app.captureOwner()
                observedSettings=app.settings()
                app.metrics.reset()
                app.sessionStartedNs=SystemClock.elapsedRealtimeNanos()
                app.sessionStoppedNs=0L
                captureSession = UUID.randomUUID().toString()
                check(OpenCVLoader.initLocal()) { "OpenCV initialization failed" }
                engine = EmbeddingEngine.create(this)
                app.preferences.edit().putString("encoder_backend",engine!!.backend).apply()
                app.processingBackend=engine!!.backend
                preprocessor = CapturePreprocessor(app,engine!!)
                if (stopping) { return@post }
                calibration = if (intent?.getBooleanExtra("calibrate", false)==true || app.needsCalibration()) Calibration(if(source==CaptureSource.ACCESSIBILITY) 2.5 else 30.0) else null
                app.calibrating=calibration!=null
                app.calibrationState=CalibrationState(active=app.calibrating,description=if(app.calibrating) "Initializing calibration" else "Saved calibration")
                if (calibration == null)
                {
                    check(app.preferences.getString("calibrated_model", "") == engine!!.modelId) { "Run GPU calibration for this model first" }
                    rate = RateController(observedSettings.mode.ceiling(app.preferences.getFloat("calibrated_fps", 1.0f).toDouble()))
                }
                if(source==CaptureSource.ACCESSIBILITY) {
                    accessibilityCapture=AccessibilityCapture(app,handler)
                } else {
                @Suppress("DEPRECATION")
                val permission = intent?.getParcelableExtra<Intent>("permission") ?: error("Missing fresh screen capture consent")
                projection = getSystemService(MediaProjectionManager::class.java).getMediaProjection(intent.getIntExtra("result_code", 0), permission)
                projection!!.registerCallback(object : MediaProjection.Callback()
                {
                    override fun onStop()
                    {
                        if (stopping == false)
                        {
                            app.captureStatus = "Android stopped projection. Start capture again to grant fresh consent."
                            stopSelf()
                        }
                    }
                    override fun onCapturedContentResize(newWidth: Int, newHeight: Int)
                    {
                        if (stopping == false && newWidth > 0 && newHeight > 0 && (newWidth != width || newHeight != height))
                        {
                            resizeCapture(newWidth, newHeight)
                        }
                    }
                    override fun onCapturedContentVisibilityChanged(isVisible: Boolean)
                    {
                        if (isVisible == false && stopping == false)
                        {
                            app.captureStatus = "Captured content became hidden. Use entire-display capture, not a single app."
                            stopSelf()
                        }
                    }
                }, handler)
                val bounds = getSystemService(WindowManager::class.java).maximumWindowMetrics.bounds
                width = bounds.width()
                height = bounds.height()
                reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 3)
                // Exactly ONE virtual display per consent token. Rotation resizes this object.
                display = projection!!.createVirtualDisplay("UIEmbeddingCapture", width, height, resources.configuration.densityDpi, DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, reader!!.surface, null, handler)
                }
                maintenance.scheduleWithFixedDelay({ runCatching { app.labels.pollUsage() } }, 0, 200, TimeUnit.MILLISECONDS)
                maintenance.scheduleWithFixedDelay({
                    val now=SystemClock.elapsedRealtimeNanos()
                    if(now>=nextUploadNs) {
                        nextUploadNs=now+if(app.settings().mode==RecordingMode.BATTERY_SAVER) 30_000_000_000L else 10_000_000_000L
                        app.uploader.flush()
                    }
                }, 0, 1, TimeUnit.SECONDS)
                dueNs = SystemClock.elapsedRealtimeNanos()
                tick()
            }
            catch (error: Throwable)
            {
                app.captureStatus = "Capture could not start: ${error.message}; ${error.cause?.message.orEmpty()}"
                stopSelf()
            }
        }
        return if(source==CaptureSource.ACCESSIBILITY) START_STICKY else START_NOT_STICKY
    }

    private fun resizeCapture(newWidth: Int, newHeight: Int)
    {
        if (display == null) { return }
        closeRun("display_resized")
        val replacement = ImageReader.newInstance(newWidth, newHeight, PixelFormat.RGBA_8888, 3)
        display!!.surface = null
        display!!.resize(newWidth, newHeight, resources.configuration.densityDpi)
        display!!.surface = replacement.surface
        reader?.close()
        reader = replacement
        width = newWidth
        height = newHeight
        preprocessor?.clear()
        lastImageTimestamp = -1L
        lastGeneration = -1L
    }

    private fun tick()
    {
        if(stopping) return
        val capture=accessibilityCapture
        if(capture==null) { processTick();return }
        val started=SystemClock.elapsedRealtimeNanos()
        val before=if(calibration==null) app.labels.snapshot() else null
        if(!Permissions.inspect(app).ready) {
            app.preferences.edit().putBoolean("continuous_recording_requested",false).apply()
            app.captureStatus="Recording stopped: a required permission was revoked"
            stopSelf();return
        }
        if(!capture.unlocked() || (calibration==null && (before==null || !allowed(before.packageName)))) {
            closeRun(if(!capture.unlocked()) "screen_locked" else "ambiguous_or_excluded_app")
            preprocessor?.clear();pendingFrame=false;capture.reset()
            app.captureStatus=if(!capture.unlocked()) "Screen locked/off · recording resumes after unlock" else "Waiting: ${app.labels.status}"
            app.refreshSnapshot()
            getSystemService(NotificationManager::class.java).notify(1,notification(app.captureStatus))
            handler.postDelayed({ tick() },1000)
            return
        }
        capture.request(calibration!=null) { frame,error ->
            if(stopping) return@request
            val after=if(calibration==null) app.labels.snapshot() else null
            if(error!=null || (calibration==null && (after==null || after.generation!=before?.generation))) {
                closeRun("capture_unavailable_or_app_changed");preprocessor?.clear();pendingFrame=false
                app.captureStatus=error ?: "Foreground changed during capture · waiting"
                handler.postDelayed({ tick() },1000)
            } else processTick(frame,started)
        }
    }

    private fun processTick(accessibilityFrame: CapturedFrame? = null, captureStartedNs: Long = SystemClock.elapsedRealtimeNanos())
    {
        if (stopping) { return }
        val started = captureStartedNs
        val processor = preprocessor ?: return
        val model = engine ?: return
        val changedSettings=app.settings()
        if(changedSettings!=observedSettings) {
            closeRun("recording_settings_changed")
            preprocessor?.clear();pendingFrame=false
            if(calibration==null) rate=RateController(changedSettings.mode.ceiling(app.preferences.getFloat("calibrated_fps",1f).toDouble()))
            observedSettings=changedSettings
        }
        val target = calibration?.fps ?: rate.fps
        app.currentFps = target
        val period = (1e9 / target).toLong()
        try
        {
            if(started-lastPermissionCheckNs>=1_000_000_000L) {
                lastPermissionCheckNs=started
                val readiness=Permissions.inspect(app)
                if(!readiness.ready) {
                    closeRun("permission_revoked")
                    app.captureStatus="Recording stopped: ${readiness.missingDescription}. Restore permissions and grant new screen sharing."
                    stopSelf();return
                }
            }
            paused=false
            // onCapturedContentResize was added in API 34. Older systems need an explicit size check.
            if (Build.VERSION.SDK_INT < 34)
            {
                val bounds = getSystemService(WindowManager::class.java).maximumWindowMetrics.bounds
                if (bounds.width() != width || bounds.height() != height) { resizeCapture(bounds.width(), bounds.height()) }
            }
            val image = reader?.acquireLatestImage()
            val frame=accessibilityFrame ?: image?.let { CapturedFrame.from(it) }
            val fresh = frame != null && (frame.timestamp == 0L || frame.timestamp != lastImageTimestamp)
            val acquisitionNs = SystemClock.elapsedRealtimeNanos()
            val acquisitionWall = System.currentTimeMillis()
            val label = if (calibration == null) app.labels.snapshot() else null
            val thermal = getSystemService(PowerManager::class.java).currentThermalStatus
            val quota = observedSettings.quotaMb * 1024 * 1024
            app.thermalStatus=thermal
            if (quotaPaused && app.store.queueStats().first < quota * 8 / 10 && filesDir.usableSpace >= 128L * 1024 * 1024) { quotaPaused = false }
            val labelEligible = label != null && allowed(label.packageName)
            val changed = label != null && label.generation != lastGeneration
            if (changed)
            {
                lastGeneration = label!!.generation
                processor.clear()
            }
            if (frame != null)
            {
                try
                {
                    if (fresh && (calibration != null || (labelEligible && quotaPaused == false && thermal < 4)))
                    {
                        processor.ingest(frame)
                        pendingFrame = true
                        pendingAcquisitionNs = acquisitionNs
                        pendingAcquisitionWall = acquisitionWall
                    }
                    lastImageTimestamp = frame.timestamp
                }
                finally { image?.close() }
            }
            if (calibration != null)
            {
                app.captureStatus = calibration!!.description
                app.calibrationState=CalibrationState(true,calibration!!.progress,calibration!!.description,calibration!!.selectedFps)
                if (thermal >= 4) { error("Device became critically hot during calibration; stopped") }
                if (processor.hasFrame)
                {
                    val prepared=processor.prepare()
                    val vector = model.encode(prepared)
                    app.latestPreprocessMs=processor.processingMs
                    app.latestInferenceMs=model.inferenceMs;app.latestHeadMs=model.headMs
                    app.preprocessingBackend=processor.backend
                    app.store.benchmarkWrite(JSONObject().put("embedding_b64", vectorBase64(vector)).put("inference_ms", model.inferenceMs))
                    val end = SystemClock.elapsedRealtimeNanos()
                    val done = calibration!!.record(started / 1000000, end / 1000000, (end - started) / 1e6, fresh, end > dueNs + period + period / 10)
                    if (done) { finishCalibration(thermal) }
                }
            }
            else if (quotaPaused)
            {
                paused=true
                closeRun("local_queue_full")
                processor.clear()
                app.captureStatus = "PAUSED: local queue/storage limit. Existing embeddings are safe; resumes after uploads free space."
            }
            else if (thermal >= 4)
            {
                paused=true
                closeRun("thermal_pause")
                processor.clear()
                app.captureStatus = "PAUSED: device critically hot; automatic retry while projection remains active"
            }
            else if (label == null || allowed(label.packageName) == false)
            {
                paused=true
                closeRun(if (label == null) "ambiguous_or_hidden_app" else "app_excluded")
                processor.clear()
                app.captureStatus = "Waiting: ${app.labels.status}"
            }
            else if (label.settled == false)
            {
                // Hold at most one candidate frame in RAM until this label generation settles.
                // A changed generation clears it, preventing reuse across foreground apps.
                if (runPackage != null && runPackage != label.packageName) { closeRun("app_switch") }
                app.captureStatus = "Waiting for ${label.packageName} to settle"
            }
            else if (pendingFrame && processor.hasFrame)
            {
                if (runPackage != null && runPackage != label.packageName) { closeRun("app_switch") }
                pendingFrame = false
                val frameNs = pendingAcquisitionNs
                val frameWall = pendingAcquisitionWall
                val input = processor.prepare()
                if(app.needsCalibration()) { closeRun("processing_backend_changed");app.captureStatus="Processing backend changed. Start with calibration before continuing.";stopSelf();return }
                val vector = model.encode(input)
                val after = app.labels.snapshot()
                if (after != null && after.generation == label.generation && after.settled)
                {
                    if (runId == null) { startRun(label.packageName, frameWall, frameNs) }
                    app.activePackage=label.packageName
                    val sample = JSONObject().put("sequence", sequence).put("wall_ms", frameWall).put("elapsed_ns", frameNs).put("image_timestamp_ns", processor.imageTimestampNs).put("activity", label.activity ?: JSONObject.NULL).put("activity_source", if (label.activity == null) "unknown" else "usage_stats").put("window_class", label.windowClass ?: JSONObject.NULL).put("label_age_ms", label.ageMs).put("source_width", processor.sourceWidth).put("source_height", processor.sourceHeight).put("rotation", rotation()).put("target_fps", target).put("preprocess_ms", processor.processingMs).put("inference_ms", model.inferenceMs).put("head_ms", model.headMs).put("processing_backend",model.backend).put("preprocessing_backend",processor.backend).put("capture_source",source.name.lowercase()).put("sampling_mode",observedSettings.mode.name).put("thermal_status", thermal).put("embedding_b64", vectorBase64(vector))
                    if (app.store.append(runId!!, owner, sample, quota))
                    {
                        sequence += 1
                        app.latestPreprocessMs=processor.processingMs
                        app.latestInferenceMs=model.inferenceMs;app.latestHeadMs=model.headMs
                        app.preprocessingBackend=processor.backend
                        app.metrics.commit(SystemClock.elapsedRealtimeNanos(),(SystemClock.elapsedRealtimeNanos()-started)/1e6)
                        app.captureStatus = "Recording ${label.packageName}: sample $sequence at up to ${"%.2f".format(target)} fps"
                    }
                    else { quotaPaused = true; closeRun("local_queue_full") }
                }
                else
                {
                    processor.clear()
                    if (after == null || after.packageName != label.packageName) { closeRun("app_changed_during_inference") }
                }
                rate.observe((SystemClock.elapsedRealtimeNanos() - started) / 1e6, thermal)
            }
            else
            {
                app.captureStatus = "Recording ${label.packageName}; waiting for a new display frame"
            }
            val nowMs = SystemClock.elapsedRealtime()
            if (nowMs - lastNotificationMs >= 2000)
            {
                getSystemService(NotificationManager::class.java).notify(1, notification("${observedSettings.mode.title} · ${app.metrics.total} samples · ${"%.1f".format(target)} /s\n${app.captureStatus}"))
                app.refreshSnapshot()
                lastNotificationMs = nowMs
            }
        }
        catch (error: Throwable)
        {
            app.captureStatus = "Capture stopped safely: ${error.message}"
            stopSelf()
            return
        }
        if (stopping == false)
        {
            val now = SystemClock.elapsedRealtimeNanos()
            dueNs += period
            if (dueNs < now) { dueNs = now + 1000000L }
            handler.postDelayed({ tick() }, if(paused) 1000L else ceil((dueNs - now).coerceAtLeast(0L) / 1e6).toLong())
        }
    }

    private fun finishCalibration(thermal: Int)
    {
        val calibrated = calibration ?: return
        val model = engine!!
        val report = JSONObject().put("id", UUID.randomUUID().toString()).put("device_id", app.deviceId).put("model_id", model.modelId).put("wall_ms", System.currentTimeMillis()).put("backend",model.backend).put("preprocessing_backend",preprocessor!!.backend).put("capture_source",source.name.lowercase()).put("gpu_parity_cosine", model.parityCosine).put("selected_fps", calibrated.selectedFps).put("thermal_status", thermal).put("results", calibrated.report())
        app.store.saveReport(owner, report)
        app.store.clearBenchmarkProbe()
        check(calibrated.hasSustainableRate) { "No sustainable fresh-frame rate verified. Keep the calibration animation visible and retry." }
        app.preferences.edit().putString("calibrated_signature",app.calibrationSignature(model.backend)).putString("calibrated_model", model.modelId).putFloat("calibrated_fps", calibrated.selectedFps.toFloat()).putString("last_benchmark", report.toString()).commit()
        rate = RateController(observedSettings.mode.ceiling(calibrated.selectedFps))
        calibration = null
        app.calibrating = false
        app.calibrationState=CalibrationState(false,1f,"Calibration complete",calibrated.selectedFps)
        preprocessor?.clear()
        lastGeneration = -1L
        app.captureStatus = "Calibration complete. Continuous collection is active; open an allowed app."
        app.requestUpload()
    }

    private fun allowed(packageName: String): Boolean = observedSettings.allows(packageName,this.packageName)

    private fun startRun(packageName: String, wallMs: Long, elapsedNs: Long)
    {
        runId = UUID.randomUUID().toString()
        runPackage = packageName
        sequence = 0
        @Suppress("DEPRECATION")
        val info = runCatching { packageManager.getPackageInfo(packageName, 0) }.getOrNull()
        val metadata = JSONObject().put("id", runId).put("device_id", app.deviceId).put("capture_session_id", captureSession).put("model_id", engine!!.modelId).put("package_name", packageName).put("app_version", info?.versionName?.take(128) ?: JSONObject.NULL).put("app_version_code", info?.longVersionCode ?: JSONObject.NULL).put("device_model", "${Build.MANUFACTURER} ${Build.MODEL}".take(128)).put("android_sdk", Build.VERSION.SDK_INT).put("start_wall_ms", wallMs).put("start_elapsed_ns", elapsedNs).put("revision", 1).put("end_wall_ms", JSONObject.NULL).put("end_elapsed_ns", JSONObject.NULL).put("end_reason", JSONObject.NULL).put("expected_samples", JSONObject.NULL).put("recording_policy", "new_frames_only_v1")
        app.store.createRun(owner, metadata)
    }

    private fun closeRun(reason: String)
    {
        runId?.let { app.store.closeRun(it, reason) }
        runId = null
        runPackage = null
        sequence = 0
    }

    private fun rotation(): Int
    {
        return getSystemService(DisplayManager::class.java).getDisplay(android.view.Display.DEFAULT_DISPLAY)?.rotation ?: 0
    }

    private fun vectorBase64(vector: FloatArray): String
    {
        val data = ByteBuffer.allocate(vector.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        vector.forEach { data.putFloat(it) }
        return Base64.encodeToString(data.array(), Base64.NO_WRAP)
    }

    private fun notification(text: String): Notification
    {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val stop = PendingIntent.getService(this, 1, Intent(this, CaptureService::class.java).setAction("STOP"), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        return Notification.Builder(this, "capture").setContentTitle("Screen embeddings are being collected").setContentText(text).setSmallIcon(android.R.drawable.ic_menu_view).setContentIntent(open).setOngoing(true).setOnlyAlertOnce(true).addAction(Notification.Action.Builder(null, "Stop", stop).build()).build()
    }

    override fun onDestroy()
    {
        stopping = true
        accessibilityCapture?.closed=true
        app.captureActive = false
        app.sessionStoppedNs=SystemClock.elapsedRealtimeNanos()
        app.calibrating = false
        app.calibrationState=app.calibrationState.copy(active=false)
        app.currentFps=0.0
        runCatching { unregisterReceiver(screenOff) }
        maintenance.shutdownNow()
        handler.removeCallbacksAndMessages(null)
        handler.post {
            runCatching { closeRun("capture_stopped") }
            runCatching { display?.release() }
            runCatching { reader?.close() }
            runCatching { projection?.stop() }
            runCatching { preprocessor?.close() }
            runCatching { engine?.close() }
            app.requestUpload()
            thread.quitSafely()
        }
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
