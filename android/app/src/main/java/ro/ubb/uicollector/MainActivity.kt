package ro.ubb.uicollector

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionConfig
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.text.InputType
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.json.JSONObject
import java.util.concurrent.Executors

class MainActivity : Activity()
{
    private val app get() = application as RecorderApp
    private val main = Handler(Looper.getMainLooper())
    private val tasks = Executors.newSingleThreadExecutor()
    private lateinit var endpoint: EditText
    private lateinit var username: EditText
    private lateinit var password: EditText
    private lateinit var packages: EditText
    private lateinit var quota: EditText
    private lateinit var status: TextView
    private lateinit var animation: CalibrationView
    private var pendingCalibration = false
    private var screenConsentPending = false
    private val update = object : Runnable
    {
        override fun run()
        {
            val stats = app.store.queueStats()
            status.text = "${app.captureStatus}\n\n${app.uploadStatus}\n\nQueued: ${stats.second} embeddings, ${"%.1f".format(stats.first / 1048576.0)} MiB encrypted payloads\nLabel status: ${app.labels.status}"
            animation.visibility = if (app.captureActive && app.captureStatus.contains("calibration", ignoreCase = true)) View.VISIBLE else View.GONE
            main.postDelayed(this, 1000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?)
    {
        super.onCreate(savedInstanceState)
        pendingCalibration = savedInstanceState?.getBoolean("pending_calibration") ?: false
        screenConsentPending = savedInstanceState?.getBoolean("screen_consent_pending") ?: false
        val identity = runCatching { app.vault.identity() }.getOrNull()
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(28, 44, 28, 28)
        }
        fun label(text: String)
        {
            layout.addView(TextView(this).apply { this.text = text; textSize = 16f; setPadding(0, 12, 0, 10) })
        }
        fun field(hint: String, value: String = ""): EditText
        {
            val input = EditText(this).apply { this.hint = hint; setText(value); setSingleLine(true) }
            layout.addView(input)
            return input
        }
        fun button(text: String, action: () -> Unit)
        {
            layout.addView(Button(this).apply { this.text = text; setOnClickListener { action() } })
        }
        label("UI Embedding Collector · F6")
        label("Research collection on this device. Screen pixels exist only in RAM; neither screenshots nor video files are saved or uploaded. Embeddings and app/activity metadata can still reveal sensitive information. Capture uses a visible notification and can be stopped at any time.")
        endpoint = field("HTTPS server, e.g. https://embeddings.example.org/", identity?.optString("server").orEmpty())
        endpoint.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        username = field("Username", identity?.optString("username").orEmpty())
        password = field("Password")
        password.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        button("Sign in")
        {
            if (app.captureActive) { message("Stop capture before signing in"); return@button }
            val url = endpoint.text.toString()
            val user = username.text.toString()
            val secret = password.text.toString()
            password.setText("")
            tasks.execute {
                val error = runCatching { app.uploader.login(url, user, secret) }.exceptionOrNull()
                main.post { if (error != null) message(error.message.orEmpty()) else message("Authenticated. Pending uploads use this account only.") }
            }
        }
        label("Record only selected foreground apps. Returning to an app starts a new run. Keyboard overlays, ambiguous windows and the collector itself are excluded.")
        packages = field("Package allowlist, comma separated", app.preferences.getString("allowed_packages", "").orEmpty())
        packages.setSingleLine(false)
        button("Select installed applications") { selectApplications() }
        quota = field("Queue payload limit (MiB)", app.preferences.getLong("queue_quota_mb", 512L).toString())
        quota.inputType = InputType.TYPE_CLASS_NUMBER
        button("Enable accessibility app labels") { startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }
        button("Enable usage access for activity labels") { startActivity(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)) }
        button("Calibrate GPU and start continuous capture") { begin(true) }
        button("Start continuous capture with saved calibration") { begin(false) }
        button("View last benchmark")
        {
            val saved = app.preferences.getString("last_benchmark", null)
            if (saved == null) { message("No completed benchmark yet"); return@button }
            val report = JSONObject(saved)
            val rows = report.getJSONArray("results")
            val lines = (0 until rows.length()).map { index ->
                val row = rows.getJSONObject(index)
                "${row.getDouble("target_fps")} target fps: ${"%.2f".format(row.getDouble("achieved_fps"))} achieved, p95 ${"%.1f".format(row.getDouble("p95_ms"))} ms, passed=${row.getBoolean("sustainable")}" 
            }
            message("Chosen rate: ${report.getDouble("selected_fps")} fps\nGPU parity cosine: ${report.getDouble("gpu_parity_cosine")}\n\n" + lines.joinToString("\n"))
        }
        button("Stop capture")
        {
            if (app.captureActive)
            {
                startService(Intent(this, CaptureService::class.java).setAction("STOP"))
            }
        }
        button("Retry upload now") { app.uploader.retryNow(); app.requestUpload() }
        button("Show last benchmark") { message(app.preferences.getString("last_benchmark", "No completed benchmark yet").orEmpty()) }
        button("Delete queued data from this device")
        {
            if (app.captureActive) { message("Stop capture before deleting queued data"); return@button }
            AlertDialog.Builder(this).setTitle("Delete unuploaded embeddings?").setMessage("This permanently deletes the local queue. It does not delete data already stored on the server.").setNegativeButton("Cancel", null).setPositiveButton("Delete locally") { _, _ -> app.store.purgePending(); message("Local queue deleted") }.show()
        }
        status = TextView(this).apply { setPadding(0, 24, 0, 12); textSize = 15f }
        layout.addView(status)
        animation = CalibrationView(this)
        layout.addView(animation, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 360))
        setContentView(ScrollView(this).apply { addView(layout) })
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED)
        {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 50)
        }
    }

    private fun begin(calibrate: Boolean)
    {
        if (app.captureActive || screenConsentPending) { message("Capture is already active or permission is pending"); return }
        if (app.vault.identity() == null) { message("Sign in once before capture. Later network outages are supported."); return }
        if (app.labels.hasUsagePermission() == false) { message("Enable usage access first"); return }
        if (packages.text.toString().trim().isEmpty()) { message("Choose the packages that may be recorded"); return }
        val limit = quota.text.toString().toLongOrNull()
        if (limit == null || limit !in 64L..8192L) { message("Queue payload quota must be 64–8192 MiB"); return }
        val model = runCatching { JSONObject(assets.open("f6_manifest.json").bufferedReader().use { it.readText() }) }.getOrNull()
        if (model == null) { message("Deployment assets are missing. Run 00_export_model.sh, then build the APK again."); return }
        if (calibrate == false && app.preferences.getString("calibrated_model", "") != model.getString("model_id"))
        {
            message("Calibrate this model on this device first")
            return
        }
        app.preferences.edit().putString("allowed_packages", packages.text.toString().trim()).putLong("queue_quota_mb", limit).commit()
        AlertDialog.Builder(this).setTitle("Start continuous screen analysis?").setMessage("Only selected apps will contribute embeddings. Use entire-screen capture in Android's consent dialog, not a single app. Screen lock or Android revoking capture ends this session; the encrypted upload queue survives. Calibration displays a moving test pattern and then continues collecting until stopped.").setNegativeButton("Cancel", null).setPositiveButton("Continue") { _, _ ->
            pendingCalibration = calibrate
            screenConsentPending = true
            val manager = getSystemService(MediaProjectionManager::class.java)
            val request = if (Build.VERSION.SDK_INT >= 34) manager.createScreenCaptureIntent(MediaProjectionConfig.createConfigForDefaultDisplay()) else manager.createScreenCaptureIntent()
            @Suppress("DEPRECATION")
            startActivityForResult(request, 100)
        }.show()
    }

    @Deprecated("Activity result callback retained for this minimal Android-only activity")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?)
    {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 100)
        {
            screenConsentPending = false
            if (resultCode == RESULT_OK && data != null)
            {
                val request = Intent(this, CaptureService::class.java).putExtra("permission", data).putExtra("result_code", resultCode).putExtra("calibrate", pendingCalibration)
                startForegroundService(request)
                if (pendingCalibration) { startActivity(Intent(this, BenchmarkActivity::class.java)) }
            }
        }
    }

    private fun selectApplications()
    {
        if (app.captureActive) { message("Stop capture before changing the allowlist"); return }
        @Suppress("DEPRECATION")
        val entries = packageManager.queryIntentActivities(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0).map { Pair(it.activityInfo.packageName, it.loadLabel(packageManager).toString()) }.distinctBy { it.first }.sortedBy { it.second.lowercase() }
        val selected = packages.text.toString().split(',', '\n').map { it.trim() }.toMutableSet()
        val checks = BooleanArray(entries.size) { entries[it].first in selected }
        AlertDialog.Builder(this).setTitle("Applications included in collection").setMultiChoiceItems(entries.map { "${it.second}\n${it.first}" }.toTypedArray(), checks) { _, index, checked -> if (checked) selected.add(entries[index].first) else selected.remove(entries[index].first) }.setPositiveButton("Use selection") { _, _ -> packages.setText(selected.filter { it.isNotEmpty() }.sorted().joinToString(",")) }.setNegativeButton("Cancel", null).show()
    }

    private fun message(text: String)
    {
        if (isFinishing == false && isDestroyed == false) { AlertDialog.Builder(this).setMessage(text).setPositiveButton("OK", null).show() }
    }

    override fun onSaveInstanceState(outState: Bundle)
    {
        outState.putBoolean("pending_calibration", pendingCalibration)
        outState.putBoolean("screen_consent_pending", screenConsentPending)
        super.onSaveInstanceState(outState)
    }

    override fun onResume() { super.onResume(); main.post(update) }
    override fun onPause() { main.removeCallbacks(update); password.setText(""); super.onPause() }
    override fun onDestroy() { main.removeCallbacksAndMessages(null); tasks.shutdown(); super.onDestroy() }
}

class CalibrationView(context: android.content.Context) : View(context)
{
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var phase = 0

    override fun onDraw(canvas: Canvas)
    {
        super.onDraw(canvas)
        canvas.drawColor(Color.WHITE)
        for (i in 0 until 12)
        {
            paint.color = Color.rgb((phase + i * 31) % 255, (i * 47) % 255, 160)
            val left = ((phase * 3 + i * 70) % (width.coerceAtLeast(1))).toFloat()
            canvas.drawRect(left, (i * 23).toFloat(), left + 55, (i * 23 + 18).toFloat(), paint)
        }
        paint.color = Color.BLACK
        paint.textSize = 28f
        canvas.drawText("GPU calibration: keep this moving pattern visible", 4f, height - 12f, paint)
        phase = (phase + 1) % 10000
        if (visibility == VISIBLE) { postInvalidateOnAnimation() }
    }
}
