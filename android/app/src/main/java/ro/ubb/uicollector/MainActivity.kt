package ro.ubb.uicollector

import android.Manifest
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionConfig
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.selection.selectable
import androidx.compose.ui.semantics.Role
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Apps
import androidx.compose.material.icons.outlined.FiberManualRecord
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

enum class Menu { RECORDING, APPLICATIONS, SETTINGS, PERMISSIONS, CAPTURE, SERVER, CALIBRATION, STORAGE, DIAGNOSTICS }
data class InstalledApplication(val packageName: String, val label: String, val system: Boolean)

class MainActivity : ComponentActivity() {
    private val app get() = application as RecorderApp
    private val tasks = Executors.newSingleThreadExecutor()
    private var permissionState by mutableStateOf(PermissionState())
    private var screen by mutableStateOf(Menu.RECORDING)
    private var setupStep by mutableIntStateOf(0)
    private var onboardingComplete by mutableStateOf(false)
    private var recordingSettings by mutableStateOf(RecordingSettings())
    private var pendingCalibration = false
    private var consentPending by mutableStateOf(false)
    private var applications by mutableStateOf(emptyList<InstalledApplication>())
    private val notificationPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { refreshPermissions() }
    private val projectionConsent = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        consentPending = false
        refreshPermissions()
        if (result.resultCode == RESULT_OK && result.data != null && permissionState.ready) {
            onboardingComplete = true
            app.preferences.edit().putBoolean("onboarding_complete", true).apply()
            startForegroundService(Intent(this, CaptureService::class.java).putExtra("permission", result.data)
                .putExtra("result_code", result.resultCode).putExtra("calibrate", pendingCalibration))
            screen = Menu.RECORDING
        } else if (result.resultCode != RESULT_OK) message("Screen sharing was not granted. Recording has not started.")
        else message("A required permission changed. Complete permission setup before recording.")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupStep = app.preferences.getInt("onboarding_step", 0).coerceIn(0,4)
        onboardingComplete = app.preferences.getBoolean("onboarding_complete", false)
        recordingSettings = app.settings()
        pendingCalibration = savedInstanceState?.getBoolean("pending_calibration") ?: false
        consentPending = savedInstanceState?.getBoolean("consent_pending") ?: false
        screen = runCatching { Menu.valueOf(savedInstanceState?.getString("menu") ?: "RECORDING") }.getOrDefault(Menu.RECORDING)
        refreshPermissions()
        loadApplications()
        setContent {
            MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
                val snapshot by app.recording.collectAsStateWithLifecycle()
                DisposableEffect(snapshot.calibration.active) {
                    if(snapshot.calibration.active) window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    else window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    onDispose { window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
                }
                LaunchedEffect(Unit) {
                    lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
                        while (true) {
                            withContext(Dispatchers.IO) { app.refreshSnapshot() }
                            refreshPermissions()
                            delay(1000)
                        }
                    }
                }
                BackHandler(screen != Menu.RECORDING || (!onboardingComplete && setupStep > 0)) {
                    if(screen in setOf(Menu.RECORDING, Menu.APPLICATIONS, Menu.SETTINGS)) {
                        if(screen == Menu.RECORDING) moveStep(setupStep - 1) else screen = Menu.RECORDING
                    } else screen = Menu.SETTINGS
                }
                Scaffold(
                    topBar = { Column(Modifier.fillMaxWidth().statusBarsPadding().padding(horizontal=24.dp, vertical=16.dp)) {
                        Text("UI Embedding Collector", style=MaterialTheme.typography.titleLarge, fontWeight=FontWeight.Bold)
                        Text(screen.name.lowercase().replaceFirstChar { it.uppercase() }, style=MaterialTheme.typography.labelLarge, color=MaterialTheme.colorScheme.primary)
                    } },
                    bottomBar = { NavigationBar {
                        listOf(Menu.RECORDING,Menu.APPLICATIONS,Menu.SETTINGS).forEach { menu ->
                            NavigationBarItem(selected = screen == menu || (menu == Menu.SETTINGS && screen !in setOf(Menu.RECORDING,Menu.APPLICATIONS)),
                                onClick = { screen = menu; if(menu==Menu.APPLICATIONS) loadApplications() }, label = { Text(menu.name.lowercase().replaceFirstChar { it.uppercase() }) },
                                icon = { Icon(when(menu) { Menu.RECORDING -> Icons.Outlined.FiberManualRecord; Menu.APPLICATIONS -> Icons.Outlined.Apps; else -> Icons.Outlined.Settings }, contentDescription=null) })
                        }
                    } }
                ) { padding ->
                    val menuScroll=remember(screen) { androidx.compose.foundation.ScrollState(0) }
                    Column(Modifier.padding(padding).fillMaxSize().verticalScroll(menuScroll).padding(horizontal=20.dp), verticalArrangement=Arrangement.spacedBy(16.dp)) {
                        when(screen) {
                            Menu.RECORDING -> if (!onboardingComplete && !snapshot.active) Setup() else Recording(snapshot)
                            Menu.APPLICATIONS -> Applications()
                            Menu.SETTINGS -> SettingsMenu()
                            Menu.PERMISSIONS -> PermissionChecklist()
                            Menu.CAPTURE -> CaptureSettings(snapshot)
                            Menu.SERVER -> Server(snapshot)
                            Menu.CALIBRATION -> CalibrationMenu(snapshot)
                            Menu.STORAGE -> Storage(snapshot)
                            Menu.DIAGNOSTICS -> Diagnostics(snapshot)
                        }
                        Spacer(Modifier.height(20.dp))
                    }
                }
            }
        }
    }

    private fun loadApplications() {
        tasks.execute {
            @Suppress("DEPRECATION")
            val installed = packageManager.getInstalledApplications(0).map {
                InstalledApplication(it.packageName, it.loadLabel(packageManager).toString(), it.flags and ApplicationInfo.FLAG_SYSTEM != 0)
            }.sortedBy { it.label.lowercase() }
            runOnUiThread { applications = installed }
        }
    }

    private fun refreshPermissions() { permissionState = Permissions.inspect(app) }
    private fun moveStep(step: Int) { setupStep = step.coerceIn(0,4); app.preferences.edit().putInt("onboarding_step",setupStep).apply() }
    private fun begin(forceCalibration: Boolean = false) {
        refreshPermissions()
        if (!permissionState.ready) { screen = Menu.PERMISSIONS; message("Required permissions are incomplete: ${permissionState.missingDescription}"); return }
        if(app.accountOperation) { message("Wait for the account/data operation to finish"); return }
        if(app.captureActive || consentPending) return
        pendingCalibration = forceCalibration || app.needsCalibration()
        if(app.captureSource()==CaptureSource.ACCESSIBILITY) {
            val service=app.accessibilityService
            if(service==null || service.serviceInfo.capabilities and android.accessibilityservice.AccessibilityServiceInfo.CAPABILITY_CAN_TAKE_SCREENSHOT == 0) {
                message(if(service==null) "Accessibility permission is enabled, but Android has not connected the collector service. Switch the collector off and on in Accessibility settings, then try again." else "The running Accessibility service has not loaded screen capture support. Switch the collector off and on in Accessibility settings after this update.")
                screen=Menu.PERMISSIONS
                return
            }
            onboardingComplete=true
            app.preferences.edit().putBoolean("onboarding_complete",true).apply()
            startForegroundService(Intent(this,CaptureService::class.java).putExtra("calibrate",pendingCalibration))
            screen=Menu.RECORDING
            return
        }
        consentPending = true
        val manager = getSystemService(MediaProjectionManager::class.java)
        projectionConsent.launch(if(Build.VERSION.SDK_INT >= 34) manager.createScreenCaptureIntent(MediaProjectionConfig.createConfigForDefaultDisplay()) else manager.createScreenCaptureIntent())
    }

    @Composable private fun Panel(title: String, text: String, content: @Composable ColumnScope.() -> Unit = {}) {
        ElevatedCard(Modifier.fillMaxWidth()) { Column(Modifier.padding(20.dp), verticalArrangement=Arrangement.spacedBy(12.dp)) {
            Text(title, style=MaterialTheme.typography.titleMedium, fontWeight=FontWeight.SemiBold)
            if(text.isNotEmpty()) Text(text, style=MaterialTheme.typography.bodyMedium)
            content()
        } }
    }
    @Composable private fun Action(title: String, enabled: Boolean = true, onClick: () -> Unit) {
        Button(onClick=onClick, enabled=enabled, modifier=Modifier.fillMaxWidth().heightIn(min=48.dp)) { Text(title) }
    }
    @Composable private fun Setup() {
        Text("Welcome · step ${setupStep+1} of 5", style=MaterialTheme.typography.headlineSmall)
        LinearProgressIndicator(progress={ (setupStep+1)/5f }, modifier=Modifier.fillMaxWidth())
        when(setupStep) {
            0 -> Panel("Record locally, connect later", "Your phone turns full-screen views into embeddings. Pixels stay in memory; screenshots and videos are never saved. All eligible apps are included unless you opt out. Embeddings and application metadata are stored encrypted on this phone.") {
                OutlinedButton(onClick={screen=Menu.APPLICATIONS}) { Text("Review application exclusions") }
            }
            1 -> GrantCard("Notifications", permissionState.notifications, "A visible recording notification shows status and lets you stop at any time.") { requestNotifications() }
            2 -> GrantCard("Accessibility", permissionState.accessibility, "Enable UI Embedding Collector in Android Accessibility settings. It captures full-screen pixels into memory for local embeddings and reads application/window identity. It never reads text nodes or saves screenshots. If Android blocks a sideloaded app, open App info → ⋮ → Allow restricted settings, then return here.") { openSettings(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }
            3 -> GrantCard("Usage Access", permissionState.usageAccess, "Enable usage access so each embedding can carry the correct foreground Activity label.") { openSettings(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS).setData(Uri.parse("package:$packageName")), Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)) }
            4 -> Panel("Ready to calibrate", "Keep the moving calibration pattern visible while the phone measures a sustainable rate. Continuous capture pauses while locked and resumes after unlock. Recording continues locally in Maximum Detail mode; open an eligible app.") {
                Text("Capture: ${app.captureSource().title}")
                PermissionRows()
                Action(if(consentPending) "Waiting for screen-sharing consent" else "Start calibration", permissionState.ready && !consentPending) { begin(true) }
            }
        }
        if(setupStep < 4) {
            val complete = when(setupStep) { 1 -> permissionState.notifications == GrantState.GRANTED; 2 -> permissionState.accessibility == GrantState.GRANTED; 3 -> permissionState.usageAccess == GrantState.GRANTED; else -> true }
            Action("Next", complete) { moveStep(setupStep+1) }
        }
        if(setupStep > 0) TextButton(onClick={moveStep(setupStep-1)}) { Text("Back") }
    }
    private fun requestNotifications() {
        if(Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED &&
            (!app.preferences.getBoolean("notification_requested",false) || shouldShowRequestPermissionRationale(Manifest.permission.POST_NOTIFICATIONS))) {
            app.preferences.edit().putBoolean("notification_requested",true).apply()
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else openSettings(Permissions.notificationSettings(this))
    }
    private fun openSettings(intent: Intent, fallback: Intent? = null) {
        try { startActivity(intent) } catch(error: Exception) {
            if(fallback != null) startActivity(fallback) else message("Android could not open this settings screen: ${error.message}")
        }
    }
    @Composable private fun GrantCard(title: String, state: GrantState, description: String, action: () -> Unit) {
        Panel(title, description) {
            Text(state.label, color=if(state == GrantState.GRANTED) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error, fontWeight=FontWeight.Bold)
            Action(if(state == GrantState.GRANTED) "Review $title settings" else "Grant $title", onClick=action)
        }
    }
    @Composable private fun PermissionRows() {
        listOf("Notifications" to permissionState.notifications, "Accessibility labels" to permissionState.accessibility, "Usage Access" to permissionState.usageAccess).forEach { (title,state) ->
            Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.SpaceBetween) { Text(title); Text(state.label,color=if(state==GrantState.GRANTED) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error) }
        }
    }
    @Composable private fun PermissionChecklist() {
        Text(if(permissionState.ready) "All required permissions are ready" else "Complete these before recording",style=MaterialTheme.typography.headlineSmall)
        GrantCard("Notifications",permissionState.notifications,"Enable app notifications and the Recording status channel.") { requestNotifications() }
        GrantCard("Accessibility labels",permissionState.accessibility,"Enable the collector's Accessibility service. Sideloaded apps may first require App info → Allow restricted settings.") { openSettings(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }
        AccessibilityRecovery()
        GrantCard("Usage Access",permissionState.usageAccess,"Allow foreground application and Activity labeling.") { openSettings(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS).setData(Uri.parse("package:$packageName")),Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)) }
        Panel("Capture method",if(app.captureSource()==CaptureSource.ACCESSIBILITY) "Continuous Accessibility capture uses your enabled service and resumes after unlock." else "Fast sessions use Android screen sharing. Private notification content may be hidden, and locking ends the session.")
    }
    @Composable private fun Recording(snapshot: RecordingSnapshot) {
        if(!snapshot.active && (permissionState.notifications!=GrantState.GRANTED || permissionState.usageAccess!=GrantState.GRANTED)) Panel("Permissions need attention",permissionState.missingDescription) { Action("Review permissions") { screen=Menu.PERMISSIONS } }
        AccessibilityRecovery()
        Panel(if(snapshot.calibration.active) "Calibrating" else if(snapshot.active) "Recording session active" else "Ready to record",snapshot.status) {
            if(snapshot.active && app.captureSource()==CaptureSource.MEDIA_PROJECTION && Build.VERSION.SDK_INT >= 35) Text("Android hides private notification content while screen sharing is active.",style=MaterialTheme.typography.bodySmall)
            if(!snapshot.calibration.active && snapshot.calibration.selectedFps>0) Text("Calibration complete · ${snapshot.calibration.selectedFps} samples/s ceiling")
            if(snapshot.active) Action("Stop recording") { startService(Intent(this@MainActivity,CaptureService::class.java).setAction("STOP")) }
            else Action(if(consentPending) "Waiting for consent" else if(app.needsCalibration()) "Start with calibration" else "Start recording",permissionState.ready && !consentPending) { begin() }
        }
        if(snapshot.calibration.active) {
            Panel(snapshot.calibration.description,"Keep this moving pattern visible. Calibration includes capture, inference and encrypted storage.") {
                LinearProgressIndicator(progress={snapshot.calibration.progress},modifier=Modifier.fillMaxWidth())
                AndroidView(factory={CalibrationView(it)},modifier=Modifier.fillMaxWidth().height(300.dp))
            }
        }
        ModeSelector()
        Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(12.dp)) {
            Metric("Samples",snapshot.committed.toString(),Modifier.weight(1f))
            Metric("Elapsed","${snapshot.elapsedSeconds}s",Modifier.weight(1f))
        }
        Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(12.dp)) {
            Metric("Achieved","${"%.1f".format(snapshot.achievedFps)} /s",Modifier.weight(1f))
            Metric("Target","${"%.1f".format(snapshot.targetFps)} /s",Modifier.weight(1f))
        }
        Panel("Live processing", "${snapshot.packageName ?: "No recorded application yet"}\nThroughput counts committed embeddings over the last 10 seconds; static screens and excluded apps create no samples.") {
            Text("Preprocess ${"%.1f".format(snapshot.preprocessMs)} ms · inference ${"%.1f".format(snapshot.inferenceMs)} ms")
            Text("Processing p95 ${"%.1f".format(snapshot.processingP95Ms)} ms · thermal ${snapshot.thermalStatus}")
            val color=MaterialTheme.colorScheme.primary
            Canvas(Modifier.fillMaxWidth().height(64.dp)) {
                val values=snapshot.history
                val max=(values.maxOrNull() ?: 1.0).coerceAtLeast(1.0)
                values.zipWithNext().forEachIndexed { index,(a,b) ->
                    drawLine(color,Offset(index*size.width/59, size.height-(a/max*size.height).toFloat()),Offset((index+1)*size.width/59,size.height-(b/max*size.height).toFloat()),3f)
                }
            }
        }
        PendingUploadsPanel(snapshot)
        Text("Battery ${snapshot.batteryPercent}%${if(snapshot.charging) " · charging (power measurement unavailable)" else ""}",style=MaterialTheme.typography.bodySmall)
    }
    @Composable private fun AccessibilityRecovery() {
        if(permissionState.accessibility!=GrantState.GRANTED) {
            Panel("Accessibility is off","Android can turn off this service after Force Stop. Enable UI Embedding Collector in Accessibility settings, then return here and start recording.") {
                Action("Open Accessibility settings") { openSettings(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }
            }
        } else if(!app.labels.isConnected()) {
            Panel("Accessibility needs reconnecting","Permission is granted, but Android has not connected the collector. This can happen after Force Stop. If it stays disconnected, switch the collector off and on in Accessibility settings, then return here.") {
                Action("Open Accessibility settings") { openSettings(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }
            }
        }
    }
    @Composable private fun Metric(title: String, value: String, modifier: Modifier) {
        FilledTonalButton(onClick={},modifier=modifier.heightIn(min=72.dp)) { Column { Text(value,style=MaterialTheme.typography.titleLarge); Text(title,style=MaterialTheme.typography.labelMedium) } }
    }
    @Composable private fun ModeSelector() {
        val calibrated=if(app.needsCalibration()) null else app.preferences.getFloat("calibrated_fps",0f).toDouble()
        Panel("Recording mode","Rates apply to the selected capture method. Static screens, heat and excluded apps can lower actual throughput.") {
            RecordingMode.entries.forEach { mode ->
                Row(Modifier.fillMaxWidth().selectable(selected=recordingSettings.mode==mode,role=Role.RadioButton,onClick={ app.preferences.edit().putString("recording_mode",mode.name).apply(); recordingSettings=app.settings() }),horizontalArrangement=Arrangement.SpaceBetween) {
                    Column(Modifier.weight(1f).padding(vertical=8.dp)) {
                        Text(mode.title,fontWeight=FontWeight.SemiBold)
                        Text(mode.description,style=MaterialTheme.typography.bodySmall)
                        Text(if(calibrated==null) "Effective rate available after calibration" else "Up to ${mode.ceiling(calibrated)} samples/s",style=MaterialTheme.typography.bodySmall)
                    }
                    RadioButton(selected=recordingSettings.mode==mode,onClick=null)
                }
            }
        }
    }
    @Composable private fun Applications() {
        var query by rememberSaveable { mutableStateOf("") }
        var excludedOnly by rememberSaveable { mutableStateOf(false) }
        var showSystem by rememberSaveable { mutableStateOf(false) }
        Panel("All applications by default","Newly installed apps are included automatically. Opt out below to keep an app out of your recordings. The collector and system UI are always excluded.") {
            if(recordingSettings.policy == CapturePolicy.LEGACY_SELECTED) {
                Text("Your earlier selected-app policy is preserved.")
                Action("Switch to all applications") { app.preferences.edit().putString("capture_policy",CapturePolicy.ALL_EXCEPT_EXCLUDED.name).apply(); recordingSettings=app.settings() }
            }
            Text("${recordingSettings.excluded.size} applications opted out")
        }
        OutlinedTextField(value=query,onValueChange={query=it},label={Text("Search applications")},modifier=Modifier.fillMaxWidth(),singleLine=true)
        Row { FilterChip(selected=excludedOnly,onClick={excludedOnly=!excludedOnly},label={Text("Excluded only")}); Spacer(Modifier.width(8.dp)); FilterChip(selected=showSystem,onClick={showSystem=!showSystem},label={Text("System apps")}) }
        val shown=applications.filter { (showSystem || !it.system || query.isNotEmpty() || it.packageName in recordingSettings.excluded) && (!excludedOnly || it.packageName in recordingSettings.excluded) && (it.label.contains(query,true) || it.packageName.contains(query,true)) }
        shown.forEach { item ->
            val permanent=item.packageName in setOf(packageName,"android","com.android.systemui")
            ElevatedCard(Modifier.fillMaxWidth()) { Row(Modifier.padding(12.dp)) {
                Column(Modifier.weight(1f)) { Text(item.label,fontWeight=FontWeight.Medium); Text(item.packageName,style=MaterialTheme.typography.bodySmall); Text(if(permanent) "Always excluded" else if(recordingSettings.allows(item.packageName,packageName)) "Included" else "Excluded",style=MaterialTheme.typography.labelSmall) }
                Checkbox(checked=recordingSettings.allows(item.packageName,packageName),enabled=!permanent && recordingSettings.policy != CapturePolicy.LEGACY_SELECTED,onCheckedChange={ enabled ->
                    val excluded=recordingSettings.excluded.toMutableSet()
                    if(enabled) excluded.remove(item.packageName) else excluded.add(item.packageName)
                    app.preferences.edit().putStringSet("excluded_packages",excluded).apply(); recordingSettings=app.settings()
                })
            } }
        }
        if(shown.isEmpty()) Text("No matching applications")
    }
    @Composable private fun SettingsMenu() {
        listOf(Menu.PERMISSIONS to "Permissions and readiness",Menu.CAPTURE to "Capture and automatic resume",Menu.SERVER to "Server and account",Menu.CALIBRATION to "Calibration",Menu.STORAGE to "Storage and pending data",Menu.DIAGNOSTICS to "Diagnostics").forEach { (menu,title) ->
            OutlinedButton(onClick={screen=menu},modifier=Modifier.fillMaxWidth().heightIn(min=56.dp)) { Text(title) }
        }
        Text("Version ${packageManager.getPackageInfo(packageName,0).versionName} · F6 · screen pixels stay in memory",style=MaterialTheme.typography.bodySmall)
    }
    @Composable private fun CaptureSettings(snapshot: RecordingSnapshot) {
        var selected by remember { mutableStateOf(app.captureSource()) }
        Panel("Capture and automatic resume","Continuous capture pauses while locked or off and resumes after unlock. A running session is restored after reboot and first unlock when all permissions remain granted; Stop disables automatic resumption. Calibration increases capture frequency until Android rejects a request or the complete pipeline cannot keep up, then confirms a lower rate for 30 seconds. Stop recording to change methods.") {
            CaptureSource.entries.forEach { source ->
                Row(Modifier.fillMaxWidth().selectable(selected==source,enabled=!snapshot.active,role=Role.RadioButton,onClick={
                    selected=source;app.preferences.edit().putString("capture_source",source.name).putBoolean("continuous_recording_requested",false).apply()
                }).padding(vertical=12.dp)) { RadioButton(selected==source,onClick=null,enabled=!snapshot.active);Text(source.title,Modifier.padding(start=8.dp)) }
            }
        }
        Panel("Fast sessions","MediaProjection may capture faster, but Android ends sharing on lock and requires fresh consent. Android 15 can hide private notification content throughout a fast session.")
    }
    @Composable private fun PendingUploadsPanel(snapshot: RecordingSnapshot) {
        val pending=snapshot.pendingUploads
        val identity=app.vault.identity()
        Panel("Uploads and local storage", "${snapshot.queuedSamples} embeddings stored locally · ${"%.1f".format(snapshot.queuedBytes/1048576.0)} MiB") {
            if(identity!=null) {
                Text("Automatic uploads to ${identity.optString("username")} at ${identity.optString("server")}")
                Text("Awaiting server acknowledgment: ${pending.accountSamples} embeddings · ${pending.accountReports} calibration reports")
                Text(snapshot.uploadStatus)
            } else Text("No account connected. Recordings stay on this phone.")
            if(pending.hasLocal) {
                Text("Not assigned to an account: ${pending.localSamples} embeddings · ${pending.localReports} calibration reports")
                Text("These were recorded before signing in. They will stay local until you choose Sync older local recordings in Server and account. Current account uploads do not include them.")
            }
            if(pending.otherSamples>0 || pending.otherReports>0) Text("Another account/server: ${pending.otherSamples} embeddings · ${pending.otherReports} reports. Sign in to their original account to upload them.")
            Text("Acknowledged recordings are removed from this queue. A small changing count is normal while recording.")
        }
    }

    @Composable private fun Server(snapshot: RecordingSnapshot) {
        var url by remember { mutableStateOf(app.vault.identity()?.optString("server").orEmpty()) }
        var user by remember { mutableStateOf(app.vault.identity()?.optString("username").orEmpty()) }
        var password by remember { mutableStateOf("") }
        var busy by remember { mutableStateOf(false) }
        Panel("Optional uploads","Recording works without a server. Local recordings stay encrypted on this phone until you choose an upload destination.")
        OutlinedTextField(value=url,onValueChange={url=it},label={Text("HTTPS server URL")},singleLine=true,modifier=Modifier.fillMaxWidth())
        OutlinedTextField(value=user,onValueChange={user=it},label={Text("Username")},singleLine=true,modifier=Modifier.fillMaxWidth())
        OutlinedTextField(value=password,onValueChange={password=it},label={Text("Password")},visualTransformation=androidx.compose.ui.text.input.PasswordVisualTransformation(),singleLine=true,modifier=Modifier.fillMaxWidth())
        Action(if(busy) "Signing in…" else "Sign in", !busy && !app.captureActive) {
            busy=true; app.accountOperation=true; val secret=password; password=""
            tasks.execute { val error=runCatching { app.uploader.login(url,user,secret) }.exceptionOrNull(); runOnUiThread { busy=false; app.accountOperation=false; message(error?.message ?: "Signed in. Existing local recordings are not uploaded until you select Transfer below.") } }
        }
        if(app.captureActive) Text("Stop recording before changing accounts. Older local recordings can be transferred while recording continues.")
        PendingUploadsPanel(snapshot)
        if(app.vault.identity()!=null) {
            Action("Sync older local recordings to this account",!busy && snapshot.pendingUploads.hasLocal) {
                val identity=app.vault.identity() ?: return@Action
                AlertDialog.Builder(this@MainActivity).setTitle("Transfer local recordings?")
                    .setMessage("Upload unbound local recordings to ${identity.optString("username")} at ${identity.optString("server")}? ${snapshot.pendingUploads.localSamples} embeddings and ${snapshot.pendingUploads.localReports} calibration reports were recorded before signing in. Their original applications and times are preserved. Current recording and account-bound data stay with their account.")
                    .setNegativeButton("Cancel",null).setPositiveButton("Transfer") { _,_ ->
                        busy=true; app.accountOperation=true
                        tasks.execute { val error=runCatching { check(app.captureOwner()==ownerKey(identity)) { "Account changed; reopen this page before transferring" }; app.store.adoptLocal(app.localOwner,ownerKey(identity)); app.uploader.retryNow(); app.requestUpload() }.exceptionOrNull(); runOnUiThread { busy=false; app.accountOperation=false; message(error?.message ?: "Older recordings queued for upload. Recording can continue; the local copies are removed only after server acknowledgment.") } }
                    }.show()
            }
            OutlinedButton(onClick={app.uploader.retryNow();app.requestUpload()}) { Text("Retry uploads") }
        }
    }
    @Composable private fun CalibrationMenu(snapshot: RecordingSnapshot) {
        Panel("Automatic calibration", "Measures capture, inference and storage at increasing rates, up to the app's 30/s test ceiling. Stops at the first unsustainable rate and confirms a lower rate for 30 seconds. Repeats when the model, backend, capture method or GPU driver changes. ${if(app.needsCalibration()) "New calibration required for this configuration." else "Measured sustainable rate: ${app.preferences.getFloat("calibrated_fps",0f)} samples/second."}") {
            Action("Recalibrate and start",permissionState.ready && !snapshot.active && !consentPending) { begin(true) }
        }
        val saved=app.preferences.getString("last_benchmark",null)
        if(saved!=null) {
            val report=org.json.JSONObject(saved); val rows=report.getJSONArray("results")
            Panel("Last calibration","Backend: ${report.getString("backend")}\nSelected ${report.getDouble("selected_fps")} /s") {
                repeat(rows.length()) { val row=rows.getJSONObject(it); Text("${row.getDouble("target_fps")} /s → ${"%.1f".format(row.getDouble("achieved_fps"))} /s · p95 ${"%.1f".format(row.getDouble("p95_ms"))} ms · ${if(row.optBoolean("rate_limited")) "Android screenshot rate limit" else if(row.getBoolean("sustainable")) "passed" else "not sustainable"}${if(row.optBoolean("confirmation")) " · sustained confirmation" else ""}") }
            }
        }
    }
    @Composable private fun Storage(snapshot: RecordingSnapshot) {
        var quota by remember { mutableStateOf(recordingSettings.quotaMb.toString()) }
        PendingUploadsPanel(snapshot)
        Text("Queue limit: recording pauses at this limit; older data is never evicted.")
        if(snapshot.pendingUploads.hasLocal) Action("Review and sync older local recordings") { screen=Menu.SERVER }
        OutlinedTextField(value=quota,onValueChange={quota=it},label={Text("Queue limit, MiB (64–8192)")},modifier=Modifier.fillMaxWidth())
        Action("Save queue limit") { val amount=quota.toLongOrNull(); if(amount==null || amount !in 64L..8192L) message("Enter a queue limit between 64 and 8192 MiB") else { app.preferences.edit().putLong("queue_quota_mb",amount).apply(); recordingSettings=app.settings() } }
        OutlinedButton(enabled=!snapshot.active,onClick={ AlertDialog.Builder(this@MainActivity).setTitle("Delete pending data?").setMessage("Permanently delete all local pending recordings? Server data is unaffected.").setNegativeButton("Cancel",null).setPositiveButton("Delete locally") { _,_ -> tasks.execute { app.store.purgePending() } }.show() }) { Text("Delete pending data") }
    }
    @Composable private fun Diagnostics(snapshot: RecordingSnapshot) {
        Panel("Processing path", "${snapshot.backend}\nPreprocessing: ${snapshot.preprocessing}\nGPU encoder timing includes the head when the full encoder is active. Battery savings require discharge measurements; charging data is not a power benchmark.")
        Action("Run synthetic inference check",!snapshot.active) {
            tasks.execute {
                val result=runCatching { EmbeddingEngine(this@MainActivity).use { "OpenCL encoder passed · cosine ${it.parityCosine}" } }
                runOnUiThread { message(result.getOrElse { "Inference check failed: ${it.message}; ${it.cause?.message.orEmpty()}" }) }
            }
        }
    }
    private fun message(text: String) { if(!isFinishing && !isDestroyed) AlertDialog.Builder(this).setMessage(text).setPositiveButton("OK",null).show() }
    override fun onResume() { super.onResume(); refreshPermissions(); if(screen==Menu.APPLICATIONS) loadApplications() }
    override fun onSaveInstanceState(outState: Bundle) { outState.putBoolean("pending_calibration",pendingCalibration);outState.putBoolean("consent_pending",consentPending);outState.putString("menu",screen.name);super.onSaveInstanceState(outState) }
    override fun onDestroy() { tasks.shutdown();super.onDestroy() }
}
