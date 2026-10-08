package ro.ubb.uicollector

import android.accessibilityservice.AccessibilityService
import android.app.AppOpsManager
import android.app.KeyguardManager
import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.os.Process
import android.os.SystemClock
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityWindowInfo
import android.graphics.Rect

/** Never contains screen text or a view hierarchy. */
data class AppLabel(val packageName: String, val activity: String?, val windowClass: String?, val generation: Long, val ageMs: Long, val settled: Boolean)

/** Tracks the currently bound service instance so stale lifecycle callbacks are harmless. */
class AccessibilityConnectionState
{
    @Volatile private var owner: Any? = null

    @Synchronized
    fun connect(token: Any): Boolean
    {
        if (owner === token) { return false }
        owner = token
        return true
    }

    @Synchronized
    fun disconnect(token: Any): Boolean
    {
        if (owner !== token) { return false }
        owner = null
        return true
    }

    fun isConnected(): Boolean = owner != null
    fun owns(token: Any): Boolean = owner === token
}

class LabelTracker(private val context: Context)
{
    private val connectionState = AccessibilityConnectionState()
    private var windowPackage: String? = null
    private var windowClass: String? = null
    private var windowId = -1
    private var unsafeReason: String? = "Accessibility labeling not connected"
    private var usagePackage: String? = null
    private var usageActivity: String? = null
    private var lastUsagePoll = 0L
    private var lastUsageTimestamp = 0L
    private val usageKeys = mutableSetOf<String>()
    private var generation = 0L
    private var changedAt = SystemClock.elapsedRealtime()
    @Volatile var status = "Enable accessibility labels and usage access"
        private set

    fun isConnected(): Boolean = connectionState.isConnected()

    fun hasUsagePermission(): Boolean
    {
        val manager = context.getSystemService(AppOpsManager::class.java)
        return manager.unsafeCheckOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), context.packageName) == AppOpsManager.MODE_ALLOWED
    }

    @Synchronized
    fun connection(token: Any, value: Boolean)
    {
        val changedConnection = if (value) connectionState.connect(token) else connectionState.disconnect(token)
        if (changedConnection)
        {
            windowPackage = null
            windowClass = null
            windowId = -1
            unsafeReason = if (value) "Waiting for an accessibility window event" else "Accessibility labeling not connected"
            status = if (value) "Waiting for a foreground window event" else "Accessibility service disconnected"
            changed()
        }
    }

    @Synchronized
    fun windowState(token: Any, packageName: String?, className: String?, eventWindowId: Int, focusedWindowId: Int, reason: String?)
    {
        if (!connectionState.owns(token)) { return }
        if (windowId != focusedWindowId || unsafeReason != reason)
        {
            windowId = focusedWindowId
            unsafeReason = reason
            windowPackage = null
            windowClass = null
            changed()
        }
        if (packageName != null && eventWindowId == focusedWindowId && (packageName != windowPackage || (className != null && className != windowClass)))
        {
            if (windowPackage != packageName) { windowClass = null }
            windowPackage = packageName
            if (className != null) { windowClass = className }
            changed()
        }
    }

    @Synchronized
    fun pollUsage()
    {
        if (hasUsagePermission() == false)
        {
            if (usagePackage != null) { changed() }
            usagePackage = null
            usageActivity = null
            return
        }
        val now = System.currentTimeMillis()
        if (now < lastUsagePoll)
        {
            lastUsageTimestamp = 0L
            usageKeys.clear()
            usagePackage = null
            usageActivity = null
            changed()
        }
        val start = if (lastUsagePoll == 0L || now < lastUsagePoll) now - 60000 else lastUsagePoll - 1000
        val events = context.getSystemService(UsageStatsManager::class.java).queryEvents(start, now)
        val event = UsageEvents.Event()
        while (events != null && events.hasNextEvent())
        {
            events.getNextEvent(event)
            val key = "${event.timeStamp}/${event.eventType}/${event.packageName}/${event.className}"
            if (event.eventType != UsageEvents.Event.ACTIVITY_RESUMED && event.eventType != UsageEvents.Event.ACTIVITY_PAUSED && event.eventType != UsageEvents.Event.ACTIVITY_STOPPED) { continue }
            if (event.timeStamp < lastUsageTimestamp) { continue }
            if (event.timeStamp > lastUsageTimestamp)
            {
                lastUsageTimestamp = event.timeStamp
                usageKeys.clear()
            }
            if (usageKeys.add(key) == false) { continue }
            if (event.eventType == UsageEvents.Event.ACTIVITY_RESUMED)
            {
                if (usagePackage != event.packageName || usageActivity != event.className)
                {
                    usagePackage = event.packageName
                    usageActivity = event.className?.take(512)
                    changed()
                }
            }
            else if ((event.eventType == UsageEvents.Event.ACTIVITY_PAUSED || event.eventType == UsageEvents.Event.ACTIVITY_STOPPED) && usagePackage == event.packageName && usageActivity == event.className)
            {
                usagePackage = null
                usageActivity = null
                changed()
            }
        }
        lastUsagePoll = now
    }

    @Synchronized
    fun snapshot(): AppLabel?
    {
        if (!connectionState.isConnected()) { status = "Accessibility service disconnected"; return null }
        if (context.getSystemService(KeyguardManager::class.java).isKeyguardLocked) { status = "Device locked"; return null }
        if (hasUsagePermission() == false) { status = "Usage access revoked"; return null }
        if (unsafeReason != null) { status = unsafeReason!!; return null }
        val pkg = windowPackage
        if (pkg == null) { status = "Waiting for a foreground window event"; return null }
        if (usagePackage != null && usagePackage != pkg) { status = "Foreground sources disagree"; return null }
        val age = (SystemClock.elapsedRealtime() - changedAt).coerceAtLeast(0L)
        status = pkg
        return AppLabel(pkg, if (usagePackage == pkg) usageActivity else null, windowClass, generation, age, age >= 300L)
    }

    private fun changed()
    {
        generation += 1
        changedAt = SystemClock.elapsedRealtime()
    }
}

class LabelAccessibilityService : AccessibilityService()
{
    private val app get() = application as RecorderApp

    override fun onServiceConnected()
    {
        super.onServiceConnected()
        app.labels.connection(this, true)
        app.accessibilityService=this
        app.resumeRequestedRecording()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?)
    {
        if (event == null) { return }
        // Only window type/focus/bounds and the root package name are inspected.
        // Never traverse children or read node text, event.text, or content descriptions.
        val visible = windows.filter {
            val bounds = Rect()
            it.getBoundsInScreen(bounds)
            bounds.isEmpty == false
        }
        val applications = visible.filter { it.type == AccessibilityWindowInfo.TYPE_APPLICATION }
        val focused = applications.firstOrNull { it.isFocused && it.isActive }
        val keyboard = visible.any { it.type == AccessibilityWindowInfo.TYPE_INPUT_METHOD }
        val systemOverlay = visible.any { it.type == AccessibilityWindowInfo.TYPE_SYSTEM && it.isFocused }
        val reason = when
        {
            keyboard -> "Keyboard visible; collection paused"
            systemOverlay -> "System overlay visible; collection paused"
            applications.size != 1 -> "Multiple or no application windows; collection paused"
            focused == null -> "No unambiguous focused app window"
            else -> null
        }
        val isState = event.eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED
        val root = focused?.root
        val focusedPackage = root?.packageName?.toString()
        @Suppress("DEPRECATION")
        root?.recycle()
        val sameWindow = event.windowId == focused?.id
        val packageName = focusedPackage ?: if (isState && sameWindow) event.packageName?.toString() else null
        app.labels.windowState(this, packageName, if (isState && sameWindow) event.className?.toString()?.take(512) else null, focused?.id ?: -1, focused?.id ?: -1, reason)
    }

    // Android interrupts feedback without unbinding the service (for example when
    // another accessibility client requests control). This is not loss of permission.
    override fun onInterrupt() {}
    private fun disconnected() {
        app.labels.connection(this,false)
        if(app.accessibilityService===this) app.accessibilityService=null
    }
    override fun onUnbind(intent: android.content.Intent?): Boolean { disconnected(); return super.onUnbind(intent) }
    override fun onDestroy() { disconnected(); super.onDestroy() }
}
