package ro.ubb.uicollector

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.provider.Settings

object Permissions {
    fun createChannel(context: Context) {
        context.getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel("capture", "Recording status", NotificationManager.IMPORTANCE_LOW)
        )
    }

    fun inspect(app: RecorderApp): PermissionState {
        val manager = app.getSystemService(NotificationManager::class.java)
        val runtime = Build.VERSION.SDK_INT < 33 || app.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        val enabled = manager.areNotificationsEnabled() && manager.getNotificationChannel("capture")?.importance != NotificationManager.IMPORTANCE_NONE
        val notifications = when {
            runtime && enabled -> GrantState.GRANTED
            !runtime && app.preferences.getBoolean("notification_requested", false) -> GrantState.DENIED
            !enabled && runtime -> GrantState.NEEDS_SETTINGS
            else -> GrantState.MISSING
        }
        val service = ComponentName(app, LabelAccessibilityService::class.java)
        val accessibilityEnabled = Settings.Secure.getString(app.contentResolver, Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES)
            .orEmpty().split(':').mapNotNull { ComponentName.unflattenFromString(it) }.contains(service)
        // The secure setting is the permission grant. Service binding is runtime
        // readiness and can briefly change while Android reconfigures accessibility;
        // it must not make an already-granted onboarding step flicker.
        val accessibility = if (accessibilityEnabled) GrantState.GRANTED else GrantState.NEEDS_SETTINGS
        return PermissionState(notifications, accessibility, if (app.labels.hasUsagePermission()) GrantState.GRANTED else GrantState.NEEDS_SETTINGS)
    }

    fun notificationSettings(context: Context): Intent = Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
        .putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName)
}
