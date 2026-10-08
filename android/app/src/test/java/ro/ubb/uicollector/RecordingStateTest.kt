package ro.ubb.uicollector

import org.junit.Assert.*
import org.junit.Test

class RecordingStateTest {
    @Test fun freshPolicyIncludesFutureAppsAndHonorsOptOut() {
        val settings=RecordingSettings(excluded=setOf("com.example.excluded"))
        assertTrue(settings.allows("com.example.installed_later","ro.ubb.uicollector"))
        assertFalse(settings.allows("com.example.excluded","ro.ubb.uicollector"))
        assertFalse(settings.allows("ro.ubb.uicollector","ro.ubb.uicollector"))
        assertFalse(settings.allows("com.android.systemui","ro.ubb.uicollector"))
        assertFalse(RecordingSettings(policy=CapturePolicy.LEGACY_SELECTED,legacySelected=setOf("com.example.a")).allows("com.example.b","ro.ubb.uicollector"))
    }
    @Test fun everyRequiredPermissionGatesReadiness() {
        val ready=PermissionState(GrantState.GRANTED,GrantState.GRANTED,GrantState.GRANTED)
        assertTrue(ready.ready)
        assertFalse(ready.copy(notifications=GrantState.DENIED).ready)
        assertFalse(ready.copy(accessibility=GrantState.NEEDS_SETTINGS).ready)
        assertFalse(ready.copy(usageAccess=GrantState.NEEDS_SETTINGS).ready)
    }
    @Test fun staleAccessibilityInstanceCannotDisconnectCurrentInstance() {
        val state=AccessibilityConnectionState()
        val first=Any()
        val second=Any()
        assertTrue(state.connect(first))
        assertTrue(state.connect(second))
        assertFalse(state.disconnect(first))
        assertTrue(state.isConnected())
        assertTrue(state.owns(second))
        assertTrue(state.disconnect(second))
        assertFalse(state.isConnected())
    }
    @Test fun modesStayBelowTheirCalibrationAndMetricsExpire() {
        assertEquals(2.0,RecordingMode.BALANCED.ceiling(2.0),0.0)
        assertEquals(1.0,RecordingMode.BATTERY_SAVER.ceiling(30.0),0.0)
        val metrics=RecordingMetrics()
        repeat(10) { metrics.commit(1_000_000_000L,20.0) }
        assertEquals(1.0,metrics.snapshot(2_000_000_000L).first,0.0)
        assertEquals(0.0,metrics.snapshot(12_000_000_000L).first,0.0)
        assertEquals(10L,metrics.total)
    }
}
