package ro.ubb.uicollector

import org.junit.Assert.*
import org.junit.Test

class RecordingStateTest {
    @Test fun freshPolicyIncludesFutureAppsAndHonorsOptOut() {
        val settings=RecordingSettings(excluded=setOf("com.example.excluded"))
        assertTrue(settings.allows("com.example.installed_later","ro.ubb.uicollector"))
        assertFalse(settings.allows("com.example.excluded","ro.ubb.uicollector"))
        assertFalse(settings.allows("ro.ubb.uicollector","ro.ubb.uicollector"))
        assertTrue(settings.allows("com.android.systemui","ro.ubb.uicollector"))
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
        assertEquals(1.0,RecordingMode.BALANCED.ceiling(2.0),0.0)
        assertEquals(1.0,RecordingMode.BATTERY_SAVER.ceiling(30.0),0.0)
        val metrics=RecordingMetrics()
        repeat(10) { metrics.commit(1_000_000_000L,20.0) }
        assertEquals(1.0,metrics.snapshot(2_000_000_000L).first,0.0)
        assertEquals(0.0,metrics.snapshot(12_000_000_000L).first,0.0)
        assertEquals(10L,metrics.total)
    }
    @Test fun modesRemainDistinctEvenAtLowCaptureRates() {
        listOf(0.25,1.0,2.5,3.5,5.0,30.0).forEach { calibrated ->
            val maximum=RecordingMode.MAXIMUM_DETAIL.ceiling(calibrated)
            val balanced=RecordingMode.BALANCED.ceiling(calibrated)
            val battery=RecordingMode.BATTERY_SAVER.ceiling(calibrated)
            assertTrue(maximum>balanced && balanced>battery)
            val controller=RateController(battery)
            repeat(400) { controller.observe(10.0,3) }
            assertTrue(controller.fps<=battery)
        }
        assertEquals(1.75,RecordingMode.BALANCED.ceiling(3.5),0.0)
        assertEquals(0.875,RecordingMode.BATTERY_SAVER.ceiling(3.5),0.0)
    }
}
