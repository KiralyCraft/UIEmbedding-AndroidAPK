package ro.ubb.uicollector

import org.junit.Assert.*
import org.junit.Test

class WindowPolicyTest {
    private val app=VisibleWindow(1,WindowKind.APPLICATION,1,true,true,"com.termux.x11")
    private val keyboard=VisibleWindow(2,WindowKind.INPUT_METHOD,5,true,true,"com.example.keyboard")
    @Test fun keyboardKeepsHostApplicationEvenIfImeTakesFocus() {
        for(packageName in listOf("com.termux","com.termux.x11")) for(host in listOf(app.copy(packageName=packageName),app.copy(packageName=packageName,focused=false,active=false))) {
            val resolved=resolveCaptureWindow(listOf(host,keyboard))!!
            assertEquals(packageName,resolved.primary.packageName)
            assertTrue(resolved.context.keyboardVisible)
            assertEquals("application",resolved.context.screenKind)
        }
    }
    @Test fun drawerLabelsSystemUiInsteadOfUnderlyingApp() {
        val drawer=VisibleWindow(3,WindowKind.SYSTEM,9,true,true,"com.android.systemui")
        val result=resolveCaptureWindow(listOf(app,keyboard,drawer))!!
        assertEquals("com.android.systemui",result.primary.packageName)
        assertEquals("system_overlay",result.context.screenKind)
        assertEquals("com.termux.x11",resolveCaptureWindow(listOf(app,keyboard))!!.primary.packageName)
    }
    @Test fun passiveStatusBarDoesNotReplaceApplicationLabel() {
        val bar=VisibleWindow(3,WindowKind.SYSTEM,9,false,false,"com.android.systemui")
        assertEquals("com.termux.x11",resolveCaptureWindow(listOf(app,bar))!!.primary.packageName)
    }
    @Test fun focusedSplitScreenAppWinsAndExclusionsCoverVisibleContent() {
        val second=VisibleWindow(4,WindowKind.APPLICATION,2,false,false,"com.example.private")
        val resolved=resolveCaptureWindow(listOf(app,second,keyboard))!!
        assertEquals(app,resolved.primary)
        val settings=RecordingSettings(excluded=setOf("com.example.private"))
        assertFalse(settings.allowsScreen(resolved.primary.packageName!!,resolved.context.visiblePackages,"collector"))
        assertTrue(settings.allowsScreen(app.packageName!!,listOf(app.packageName!!,keyboard.packageName!!),"collector"))
    }
    @Test fun missingIdentityIsNotGuessedFromKeyboardOrUnderlyingApp() {
        assertNull(resolveCaptureWindow(listOf(keyboard)))
        assertNull(resolveCaptureWindow(listOf(app.copy(packageName=null))))
        assertNull(resolveCaptureWindow(listOf(app,VisibleWindow(3,WindowKind.SYSTEM,9,true,true,null))))
    }
}
