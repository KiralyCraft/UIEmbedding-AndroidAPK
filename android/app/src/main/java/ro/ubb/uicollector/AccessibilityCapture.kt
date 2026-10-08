package ro.ubb.uicollector

import android.accessibilityservice.AccessibilityService
import android.app.KeyguardManager
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Handler
import android.os.PowerManager
import android.os.SystemClock
import android.view.Display
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executor
import java.util.zip.CRC32

/** One requested full-display frame at a time. Never captures while locked/off. */
class AccessibilityCapture(private val app: RecorderApp, private val handler: Handler) {
    private var buffer: ByteBuffer? = null
    private var previousFingerprint: Long? = null
    private var lastRequestMs = 0L
    var closed = false
    fun reset() { previousFingerprint=null }
    fun unlocked(): Boolean = app.getSystemService(PowerManager::class.java).isInteractive &&
        !app.getSystemService(KeyguardManager::class.java).isKeyguardLocked

    fun request(calibrating: Boolean, callback: (CapturedFrame?, String?) -> Unit) {
        val service=app.accessibilityService
        if(closed || !unlocked()) { callback(null,"Screen locked/off · recording resumes after unlock");return }
        if(service==null) { callback(null,"Waiting for Accessibility service");return }
        val delay=(lastRequestMs+400-SystemClock.elapsedRealtime()).coerceAtLeast(0)
        handler.postDelayed({
            if(closed || !unlocked()) { callback(null,"Screen locked/off · recording resumes after unlock");return@postDelayed }
            lastRequestMs=SystemClock.elapsedRealtime()
            try {
                service.takeScreenshot(Display.DEFAULT_DISPLAY,Executor { handler.post(it) },object: AccessibilityService.TakeScreenshotCallback {
                    override fun onSuccess(result: AccessibilityService.ScreenshotResult) {
                        val hardware=result.hardwareBuffer
                        var wrapped: Bitmap? = null
                        var pixels: Bitmap? = null
                        try {
                            if(closed || !unlocked()) { callback(null,"Screen locked/off · recording resumes after unlock");return }
                            wrapped=checkNotNull(Bitmap.wrapHardwareBuffer(hardware,result.colorSpace))
                            pixels=checkNotNull(wrapped!!.copy(Bitmap.Config.ARGB_8888,false))
                            val bitmap=pixels!!
                            val required=bitmap.rowBytes*bitmap.height
                            if(buffer?.capacity()!=required) buffer=ByteBuffer.allocateDirect(required).order(ByteOrder.nativeOrder())
                            val rgba=buffer!!;rgba.clear();bitmap.copyPixelsToBuffer(rgba);rgba.rewind()
                            val fingerprint=CRC32().apply { update(rgba.duplicate()) }.value
                            val duplicate=previousFingerprint==fingerprint
                            previousFingerprint=fingerprint
                            if(!calibrating && duplicate) callback(null,null)
                            else callback(CapturedFrame(rgba,bitmap.width,bitmap.height,bitmap.rowBytes,Rect(0,0,bitmap.width,bitmap.height),SystemClock.elapsedRealtimeNanos()),null)
                        } catch(error: Exception) { callback(null,"Screenshot unavailable: ${error.message}") }
                        finally { pixels?.recycle();wrapped?.recycle();hardware.close() }
                    }
                    override fun onFailure(errorCode: Int) {
                        callback(null,when(errorCode) {
                            AccessibilityService.ERROR_TAKE_SCREENSHOT_NO_ACCESSIBILITY_ACCESS -> "Enable screen capture for the collector in Accessibility settings"
                            AccessibilityService.ERROR_TAKE_SCREENSHOT_SECURE_WINDOW -> "Protected window · collection paused"
                            AccessibilityService.ERROR_TAKE_SCREENSHOT_INTERVAL_TIME_SHORT -> "Android capture rate limit · retrying"
                            else -> "Screenshot unavailable ($errorCode) · retrying"
                        })
                    }
                })
            } catch(error: Exception) { callback(null,"Screenshot unavailable: ${error.message}") }
        },delay)
    }
}
