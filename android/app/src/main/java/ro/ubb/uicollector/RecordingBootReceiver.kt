package ro.ubb.uicollector

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** Credential-protected preferences are available at BOOT_COMPLETED, after first unlock. */
class RecordingBootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if(intent.action!=Intent.ACTION_BOOT_COMPLETED) return
        (context.applicationContext as RecorderApp).resumeRequestedRecording()
    }
}
