package ro.ubb.uicollector

import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

/** Full-screen animated input, kept visible throughout end-to-end calibration. */
class BenchmarkActivity : Activity()
{
    private val app get() = application as RecorderApp
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var status: TextView
    private var seenRunning = false
    private val update = object : Runnable
    {
        override fun run()
        {
            status.text = app.captureStatus
            if (app.captureActive) { seenRunning = true }
            val calibrated = seenRunning && app.captureActive && app.calibrating == false
            if (seenRunning && (app.captureActive == false || calibrated))
            {
                finish()
                return
            }
            handler.postDelayed(this, 500)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?)
    {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val layout = LinearLayout(this)
        layout.orientation = LinearLayout.VERTICAL
        status = TextView(this)
        status.text = "Initializing. Leave this moving pattern visible while calibration runs."
        layout.addView(status)
        layout.addView(CalibrationView(this), LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1.0f))
        val back = Button(this)
        back.text = "Return to controls"
        back.setOnClickListener { finish() }
        layout.addView(back)
        setContentView(layout)
    }

    override fun onResume() { super.onResume(); handler.post(update) }
    override fun onPause() { handler.removeCallbacks(update); super.onPause() }
    override fun onDestroy() { handler.removeCallbacksAndMessages(null); super.onDestroy() }
}
