package ro.ubb.uicollector

import android.view.View
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint

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
