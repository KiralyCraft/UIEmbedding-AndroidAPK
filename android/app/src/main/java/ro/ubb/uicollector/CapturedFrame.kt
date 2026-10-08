package ro.ubb.uicollector

import android.graphics.Rect
import android.media.Image
import java.nio.ByteBuffer

/** Borrowed RGBA pixels. The caller keeps their owning image/buffer alive until ingest returns. */
data class CapturedFrame(val buffer: ByteBuffer, val width: Int, val height: Int, val rowStride: Int, val cropRect: Rect, val timestamp: Long) {
    companion object {
        fun from(image: Image): CapturedFrame {
            val plane=image.planes[0]
            require(plane.pixelStride==4)
            return CapturedFrame(plane.buffer,image.width,image.height,plane.rowStride,image.cropRect,image.timestamp)
        }
    }
}
