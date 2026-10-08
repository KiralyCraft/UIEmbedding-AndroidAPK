package ro.ubb.uicollector

import android.media.Image
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

/** Raw pixels live only in RAM. Full-resolution capture avoids a hidden second resize. */
class FramePreprocessor : AutoCloseable
{
    private val rgb = Mat()
    private val resized = Mat()
    private val canvas = Mat(800, 384, CvType.CV_8UC3)
    private val normalized = Mat()
    private val floats = FloatArray(800 * 384 * 3)
    val input: ByteBuffer = ByteBuffer.allocateDirect(floats.size * 4).order(ByteOrder.nativeOrder())
    var sourceWidth = 0
        private set
    var sourceHeight = 0
        private set
    var hasFrame = false
        private set
    var imageTimestampNs = 0L
        private set

    fun ingest(image: Image)
    {
        val plane = image.planes[0]
        require(plane.pixelStride == 4) { "RGBA ImageReader pixel stride must be 4" }
        val crop = image.cropRect
        val wrapped = Mat(image.height, image.width, CvType.CV_8UC4, plane.buffer, plane.rowStride.toLong())
        val region = wrapped.submat(Rect(crop.left, crop.top, crop.width(), crop.height()))
        try
        {
            Imgproc.cvtColor(region, rgb, Imgproc.COLOR_RGBA2RGB)
        }
        finally
        {
            region.release()
            wrapped.release()
        }
        sourceWidth = crop.width()
        sourceHeight = crop.height()
        imageTimestampNs = image.timestamp.coerceAtLeast(0L)
        hasFrame = true
    }

    fun prepare(): ByteBuffer
    {
        check(hasFrame)
        val scale = min(384.0 / sourceWidth, 800.0 / sourceHeight)
        // Math.rint matches Python's round-to-nearest, ties-to-even size calculation.
        val width = Math.rint(sourceWidth * scale).toInt().coerceIn(1, 384)
        val height = Math.rint(sourceHeight * scale).toInt().coerceIn(1, 800)
        Imgproc.resize(rgb, resized, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, if (scale < 1.0) Imgproc.INTER_AREA else Imgproc.INTER_LINEAR)
        canvas.setTo(Scalar(124.0, 116.0, 104.0))
        val region = canvas.submat(Rect((384 - width) / 2, (800 - height) / 2, width, height))
        try { resized.copyTo(region) } finally { region.release() }
        canvas.convertTo(normalized, CvType.CV_32FC3, 1.0 / 255.0)
        Core.subtract(normalized, Scalar(0.485, 0.456, 0.406), normalized)
        Core.divide(normalized, Scalar(0.229, 0.224, 0.225), normalized)
        normalized.get(0, 0, floats)
        input.rewind()
        input.asFloatBuffer().put(floats)
        return input
    }

    fun clear()
    {
        hasFrame = false
        rgb.release()
    }

    override fun close()
    {
        hasFrame = false
        listOf(rgb, resized, canvas, normalized).forEach { it.release() }
    }
}
