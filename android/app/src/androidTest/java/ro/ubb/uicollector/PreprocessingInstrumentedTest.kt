package ro.ubb.uicollector

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PixelFormat
import android.media.ImageReader
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.android.OpenCVLoader
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/** Exercises the production RGBA ImageReader -> OpenCV path without screen consent or a server. */
class PreprocessingInstrumentedTest
{
    @Test
    fun syntheticFullFrameMatchesExportPreprocessing()
    {
        assertTrue(OpenCVLoader.initLocal())
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        // Same first synthetic full-screen input as tools/export_f6.py.
        val width = 1080
        val height = 1920
        val pixels = IntArray(width * height) { index ->
            val x = index % width
            val y = index / width
            if (y in height / 5 until height / 3 && x in width / 6 until 5 * width / 6)
                Color.rgb(230, 230, 230)
            else
                Color.rgb((x / 7 + y / 3) % 256, (x / 13 * 29 + y / 37 * 43) % 256, (x * 3 + y * 7) % 256)
        }
        val bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        try
        {
            ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2).use { reader ->
                val canvas = reader.surface.lockCanvas(null)
                try { canvas.drawBitmap(bitmap, 0f, 0f, null) }
                finally { reader.surface.unlockCanvasAndPost(canvas) }
                val deadline = SystemClock.elapsedRealtime() + 5000
                var image = reader.acquireNextImage()
                while (image == null && SystemClock.elapsedRealtime() < deadline)
                {
                    SystemClock.sleep(10)
                    image = reader.acquireNextImage()
                }
                checkNotNull(image) { "Synthetic ImageReader frame was not delivered" }.use { frame ->
                    val plane = frame.planes[0]
                    val source = plane.buffer.duplicate()
                    var maximumSourceError = 0
                    for (y in 0 until height)
                        for (x in 0 until width)
                        {
                            val offset = y * plane.rowStride + x * plane.pixelStride
                            val expected = pixels[y * width + x]
                            maximumSourceError = maxOf(maximumSourceError,
                                abs((source.get(offset).toInt() and 255) - Color.red(expected)),
                                abs((source.get(offset + 1).toInt() and 255) - Color.green(expected)),
                                abs((source.get(offset + 2).toInt() and 255) - Color.blue(expected)))
                        }
                    assertEquals("ImageReader must preserve synthetic RGB exactly", 0, maximumSourceError)
                    FramePreprocessor().use { preprocessor ->
                        preprocessor.ingest(frame)
                        assertEquals(width, preprocessor.sourceWidth)
                        assertEquals(height, preprocessor.sourceHeight)
                        val actual = preprocessor.prepare().asFloatBuffer()
                        val golden = ByteBuffer.wrap(context.assets.open("golden_input.bin").use { it.readBytes() })
                            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                        assertEquals(golden.remaining(), actual.remaining())
                        var maximumError = 0.0
                        var maximumRgbError = 0.0
                        var index = 0
                        val standardDeviation = doubleArrayOf(0.229, 0.224, 0.225)
                        while (golden.hasRemaining())
                        {
                            val error = abs(actual.get().toDouble() - golden.get())
                            maximumError = maxOf(maximumError, error)
                            maximumRgbError = maxOf(maximumRgbError, error * standardDeviation[index % 3] * 255)
                            index++
                        }
                        // OpenCV uint8 INTER_AREA rounding can differ between ARM and x86.
                        // Bound it to one RGB level, then require full embedding parity as well.
                        assertTrue("Android resize RGB error=$maximumRgbError", maximumRgbError <= 1.0001)
                        val expectedBuffer = ByteBuffer.wrap(context.assets.open("golden_embedding.bin").use { it.readBytes() })
                            .order(ByteOrder.LITTLE_ENDIAN)
                        val expectedEmbedding = FloatArray(384) { expectedBuffer.float }
                        val embeddingCosine = EmbeddingEngine(context).use { engine ->
                            cosine(engine.encode(preprocessor.input), expectedEmbedding)
                        }
                        assertTrue("Preprocessed embedding cosine=$embeddingCosine", embeddingCosine >= 0.9999)
                        val report = JSONObject().put("passed", true).put("source_width", width)
                            .put("source_height", height).put("maximum_absolute_error", maximumError)
                            .put("maximum_source_rgb_error", maximumSourceError)
                            .put("maximum_resize_rgb_error", maximumRgbError)
                            .put("resize_rgb_error_limit", 1.0001)
                            .put("embedding_cosine", embeddingCosine)
                            .put("scope", "synthetic RGBA ImageReader frame through production OpenCV preprocessing")
                        File(context.cacheDir, "preprocessing-report.json").writeText(report.toString(2))
                    }
                }
            }
        }
        finally { bitmap.recycle() }
    }
}
