package ro.ubb.uicollector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.android.OpenCVLoader
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.sqrt

/** Explicit diagnostic inputs only; raw captures are removed after use. */
class StaticScreenInstrumentedTest {
    private fun difference(a: FloatArray, b: FloatArray): JSONObject = JSONObject()
        .put("cosine",cosine(a,b))
        .put("l2_distance",sqrt(a.indices.sumOf { val d=a[it].toDouble()-b[it];d*d }))
        .put("maximum_absolute_difference",a.indices.maxOf { abs(a[it].toDouble()-b[it]) })
        .put("bitwise_equal",a.indices.all { a[it].toRawBits()==b[it].toRawBits() })

    private fun hash(input: ByteBuffer): String {
        val bytes=ByteArray(input.capacity())
        input.duplicate().apply { clear();get(bytes) }
        return MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }

    @Test fun repeatedFullScreenCapturesAndIdenticalInputAreConsistent() {
        val instrumentation=InstrumentationRegistry.getInstrumentation()
        val app=instrumentation.targetContext.applicationContext as RecorderApp
        val args=InstrumentationRegistry.getArguments()
        assertEquals("Explicit diagnostic invocation required", "true",args.getString("static_screen_probe"))
        assertTrue(OpenCVLoader.initLocal())
        val files=(0..2).map { File(app.cacheDir,"static-screen-$it.png") }
        val inputs=mutableListOf<String>()
        val embeddings=mutableListOf<FloatArray>()
        val repeated=JSONArray()
        val captures=JSONArray()
        try {
            EmbeddingEngine.create(app).use { engine ->
                val useGpu=app.preferences.getString("selected_preprocessor","opencv_cpu")=="opencl"
                val gpu=if(useGpu) OpenClPreprocessor() else null
                val cpu=if(useGpu) null else FramePreprocessor()
                try {
                    files.forEachIndexed { index,file ->
                        val decoded=checkNotNull(BitmapFactory.decodeFile(file.path)) { "Missing diagnostic capture $index" }
                        val bitmap=checkNotNull(decoded.copy(Bitmap.Config.ARGB_8888,false))
                        decoded.recycle()
                        try {
                            val rgba=ByteBuffer.allocateDirect(bitmap.rowBytes*bitmap.height).order(ByteOrder.nativeOrder())
                            bitmap.copyPixelsToBuffer(rgba);rgba.rewind()
                            val frame=CapturedFrame(rgba,bitmap.width,bitmap.height,bitmap.rowBytes,Rect(0,0,bitmap.width,bitmap.height),0L)
                            repeat(if(index==0) 20 else 1) { iteration ->
                                val input=if(gpu!=null) { gpu.ingest(frame);gpu.prepare(bitmap.width,bitmap.height) }
                                    else { cpu!!.ingest(frame);cpu.prepare() }
                                val fingerprint=hash(input)
                                val vector=engine.encode(input)
                                if(iteration==0) {
                                    inputs.add(fingerprint);embeddings.add(vector)
                                    captures.put(JSONObject().put("index",index).put("width",bitmap.width).put("height",bitmap.height).put("normalized_input_sha256",fingerprint))
                                } else {
                                    assertEquals("Preprocessing changed for identical RGBA input",inputs[0],fingerprint)
                                    val delta=difference(embeddings[0],vector)
                                    assertTrue("Identical input GPU inference changed materially",delta.getDouble("cosine")>=.999999)
                                    repeated.put(delta)
                                }
                            }
                        } finally { bitmap.recycle() }
                    }
                    val pairs=JSONArray()
                    for(index in 1..2) pairs.put(difference(embeddings[0],embeddings[index])
                        .put("pair",JSONArray(listOf(0,index))).put("normalized_inputs_identical",inputs[0]==inputs[index]))
                    pairs.put(difference(embeddings[1],embeddings[2]).put("pair",JSONArray(listOf(1,2)))
                        .put("normalized_inputs_identical",inputs[1]==inputs[2]))
                    File(app.cacheDir,"static-screen-report.json").writeText(JSONObject()
                        .put("model_id",engine.modelId).put("inference_backend",engine.backend)
                        .put("preprocessing_backend",if(useGpu) "opencl" else "opencv_cpu")
                        .put("captures",captures).put("pairs",pairs).put("identical_capture_repeats",repeated).toString(2))
                } finally { gpu?.close();cpu?.close() }
            }
        } finally {
            files.forEach { it.delete() }
            if(args.getString("resume_recording")=="true")
                app.preferences.edit().putBoolean("continuous_recording_requested",true).commit()
        }
    }
}
