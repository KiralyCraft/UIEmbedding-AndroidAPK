package ro.ubb.uicollector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Base64
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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.sqrt

/** Explicit diagnostic inputs and SHA-256 checksums only; raw captures are removed after use. */
class StaticScreenInstrumentedTest {
    private fun liveFrame(app: RecorderApp): CapturedFrame {
        val deadline=SystemClock.elapsedRealtime()+20_000
        while(app.accessibilityService==null && SystemClock.elapsedRealtime()<deadline) SystemClock.sleep(50)
        checkNotNull(app.accessibilityService) { "Rebind Accessibility while the live diagnostic is waiting" }
        val thread=HandlerThread("diagnostic-screenshot").apply { start() }
        val frame=AtomicReference<CapturedFrame>()
        val error=AtomicReference<String>()
        val done=CountDownLatch(1)
        try {
            AccessibilityCapture(app,Handler(thread.looper)).request(true) { value,reason ->
                if(value!=null) {
                    val copy=ByteBuffer.allocateDirect(value.buffer.capacity()).order(ByteOrder.nativeOrder())
                    copy.put(value.buffer.duplicate().apply { clear() });copy.rewind()
                    frame.set(value.copy(buffer=copy,cropRect=Rect(value.cropRect)))
                }
                error.set(reason);done.countDown()
            }
            check(done.await(10,TimeUnit.SECONDS)) { "Live screenshot timed out" }
            return checkNotNull(frame.get()) { error.get() ?: "No live screenshot" }
        } finally { thread.quitSafely() }
    }

    private data class HistoricalEmbedding(val vector: FloatArray, val sampleWallMs: Long, val visitEndWallMs: Long, val details: JSONObject)

    private fun historicalEmbedding(app: RecorderApp, packageName: String, beforeWallMs: Long): HistoricalEmbedding {
        val owner=app.captureOwner()
        var latest: HistoricalEmbedding? = null
        app.store.readableDatabase.rawQuery("SELECT id,metadata FROM runs WHERE owner=? AND closed=1",arrayOf(owner)).use { runs ->
            while(runs.moveToNext()) {
                val id=runs.getString(0)
                val metadata=JSONObject(String(app.vault.open(runs.getBlob(1),"$owner/run/$id"),Charsets.UTF_8))
                val end=metadata.getLong("end_wall_ms")
                if(metadata.getString("package_name")!=packageName || end>beforeWallMs) continue
                app.store.readableDatabase.rawQuery("SELECT sequence,payload FROM samples WHERE run_id=? ORDER BY sequence DESC LIMIT 1",arrayOf(id)).use { samples ->
                    if(samples.moveToFirst()) {
                        val sequence=samples.getInt(0)
                        val sample=JSONObject(String(app.vault.open(samples.getBlob(1),"$owner/sample/$id/$sequence"),Charsets.UTF_8))
                        val wall=sample.getLong("wall_ms")
                        if(wall<=beforeWallMs && (latest==null || wall>latest!!.sampleWallMs)) {
                            val buffer=ByteBuffer.wrap(Base64.decode(sample.getString("embedding_b64"),Base64.NO_WRAP)).order(ByteOrder.LITTLE_ENDIAN)
                            check(buffer.remaining()==384*4)
                            val details=JSONObject().put("model_id",metadata.getString("model_id"))
                            for(key in listOf("processing_backend","preprocessing_backend","capture_source","source_width","source_height")) details.put(key,sample.opt(key))
                            latest=HistoricalEmbedding(FloatArray(384) { buffer.float },wall,end,details)
                        }
                    }
                }
            }
        }
        return checkNotNull(latest) { "No retained closed visit for the requested package and time" }
    }

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
        val rawFrames=mutableListOf<ByteArray>()
        try {
            EmbeddingEngine.create(app).use { engine ->
                val useGpu=app.preferences.getString("selected_preprocessor","opencv_cpu")=="opencl"
                val gpu=if(useGpu) OpenClPreprocessor() else null
                val cpu=if(useGpu) null else FramePreprocessor()
                try {
                    files.forEachIndexed { index,file ->
                        val png=file.readBytes()
                        val iend=byteArrayOf(0,0,0,0,0x49,0x45,0x4e,0x44,0xae.toByte(),0x42,0x60,0x82.toByte())
                        check(png.size>=12 && png.takeLast(12).toByteArray().contentEquals(iend)) { "Diagnostic PNG $index is truncated: missing IEND" }
                        val pngHash=MessageDigest.getInstance("SHA-256").digest(png).joinToString("") { "%02x".format(it) }
                        val expectedHash=checkNotNull(args.getString("capture_sha256_$index")) { "Supply capture_sha256_$index from the complete source PNG" }
                        check(expectedHash==pngHash) { "Diagnostic PNG $index transfer checksum mismatch" }
                        val decoded=checkNotNull(BitmapFactory.decodeFile(file.path)) { "Missing diagnostic capture $index" }
                        val bitmap=checkNotNull(decoded.copy(Bitmap.Config.ARGB_8888,false))
                        decoded.recycle()
                        try {
                            val rgba=ByteBuffer.allocateDirect(bitmap.rowBytes*bitmap.height).order(ByteOrder.nativeOrder())
                            bitmap.copyPixelsToBuffer(rgba);rgba.rewind()
                            val raw=ByteArray(rgba.capacity()).also { rgba.duplicate().get(it) }
                            rawFrames.add(raw)
                            val frame=CapturedFrame(rgba,bitmap.width,bitmap.height,bitmap.rowBytes,Rect(0,0,bitmap.width,bitmap.height),0L)
                            repeat(if(index==0) 20 else 1) { iteration ->
                                val input=if(gpu!=null) { gpu.ingest(frame);gpu.prepare(bitmap.width,bitmap.height) }
                                    else { cpu!!.ingest(frame);cpu.prepare() }
                                val fingerprint=hash(input)
                                val vector=engine.encode(input)
                                if(iteration==0) {
                                    inputs.add(fingerprint);embeddings.add(vector)
                                    captures.put(JSONObject().put("index",index).put("width",bitmap.width).put("height",bitmap.height)
                                        .put("png_sha256",pngHash).put("png_bytes",png.size)
                                        .put("raw_rgba_sha256",hash(rgba)).put("row_stride",bitmap.rowBytes)
                                        .put("color_space",bitmap.colorSpace?.toString()).put("normalized_input_sha256",fingerprint))
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
                        .put("pair",JSONArray(listOf(0,index))).put("normalized_inputs_identical",inputs[0]==inputs[index])
                        .put("different_rgba_bytes",rawFrames[0].indices.count { rawFrames[0][it]!=rawFrames[index][it] }))
                    pairs.put(difference(embeddings[1],embeddings[2]).put("pair",JSONArray(listOf(1,2)))
                        .put("normalized_inputs_identical",inputs[1]==inputs[2]))
                    val report=JSONObject()
                        .put("model_id",engine.modelId).put("inference_backend",engine.backend)
                        .put("preprocessing_backend",if(useGpu) "opencl" else "opencv_cpu")
                        .put("captures",captures).put("pairs",pairs).put("identical_capture_repeats",repeated)
                    args.getString("baseline_package")?.let { packageName ->
                        val baseline=historicalEmbedding(app,packageName,checkNotNull(args.getString("baseline_before_wall_ms")).toLong())
                        report.put("historical_comparison",JSONObject().put("package",packageName)
                            .put("sample_wall_ms",baseline.sampleWallMs).put("visit_end_wall_ms",baseline.visitEndWallMs).put("details",baseline.details)
                            .put("comparisons",JSONArray().apply { embeddings.forEachIndexed { index,vector -> put(difference(baseline.vector,vector).put("capture_index",index)) } }))
                        if(args.getString("retain_vectors")=="true") report.getJSONObject("historical_comparison").put("embedding",JSONArray(baseline.vector.map { it.toDouble() }))
                        if(args.getString("live_accessibility_probe")=="true") {
                            val live=liveFrame(app)
                            val input=if(gpu!=null) { gpu.ingest(live);gpu.prepare(live.cropRect.width(),live.cropRect.height()) }
                                else { cpu!!.ingest(live);cpu.prepare() }
                            val liveVector=engine.encode(input)
                            val recent=historicalEmbedding(app,packageName,System.currentTimeMillis())
                            report.put("live_accessibility",JSONObject().put("width",live.width).put("height",live.height).put("row_stride",live.rowStride)
                                .put("vs_png_capture",difference(liveVector,embeddings[0]))
                                .put("vs_original_recorded",difference(liveVector,baseline.vector))
                                .put("vs_latest_recorded",difference(liveVector,recent.vector)).put("latest_recorded_details",recent.details)
                                .put("latest_recorded_wall_ms",recent.sampleWallMs))
                        }
                    }
                    if(args.getString("retain_vectors")=="true") report.put("embeddings",JSONArray().apply {
                        embeddings.forEach { vector -> put(JSONArray(vector.map { it.toDouble() })) }
                    })
                    File(app.cacheDir,"static-screen-report.json").writeText(report.toString(2))
                } finally { gpu?.close();cpu?.close() }
            }
        } finally {
            files.forEach { it.delete() }
            if(args.getString("resume_recording")=="true")
                app.preferences.edit().putBoolean("continuous_recording_requested",true).commit()
        }
    }
}
