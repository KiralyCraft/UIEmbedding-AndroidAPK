package ro.ubb.uicollector

import androidx.test.platform.app.InstrumentationRegistry
import android.os.Build
import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.sqrt

class GpuParityInstrumentedTest
{
    @Test
    fun allNineSyntheticFullScreensMatchTrainedCheckpoint() {
        val instrumentation=InstrumentationRegistry.getInstrumentation()
        EmbeddingEngine(instrumentation.targetContext).use { engine ->
            val inputBytes=instrumentation.context.assets.open("synthetic_inputs.bin").use { it.readBytes() }
            val expectedBytes=ByteBuffer.wrap(instrumentation.context.assets.open("synthetic_embeddings.bin").use { it.readBytes() }).order(ByteOrder.LITTLE_ENDIAN)
            val input=ByteBuffer.allocateDirect(800*384*3*4).order(ByteOrder.nativeOrder())
            var minimum=1.0
            repeat(9) { index ->
                input.rewind();input.put(inputBytes,index*input.capacity(),input.capacity());input.rewind()
                val expected=FloatArray(384) { expectedBytes.float }
                val actual=engine.encode(input)
                minimum=minOf(minimum,cosine(actual,expected))
                assertTrue("Input $index cosine=$minimum",minimum>=.9999)
            }
            Log.i("UICollectorInference","Full OpenCL encoder passed all nine synthetic screens; minimum cosine=$minimum")
        }
    }

    @Test
    fun trainedF6GoldenInputMatchesOnActualGpu()
    {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val executor = Executors.newSingleThreadExecutor()
        try
        {
            val result = executor.submit<Double> {
                EmbeddingEngine(context).use { engine ->
                    val bytes = context.assets.open("golden_input.bin").use { it.readBytes() }
                    val input = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
                    input.put(bytes).rewind()
                    val expectedBytes = context.assets.open("golden_embedding.bin").use { it.readBytes() }
                    val expectedBuffer = ByteBuffer.wrap(expectedBytes).order(ByteOrder.LITTLE_ENDIAN)
                    val expected = FloatArray(384) { expectedBuffer.float }
                    repeat(5) { engine.encode(input) }
                    val totalMs = mutableListOf<Double>()
                    val gpuMs = mutableListOf<Double>()
                    val headMs = mutableListOf<Double>()
                    var minimumCosine = 1.0
                    var maximumError = 0.0
                    repeat(100) {
                        val start = SystemClock.elapsedRealtimeNanos()
                        val embedding = engine.encode(input)
                        totalMs.add((SystemClock.elapsedRealtimeNanos() - start) / 1e6)
                        gpuMs.add(engine.inferenceMs)
                        headMs.add(engine.headMs)
                        assertEquals(384, embedding.size)
                        assertTrue(embedding.all { value -> value.isFinite() })
                        val norm = sqrt(embedding.sumOf { value -> value.toDouble() * value })
                        assertEquals(1.0, norm, 1e-5)
                        minimumCosine = minOf(minimumCosine, cosine(embedding, expected))
                        maximumError = maxOf(maximumError, embedding.indices.maxOf { index -> abs(embedding[index].toDouble() - expected[index]) })
                    }
                    assertTrue("Repeated GPU parity cosine=$minimumCosine", minimumCosine >= 0.9999)
                    val report = JSONObject()
                        .put("model_id", engine.modelId)
                        .put("backend",engine.backend)
                        .put("device", Build.MODEL)
                        .put("android_sdk", Build.VERSION.SDK_INT)
                        .put("input", "synthetic_export_golden")
                        .put("iterations", totalMs.size)
                        .put("warmup_iterations", 5)
                        .put("embedding_dim", 384)
                        .put("minimum_cosine", minimumCosine)
                        .put("maximum_absolute_error", maximumError)
                        .put("total_p50_ms", percentile(totalMs, 0.5))
                        .put("total_p95_ms", percentile(totalMs, 0.95))
                        .put("gpu_p50_ms", percentile(gpuMs, 0.5))
                        .put("gpu_p95_ms", percentile(gpuMs, 0.95))
                        .put("head_p50_ms", percentile(headMs, 0.5))
                        .put("head_p95_ms", percentile(headMs, 0.95))
                        .put("scope", "model invocation only; excludes capture, preprocessing, storage and upload")
                    File(context.cacheDir, "gpu-inference-report.json").writeText(report.toString(2))
                    Log.i("UICollectorInference", report.toString())
                    minimumCosine
                }
            }.get(120, TimeUnit.SECONDS)
            assertTrue(result >= 0.9999)
        }
        finally { executor.shutdownNow() }
    }
}
