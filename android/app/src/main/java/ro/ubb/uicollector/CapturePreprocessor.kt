package ro.ubb.uicollector

import android.media.Image
import android.os.SystemClock
import android.util.Log
import kotlin.math.abs
import java.nio.ByteBuffer

/** Compares complete invocations, including uploads/readbacks, before selecting OpenCL. */
class CapturePreprocessor(private val app: RecorderApp, private val model: EmbeddingEngine) : AutoCloseable {
    private val cpu = FramePreprocessor()
    private var gpu: OpenClPreprocessor? = runCatching { OpenClPreprocessor() }.onFailure { Log.i("UICollectorPreprocess", "OpenCL preprocessing unavailable: ${it.message}") }.getOrNull()
    init {
        app.preferences.edit().putString("gpu_driver",gpu?.driver ?: "unavailable").apply()
        if(gpu==null) app.preferences.edit().putString("selected_preprocessor","opencv_cpu").apply()
    }
    private var selected = false
    private var decided = false
    private var cpuIngestMs = 0.0
    private var gpuIngestMs = 0.0
    var processingMs = 0.0
        private set
    var sourceWidth = 0
        private set
    var sourceHeight = 0
        private set
    var imageTimestampNs = 0L
        private set
    var hasFrame = false
        private set
    val backend: String get() = if(selected) "opencl" else "opencv_cpu"

    fun ingest(image: Image) = ingest(CapturedFrame.from(image))
    fun ingest(image: CapturedFrame) {
        val crop=image.cropRect
        if(sourceWidth!=crop.width() || sourceHeight!=crop.height()) {
            decided=false;selected=false
            val cacheKey=app.calibrationSignature(model.backend)+":${crop.width()}x${crop.height()}"
            if(app.preferences.getString("preprocessing_signature","")==cacheKey) {
                selected=app.preferences.getBoolean("opencl_preprocessing",false) && gpu!=null
                decided=true
            }
        }
        sourceWidth=crop.width();sourceHeight=crop.height();imageTimestampNs=image.timestamp.coerceAtLeast(0L)
        if(!decided) choose(image)
        val start=SystemClock.elapsedRealtimeNanos()
        if(!decided || !selected) cpu.ingest(image)
        cpuIngestMs=(SystemClock.elapsedRealtimeNanos()-start)/1e6
        val gpuStart=SystemClock.elapsedRealtimeNanos()
        if((!decided || selected) && gpu!=null) gpu!!.ingest(image)
        gpuIngestMs=(SystemClock.elapsedRealtimeNanos()-gpuStart)/1e6
        hasFrame=true
    }

    fun prepare(): ByteBuffer {
        check(hasFrame)
        val start=SystemClock.elapsedRealtimeNanos()
        val result=if(selected) gpu!!.prepare(sourceWidth,sourceHeight) else cpu.prepare()
        processingMs=(if(selected) gpuIngestMs else cpuIngestMs)+(SystemClock.elapsedRealtimeNanos()-start)/1e6
        return result
    }

    private fun choose(image: CapturedFrame) {
        val candidate=gpu
        val cpuTimes=mutableListOf<Double>()
        val gpuTimes=mutableListOf<Double>()
        val cpuCopies=mutableListOf<Double>();val gpuCopies=mutableListOf<Double>()
        val cpuTransforms=mutableListOf<Double>();val gpuTransforms=mutableListOf<Double>()
        var maximumRgbError=0.0
        var minimumCosine=1.0
        var reason="OpenCL unavailable"
        if(candidate!=null) {
            try {
                repeat(24) { index ->
                    // Include a fresh upload/copy on every trial, discard allocation warmups,
                    // and alternate order so neither backend always follows a cold GPU.
                    lateinit var cpuInput: ByteBuffer;lateinit var gpuInput: ByteBuffer
                    lateinit var expected: FloatArray;lateinit var actual: FloatArray
                    fun probe(useGpu: Boolean) {
                        val start=SystemClock.elapsedRealtimeNanos()
                        if(useGpu) candidate.ingest(image) else cpu.ingest(image)
                        val copied=SystemClock.elapsedRealtimeNanos()
                        val input=if(useGpu) candidate.prepare(sourceWidth,sourceHeight) else cpu.prepare()
                        val prepared=SystemClock.elapsedRealtimeNanos()
                        val vector=model.encode(input)
                        val finished=SystemClock.elapsedRealtimeNanos()
                        if(useGpu) { gpuInput=input;actual=vector } else { cpuInput=input;expected=vector }
                        if(index>=4) {
                            (if(useGpu) gpuCopies else cpuCopies).add((copied-start)/1e6)
                            (if(useGpu) gpuTransforms else cpuTransforms).add((prepared-copied)/1e6)
                            (if(useGpu) gpuTimes else cpuTimes).add((finished-start)/1e6)
                        }
                    }
                    probe(index%2==0);probe(index%2!=0)
                    val baseline=cpuInput.asFloatBuffer();val comparison=gpuInput.asFloatBuffer()
                    var channel=0
                    val deviations=doubleArrayOf(.229,.224,.225)
                    while(baseline.hasRemaining()) {
                        maximumRgbError=maxOf(maximumRgbError,abs(baseline.get().toDouble()-comparison.get())*255*deviations[channel%3]);channel++
                    }
                    minimumCosine=minOf(minimumCosine,cosine(expected,actual))
                }
                selected=maximumRgbError<=1.0001 && minimumCosine>=.9999 && percentile(gpuTimes,.95)<=percentile(cpuTimes,.95)*.95
                reason=if(selected) "OpenCL passes parity and improves pipeline p95 by at least 5%" else "OpenCV retained: parity or pipeline improvement gate not met"
            } catch(error: Exception) { selected=false;reason="OpenCV retained: ${error.message}" }
        }
        decided=true
        val report=org.json.JSONObject().put("selected",backend).put("maximum_resize_rgb_error",maximumRgbError).put("minimum_embedding_cosine",minimumCosine)
            .put("cpu_pipeline_p95_ms",if(cpuTimes.isEmpty()) org.json.JSONObject.NULL else percentile(cpuTimes,.95))
            .put("opencl_pipeline_p95_ms",if(gpuTimes.isEmpty()) org.json.JSONObject.NULL else percentile(gpuTimes,.95))
            .put("cpu_ingest_p95_ms",if(cpuCopies.isEmpty()) org.json.JSONObject.NULL else percentile(cpuCopies,.95))
            .put("opencl_ingest_p95_ms",if(gpuCopies.isEmpty()) org.json.JSONObject.NULL else percentile(gpuCopies,.95))
            .put("cpu_transform_p95_ms",if(cpuTransforms.isEmpty()) org.json.JSONObject.NULL else percentile(cpuTransforms,.95))
            .put("opencl_transform_readback_p95_ms",if(gpuTransforms.isEmpty()) org.json.JSONObject.NULL else percentile(gpuTransforms,.95))
            .put("power_measurement","unavailable; latency selection is provisional for energy").put("reason",reason)
        app.preferences.edit().putString("selected_preprocessor",backend).commit()
        app.preferences.edit().putString("preprocessing_signature",app.calibrationSignature(model.backend)+":${sourceWidth}x${sourceHeight}")
            .putBoolean("opencl_preprocessing",selected).putString("preprocessing_comparison",report.toString()).commit()
        Log.i("UICollectorPreprocess",report.toString())
        if(selected) cpu.clear()
    }
    fun clear() { hasFrame=false;cpu.clear() }
    override fun close() { cpu.close();gpu?.close();gpu=null }
}
