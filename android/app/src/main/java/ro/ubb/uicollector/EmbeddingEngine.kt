package ro.ubb.uicollector

import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.security.MessageDigest

/** Create, invoke and close this object on ONE dedicated HandlerThread. No CPU fallback. */
class EmbeddingEngine(private val context: Context) : AutoCloseable
{
    val manifest: JSONObject = JSONObject(context.assets.open("f6_manifest.json").bufferedReader().use { it.readText() })
    val modelId: String = manifest.getString("model_id")
    private val ownerThread = Thread.currentThread().id
    private var delegate: GpuDelegate? = null
    private var interpreter: Interpreter? = null
    private val head: EmbeddingHead
    private val features = FloatArray(960)
    private val output = ByteBuffer.allocateDirect(960 * 4).order(ByteOrder.nativeOrder())
    var inferenceMs = 0.0
        private set
    var headMs = 0.0
        private set
    var parityCosine = 0.0
        private set

    init
    {
        check(manifest.getBoolean("export_parity_passed")) { "Model export parity was not validated" }
        check(manifest.getInt("embedding_dim") == 384 && manifest.getInt("backbone_feature_dim") == 960)
        val files = manifest.getJSONObject("asset_sha256")
        files.keys().forEach {
            check(hashAsset(it) == files.getString(it)) { "Asset checksum mismatch: $it" }
        }
        head = EmbeddingHead(context.assets.open("f6_head.bin").use { it.readBytes() })
        try
        {
            val options = GpuDelegate.Options()
            options.setPrecisionLossAllowed(false)
            options.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
            delegate = GpuDelegate(options)
            interpreter = Interpreter(mappedAsset("f6_backbone.tflite"), Interpreter.Options().addDelegate(delegate!!).setNumThreads(1).setUseXNNPACK(false))
            check(interpreter!!.getInputTensor(0).shape().contentEquals(intArrayOf(1, 800, 384, 3)))
            check(interpreter!!.getOutputTensor(0).shape().contentEquals(intArrayOf(1, 960)))
            val bytes = context.assets.open("golden_input.bin").use { it.readBytes() }
            val input = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            input.put(bytes).rewind()
            val expectedBuffer = ByteBuffer.wrap(context.assets.open("golden_embedding.bin").use { it.readBytes() }).order(ByteOrder.LITTLE_ENDIAN)
            val expected = FloatArray(384) { expectedBuffer.float }
            parityCosine = cosine(encode(input), expected)
            check(parityCosine.isFinite() && parityCosine >= 0.9999) { "GPU/checkpoint parity failed: cosine=$parityCosine" }
        }
        catch (error: Throwable)
        {
            interpreter?.close()
            delegate?.close()
            throw IllegalStateException("GPU inference initialization or parity failed. Capture is disabled; CPU fallback is not used.", error)
        }
    }

    fun encode(input: ByteBuffer): FloatArray
    {
        check(Thread.currentThread().id == ownerThread)
        input.rewind()
        output.rewind()
        val start = SystemClock.elapsedRealtimeNanos()
        interpreter!!.run(input, output)
        val gpuEnd = SystemClock.elapsedRealtimeNanos()
        output.rewind()
        output.asFloatBuffer().get(features)
        val value = head.encode(features)
        val end = SystemClock.elapsedRealtimeNanos()
        inferenceMs = (gpuEnd - start) / 1e6
        headMs = (end - gpuEnd) / 1e6
        return value
    }

    private fun mappedAsset(name: String): ByteBuffer
    {
        context.assets.openFd(name).use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).use { input ->
                return input.channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }
        }
    }

    private fun hashAsset(name: String): String
    {
        val digest = MessageDigest.getInstance("SHA-256")
        context.assets.open(name).use { stream ->
            val buffer = ByteArray(65536)
            while (true)
            {
                val count = stream.read(buffer)
                if (count < 0) { break }
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    override fun close()
    {
        check(Thread.currentThread().id == ownerThread)
        interpreter?.close()
        delegate?.close()
        interpreter = null
        delegate = null
    }
}
