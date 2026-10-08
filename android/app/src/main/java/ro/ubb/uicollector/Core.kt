package ro.ubb.uicollector

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Android-independent implementation of the ORIGINAL checkpoint head. */
class EmbeddingHead(bytes: ByteArray)
{
    private val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val inputSize: Int
    val outputSize: Int
    private val epsilon: Float
    private val firstWeights: FloatArray
    private val firstBias: FloatArray
    private val normWeights: FloatArray
    private val normBias: FloatArray
    private val lastWeights: FloatArray

    init
    {
        require(bytes.size >= 16)
        val magic = ByteArray(4)
        buffer.get(magic)
        require(String(magic, Charsets.US_ASCII) == "UIH1") { "Invalid embedding-head format" }
        inputSize = buffer.int
        outputSize = buffer.int
        epsilon = buffer.float
        require(inputSize == 960 && outputSize == 384 && epsilon > 0.0f)
        firstWeights = readFloats(outputSize * inputSize)
        firstBias = readFloats(outputSize)
        normWeights = readFloats(outputSize)
        normBias = readFloats(outputSize)
        lastWeights = readFloats(outputSize * outputSize)
        require(buffer.remaining() == 0) { "Unexpected head bytes" }
    }

    private fun readFloats(count: Int): FloatArray
    {
        return FloatArray(count) { buffer.float }
    }

    fun encode(features: FloatArray): FloatArray
    {
        require(features.size == inputSize && features.all { it.isFinite() })
        val hidden = DoubleArray(outputSize)
        for (i in 0 until outputSize)
        {
            var sum = firstBias[i].toDouble()
            for (j in 0 until inputSize)
            {
                sum += firstWeights[i * inputSize + j].toDouble() * features[j]
            }
            hidden[i] = sum
        }
        val mean = hidden.sum() / outputSize
        val variance = hidden.sumOf { (it - mean) * (it - mean) } / outputSize
        val inverse = 1.0 / sqrt(variance + epsilon)
        for (i in hidden.indices)
        {
            val value = (hidden[i] - mean) * inverse * normWeights[i] + normBias[i]
            // erf-based GELU, not the less accurate tanh GELU substitution.
            hidden[i] = 0.5 * value * (1.0 + erf(value / sqrt(2.0)))
        }
        val result = DoubleArray(outputSize)
        for (i in 0 until outputSize)
        {
            var sum = 0.0
            for (j in 0 until outputSize)
            {
                sum += lastWeights[i * outputSize + j].toDouble() * hidden[j]
            }
            result[i] = sum
        }
        val norm = sqrt(result.sumOf { it * it })
        require(norm.isFinite() && norm > 1e-12) { "Degenerate embedding" }
        return FloatArray(outputSize) { (result[it] / norm).toFloat() }
    }

    private fun erf(value: Double): Double
    {
        // A&S 7.1.26; absolute error approximately <= 1.5e-7.
        val magnitude = abs(value)
        val t = 1.0 / (1.0 + 0.3275911 * magnitude)
        val polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
        val positive = 1.0 - polynomial * exp(-magnitude * magnitude)
        return if (value >= 0.0) positive else -positive
    }
}

class RateController(private val ceiling: Double)
{
    val candidates = (listOf(0.25, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 10.0, 15.0, 20.0, 30.0) + ceiling).distinct().sorted()
    private val floor = min(0.25, ceiling)
    var fps = ceiling.coerceAtMost(30.0)
        private set
    private val durations = mutableListOf<Double>()
    private var coolWindows = 0

    fun observe(milliseconds: Double, thermalStatus: Int)
    {
        require(milliseconds.isFinite() && milliseconds >= 0.0)
        durations.add(milliseconds)
        if (durations.size < 40)
        {
            return
        }
        val p95 = percentile(durations.toList(), 0.95)
        durations.clear()
        val safe = (750.0 / max(p95, 1.0)).coerceAtMost(ceiling)
        if (safe < fps || thermalStatus >= 3)
        {
            fps = if (thermalStatus >= 3) max(floor, fps / 2.0) else candidates.lastOrNull { it <= safe } ?: floor
            coolWindows = 0
        }
        else if (safe >= fps * 1.5 && thermalStatus <= 1)
        {
            coolWindows += 1
            if (coolWindows >= 8)
            {
                fps = candidates.firstOrNull { it > fps && it <= safe && it <= ceiling } ?: fps
                coolWindows = 0
            }
        }
        else
        {
            coolWindows = 0
        }
    }
}

fun percentile(values: List<Double>, fraction: Double): Double
{
    require(values.isNotEmpty())
    val sorted = values.sorted()
    val index = kotlin.math.ceil(fraction * sorted.size).toInt().coerceIn(1, sorted.size) - 1
    return sorted[index]
}

fun cosine(a: FloatArray, b: FloatArray): Double
{
    require(a.size == b.size)
    var dot = 0.0
    var aa = 0.0
    var bb = 0.0
    for (i in a.indices)
    {
        dot += a[i].toDouble() * b[i]
        aa += a[i].toDouble() * a[i]
        bb += b[i].toDouble() * b[i]
    }
    return dot / sqrt(aa * bb).coerceAtLeast(1e-30)
}
