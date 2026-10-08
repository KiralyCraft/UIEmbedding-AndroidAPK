package ro.ubb.uicollector

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Runs the exact Android production head and pacing code on a plain JVM. */
fun main(args: Array<String>)
{
    require(args.size == 3)
    val head = EmbeddingHead(File(args[0]).readBytes())
    val bytes = File(args[1]).readBytes()
    require(bytes.size % (960 * 4) == 0)
    val input = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val output = ByteBuffer.allocate(bytes.size / (960 * 4) * 384 * 4).order(ByteOrder.LITTLE_ENDIAN)
    repeat(bytes.size / (960 * 4))
    {
        val features = FloatArray(960) { input.float }
        head.encode(features).forEach { output.putFloat(it) }
    }
    File(args[2]).writeBytes(output.array())
    check(percentile(listOf(1.0, 2.0, 3.0, 4.0), .95) == 4.0)
    check(kotlin.math.abs(cosine(floatArrayOf(1f, 0f), floatArrayOf(1f, 0f)) - 1.0) < 1e-12)
    val slow = RateController(30.0)
    repeat(40) { slow.observe(200.0, 0) }
    check(slow.fps == 2.0)
    repeat(40) { slow.observe(10.0, 3) }
    check(slow.fps == 1.0)
    repeat(40 * 8) { slow.observe(1.0, 0) }
    check(slow.fps == 2.0)
    val bounded = RateController(5.0)
    repeat(40 * 20) { bounded.observe(1.0, 0) }
    check(bounded.fps <= 5.0)
    check(runCatching { head.encode(FloatArray(959)) }.isFailure)
    check(runCatching { head.encode(FloatArray(960) { Float.NaN }) }.isFailure)
    println("Production Kotlin head, nonfinite checks, pacing and ceiling tests passed")
}
