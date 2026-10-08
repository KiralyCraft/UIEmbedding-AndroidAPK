package ro.ubb.uicollector

import android.media.Image
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Owns one GPU context on the capture thread. Raw frames remain in process memory. */
class OpenClPreprocessor : AutoCloseable {
    private val ownerThread = Thread.currentThread().id
    private var handle: Long
    val input: ByteBuffer = ByteBuffer.allocateDirect(800 * 384 * 3 * 4).order(ByteOrder.nativeOrder())
    init { System.loadLibrary("collector_preprocessing"); handle = create(); check(handle != 0L) }
    val driver: String get() = driverNative(handle)
    private external fun driverNative(handle: Long): String
    private external fun create(): Long
    private external fun ingestNative(handle: Long, input: ByteBuffer, rowStride: Int, left: Int, top: Int, width: Int, height: Int)
    private external fun prepareNative(handle: Long, output: ByteBuffer, width: Int, height: Int, resizedWidth: Int, resizedHeight: Int)
    private external fun destroy(handle: Long)
    fun ingest(image: Image) = ingest(CapturedFrame.from(image))
    fun ingest(image: CapturedFrame) {
        check(Thread.currentThread().id == ownerThread && handle != 0L)
        val crop = image.cropRect
        ingestNative(handle, image.buffer, image.rowStride, crop.left, crop.top, crop.width(), crop.height())
    }
    fun prepare(width: Int, height: Int): ByteBuffer {
        check(Thread.currentThread().id == ownerThread && handle != 0L)
        val scale = minOf(384.0 / width, 800.0 / height)
        prepareNative(handle, input, width, height, Math.rint(width * scale).toInt().coerceIn(1,384), Math.rint(height * scale).toInt().coerceIn(1,800))
        input.rewind()
        return input
    }
    override fun close() {
        check(Thread.currentThread().id == ownerThread)
        if(handle != 0L) { destroy(handle); handle = 0 }
    }
}
