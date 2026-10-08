package ro.ubb.uicollector

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PixelFormat
import android.media.ImageReader
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.android.OpenCVLoader
import kotlin.math.abs

class OpenClPreprocessingInstrumentedTest {
    @Test fun nativeOpenClPreservesFullScreenPreprocessingAcrossAspectRatios() {
        assertTrue(OpenCVLoader.initLocal())
        val context=InstrumentationRegistry.getInstrumentation().targetContext
        val results=org.json.JSONArray()
        EmbeddingEngine(context).use { engine ->
            OpenClPreprocessor().use { gpu ->
                for((height,width) in listOf(2560 to 1096,1920 to 1080,1080 to 1920,800 to 384,2520 to 1080,701 to 333,120 to 50,1000 to 1000)) {
                    val pixels=IntArray(width*height) { i -> val x=i%width;val y=i/width;Color.rgb((x/7+y/3)%256,(x/13*29+y/37*43)%256,(x*3+y*7)%256) }
                    val bitmap=Bitmap.createBitmap(pixels,width,height,Bitmap.Config.ARGB_8888)
                    try {
                        ImageReader.newInstance(width,height,PixelFormat.RGBA_8888,2).use { reader ->
                            val canvas=reader.surface.lockCanvas(null)
                            try { canvas.drawBitmap(bitmap,0f,0f,null) } finally { reader.surface.unlockCanvasAndPost(canvas) }
                            var image=reader.acquireNextImage();val deadline=SystemClock.elapsedRealtime()+5000
                            while(image==null && SystemClock.elapsedRealtime()<deadline) { SystemClock.sleep(10);image=reader.acquireNextImage() }
                            checkNotNull(image).use { frame ->
                                FramePreprocessor().use { cpu ->
                                    cpu.ingest(frame);gpu.ingest(frame)
                                    val baseline=cpu.prepare();val input=gpu.prepare(width,height)
                                    val a=baseline.asFloatBuffer();val b=input.asFloatBuffer()
                                    var maximum=0.0;var index=0;val deviation=doubleArrayOf(.229,.224,.225)
                                    while(a.hasRemaining()) { maximum=maxOf(maximum,abs(a.get().toDouble()-b.get())*255*deviation[index%3]);index++ }
                                    val similarity=cosine(engine.encode(baseline),engine.encode(input))
                                    assertTrue("${width}x$height RGB error=$maximum",maximum<=1.0001)
                                    assertTrue("${width}x$height embedding cosine=$similarity",similarity>=.9999)
                                    results.put(org.json.JSONObject().put("width",width).put("height",height).put("rgb_error",maximum).put("embedding_cosine",similarity))
                                    if(width==1096 && height==2560) {
                                        val app=context.applicationContext as RecorderApp
                                        app.preferences.edit().remove("preprocessing_signature").commit()
                                        CapturePreprocessor(app,engine).use { production ->
                                            production.ingest(frame)
                                            production.prepare()
                                            results.put(org.json.JSONObject(app.preferences.getString("preprocessing_comparison","{}")!!)
                                                .put("scope","Synthetic Xperia-size RGBA frame, repeated complete pipeline including fresh uploads"))
                                        }
                                    }
                                }
                            }
                        }
                    } finally { bitmap.recycle() }
                }
            }
        }
        java.io.File(context.cacheDir,"opencl-preprocessing-report.json").writeText(results.toString(2))
    }
}
