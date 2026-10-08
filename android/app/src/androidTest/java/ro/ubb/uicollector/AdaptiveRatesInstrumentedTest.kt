package ro.ubb.uicollector

import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Read-only audit of controlled samples collected after the latest calibration. */
class AdaptiveRatesInstrumentedTest {
    @Test fun actualModeThroughputMatchesFreshCalibration() {
        val app=InstrumentationRegistry.getInstrumentation().targetContext.applicationContext as RecorderApp
        assertFalse(app.captureActive)
        assertFalse(app.needsCalibration())
        val report=JSONObject(app.preferences.getString("last_benchmark",null)!!)
        val calibrated=report.getDouble("selected_fps")
        val samplesByMode=RecordingMode.entries.associateWith { mutableListOf<Long>() }
        app.store.readableDatabase.rawQuery("SELECT id,metadata FROM runs WHERE owner=? AND closed=1",arrayOf(app.localOwner)).use { runs ->
            while(runs.moveToNext()) {
                val id=runs.getString(0)
                val metadata=JSONObject(String(app.vault.open(runs.getBlob(1),"${app.localOwner}/run/$id"),Charsets.UTF_8))
                if(metadata.getLong("start_wall_ms")<report.getLong("wall_ms") || metadata.getString("package_name")!="ro.ubb.uicollector.fixture.a") continue
                app.store.readableDatabase.rawQuery("SELECT sequence,payload FROM samples WHERE run_id=? ORDER BY sequence",arrayOf(id)).use { samples ->
                    while(samples.moveToNext()) {
                        val seq=samples.getInt(0)
                        val sample=JSONObject(String(app.vault.open(samples.getBlob(1),"${app.localOwner}/sample/$id/$seq"),Charsets.UTF_8))
                        val mode=RecordingMode.valueOf(sample.getString("sampling_mode"))
                        assertEquals("accessibility",sample.getString("capture_source"))
                        assertEquals(mode.ceiling(calibrated),sample.getDouble("target_fps"),0.001)
                        samplesByMode.getValue(mode).add(sample.getLong("elapsed_ns"))
                    }
                }
            }
        }
        val modes=JSONArray()
        samplesByMode.forEach { (mode,timestamps) ->
            assertTrue("${mode.title} must have at least 5 real samples",timestamps.size>=5)
            val intervals=timestamps.sorted().zipWithNext { a,b -> (b-a)/1e9 }
            val measured=1.0/percentile(intervals,0.5)
            val target=mode.ceiling(calibrated)
            assertEquals("${mode.title} actual acquisition rate",target,measured,target*0.15)
            modes.put(JSONObject().put("mode",mode.name).put("samples",timestamps.size).put("target_fps",target).put("measured_fps",measured))
        }
        File(app.cacheDir,"adaptive-rates-report.json").writeText(JSONObject().put("calibration",report).put("modes",modes).toString(2))
    }
}
