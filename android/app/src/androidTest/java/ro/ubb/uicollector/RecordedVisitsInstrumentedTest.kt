package ro.ubb.uicollector

import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Read-only inspection after tools/continuous_capture_smoke.py has stopped controlled collection. */
class RecordedVisitsInstrumentedTest {
    @Test fun controlledVisitsRemainDistinctWithMeasuredSampleModes() {
        val app=InstrumentationRegistry.getInstrumentation().targetContext.applicationContext as RecorderApp
        assertFalse(app.captureActive)
        val summaries=JSONArray()
        val visits=mutableMapOf<String,Int>()
        var lockedVisitEnd=0L
        var lastAStart=0L
        var batterySamples=0
        app.store.readableDatabase.rawQuery("SELECT id,metadata,total_samples FROM runs WHERE owner=? AND closed=1",arrayOf(app.localOwner)).use {
            while(it.moveToNext()) {
                val id=it.getString(0)
                val metadata=JSONObject(String(app.vault.open(it.getBlob(1),"${app.localOwner}/run/$id"),Charsets.UTF_8))
                val pkg=metadata.getString("package_name")
                if(!pkg.startsWith("ro.ubb.uicollector.fixture.")) continue
                if(pkg.endsWith(".a")) lastAStart=maxOf(lastAStart,metadata.getLong("start_elapsed_ns"))
                if(metadata.optString("end_reason")=="screen_locked") lockedVisitEnd=maxOf(lockedVisitEnd,metadata.getLong("end_elapsed_ns"))
                val count=it.getInt(2)
                assertTrue("Empty controlled visit",count>0)
                visits[pkg]=(visits[pkg] ?: 0)+1
                var previous=-1L
                val modes=mutableSetOf<String>()
                val preprocessing=mutableListOf<Double>();val inference=mutableListOf<Double>()
                app.store.readableDatabase.rawQuery("SELECT sequence,payload FROM samples WHERE run_id=? ORDER BY sequence",arrayOf(id)).use { samples ->
                    var expected=0
                    while(samples.moveToNext()) {
                        assertEquals(expected,samples.getInt(0))
                        val sample=JSONObject(String(app.vault.open(samples.getBlob(1),"${app.localOwner}/sample/$id/$expected"),Charsets.UTF_8))
                        assertTrue(sample.getLong("elapsed_ns")>previous)
                        previous=sample.getLong("elapsed_ns")
                        modes.add(sample.getString("sampling_mode"))
                        assertEquals("accessibility",sample.getString("capture_source"))
                        if(sample.getString("sampling_mode")=="BATTERY_SAVER") {
                            assertTrue(sample.getDouble("target_fps")<=1.0)
                            batterySamples++
                        }
                        preprocessing.add(sample.getDouble("preprocess_ms"));inference.add(sample.getDouble("inference_ms"))
                        expected++
                    }
                    assertEquals(count,expected)
                }
                summaries.put(JSONObject().put("package",pkg).put("samples",count).put("modes",JSONArray(modes.toList())).put("end_reason",metadata.getString("end_reason"))
                    .put("preprocess_p50_ms",percentile(preprocessing,.5)).put("inference_p50_ms",percentile(inference,.5)))
            }
        }
        File(app.cacheDir,"controlled-visits-report.json").writeText(summaries.toString(2))
        assertTrue("A must have multiple separate visits",(visits["ro.ubb.uicollector.fixture.a"] ?: 0)>=2)
        assertTrue("B must have at least one visit",(visits["ro.ubb.uicollector.fixture.b"] ?: 0)>=1)
        assertTrue("Lock must close a visit",lockedVisitEnd>0)
        assertTrue("Capture must resume after unlock without consent",lastAStart>lockedVisitEnd)
        assertTrue("Battery Saver must commit samples at its reduced rate",batterySamples>0)
        File(app.cacheDir,"controlled-visits-report.json").writeText(summaries.toString(2))
    }
}
