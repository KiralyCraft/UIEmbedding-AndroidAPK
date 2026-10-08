package ro.ubb.uicollector

import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.util.UUID

/** Isolated test database: never purges the application's production outbox. */
class OutboxInstrumentedTest
{
    private lateinit var store: LocalStore
    private lateinit var vault: Vault
    private val owner = "test-owner"
    private val databaseName="outbox-test-${UUID.randomUUID()}.db"

    @Before
    fun prepare()
    {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        vault = Vault(context)
        store = LocalStore(context, vault, databaseName)
        store.purgePending()
    }

    @After
    fun clean() { store.close(); InstrumentationRegistry.getInstrumentation().targetContext.deleteDatabase(databaseName) }

    @Test
    fun databaseOpensWithWalAndFullSynchronization()
    {
        store.writableDatabase.rawQuery("PRAGMA journal_mode", null).use {
            assertTrue(it.moveToFirst())
            assertEquals("wal", it.getString(0))
        }
        store.writableDatabase.rawQuery("PRAGMA synchronous", null).use {
            assertTrue(it.moveToFirst())
            assertEquals(2, it.getInt(0)) // SQLite FULL
        }
        store.writableDatabase.rawQuery("PRAGMA wal_autocheckpoint", null).use {
            assertTrue(it.moveToFirst())
            assertEquals(256, it.getInt(0))
        }
    }

    @Test
    fun unboundLocalTransferPreservesPayloadAndCannotRebindAccounts() {
        val local="local:test-install"
        val destination="a".repeat(64)
        val id=UUID.randomUUID().toString()
        store.createRun(local,JSONObject().put("id",id).put("revision",1).put("start_wall_ms",System.currentTimeMillis()).put("start_elapsed_ns",android.os.SystemClock.elapsedRealtimeNanos()))
        val sample=JSONObject().put("sequence",0).put("wall_ms",System.currentTimeMillis()).put("elapsed_ns",android.os.SystemClock.elapsedRealtimeNanos()).put("embedding_b64","original-vector")
        assertTrue(store.append(id,local,sample,1024*1024))
        assertTrue(runCatching { store.adoptLocal(local,destination) }.isFailure) // Active run must remain local.
        assertNotNull(store.nextBatch(local))
        store.closeRun(id,"test_stop")
        val before=store.nextBatch(local)!!.toString()
        store.adoptLocal(local,destination)
        assertNull(store.nextBatch(local))
        assertEquals(before,store.nextBatch(destination)!!.toString())
        assertEquals(1L,store.ownerStats(destination).second)
        assertTrue(runCatching { store.adoptLocal(destination,"b".repeat(64)) }.isFailure)
    }

    @Test
    fun corruptedLocalPayloadRollsBackAllOwnerChanges() {
        val local="local:rollback"
        val id=UUID.randomUUID().toString()
        store.createRun(local,JSONObject().put("id",id).put("revision",1).put("start_wall_ms",System.currentTimeMillis()).put("start_elapsed_ns",android.os.SystemClock.elapsedRealtimeNanos()))
        assertTrue(store.append(id,local,JSONObject().put("sequence",0).put("wall_ms",System.currentTimeMillis()).put("elapsed_ns",android.os.SystemClock.elapsedRealtimeNanos()).put("embedding_b64","vector"),1024*1024))
        store.closeRun(id,"test_stop")
        store.writableDatabase.execSQL("UPDATE samples SET payload=? WHERE run_id=?",arrayOf(byteArrayOf(1,2,3),id))
        assertTrue(runCatching { store.adoptLocal(local,"a".repeat(64)) }.isFailure)
        store.readableDatabase.rawQuery("SELECT owner FROM runs WHERE id=?",arrayOf(id)).use { assertTrue(it.moveToFirst());assertEquals(local,it.getString(0)) }
        assertEquals(1L,store.queueStats().second)
    }

    @Test
    fun transferTraversesMultipleCursorWindowsWithoutSkippingRunsOrReports() {
        val source="local:window-test"
        val destination="c".repeat(64)
        val db=store.writableDatabase
        db.beginTransaction()
        try {
            repeat(600) {
                val id=UUID.randomUUID().toString()
                store.createRun(source,JSONObject().put("id",id).put("revision",1).put("start_wall_ms",System.currentTimeMillis())
                    .put("start_elapsed_ns",android.os.SystemClock.elapsedRealtimeNanos()).put("synthetic_padding","x".repeat(5000)))
                store.closeRun(id,"fixture")
                store.saveReport(source,JSONObject().put("id",UUID.randomUUID().toString()).put("synthetic_padding","y".repeat(5000)))
            }
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        store.adoptLocal(source,destination)
        assertFalse(store.hasPending(source))
        db.rawQuery("SELECT COUNT(*) FROM runs WHERE owner=?",arrayOf(destination)).use { assertTrue(it.moveToFirst());assertEquals(600,it.getInt(0)) }
        db.rawQuery("SELECT COUNT(*) FROM reports WHERE owner=?",arrayOf(destination)).use { assertTrue(it.moveToFirst());assertEquals(600,it.getInt(0)) }
    }

    @Test
    fun transferClosedLocalDataWhileAccountRunContinues() {
        val source="local:older"
        val destination="d".repeat(64)
        fun makeRun(owner: String): String {
            val id=UUID.randomUUID().toString()
            store.createRun(owner,JSONObject().put("id",id).put("revision",1).put("start_wall_ms",1L).put("start_elapsed_ns",1L))
            assertTrue(store.append(id,owner,JSONObject().put("sequence",0).put("wall_ms",2L).put("elapsed_ns",2L).put("embedding_b64","vector"),1024*1024))
            return id
        }
        val old=makeRun(source)
        store.closeRun(old,"before_login")
        val oldPayload=store.nextBatch(source)!!.toString()
        val current=makeRun(destination)
        makeRun("another-account")
        val report=JSONObject().put("id",UUID.randomUUID().toString()).put("selected_fps",2.5)
        store.saveReport(source,report)
        assertEquals(PendingUploads(localSamples=1,localReports=1,accountSamples=1,otherSamples=1),store.pendingUploads(source,destination))
        store.adoptLocal(source,destination)
        assertEquals(PendingUploads(accountSamples=2,accountReports=1,otherSamples=1),store.pendingUploads(source,destination))
        assertEquals(report.toString(),store.nextReport(destination)!!.toString())
        assertTrue(store.append(current,destination,JSONObject().put("sequence",1).put("wall_ms",3L).put("elapsed_ns",3L).put("embedding_b64","new-vector"),1024*1024))
        val batch=store.nextBatch(destination)!!
        assertEquals(oldPayload,batch.toString())
        store.acknowledge(destination,batch,reply(batch))
        assertEquals(current,store.nextBatch(destination)!!.getJSONObject("run").getString("id"))
        assertEquals(2L,store.pendingUploads(source,destination).accountSamples)
        assertNull(store.nextBatch(source))
    }

    private fun run(): String
    {
        val id = UUID.randomUUID().toString()
        store.createRun(owner, JSONObject().put("id", id).put("revision", 1).put("start_wall_ms", System.currentTimeMillis()).put("start_elapsed_ns", android.os.SystemClock.elapsedRealtimeNanos()))
        return id
    }

    private fun append(id: String, sequence: Int)
    {
        assertTrue(store.append(id, owner, JSONObject().put("sequence", sequence).put("wall_ms", System.currentTimeMillis()).put("elapsed_ns", android.os.SystemClock.elapsedRealtimeNanos()).put("embedding_b64", "test-vector"), 1024 * 1024))
    }

    private fun reply(sent: JSONObject): JSONObject
    {
        val sequences = JSONArray()
        val samples = sent.getJSONArray("samples")
        for (i in 0 until samples.length()) { sequences.put(samples.getJSONObject(i).getInt("sequence")) }
        return JSONObject().put("run_id", sent.getJSONObject("run").getString("id")).put("committed_revision", sent.getJSONObject("run").getInt("revision")).put("committed_sequences", sequences)
    }

    @Test
    fun ackDoesNotDeleteSamplesOrClosureCreatedDuringUpload()
    {
        val id = run()
        append(id, 0)
        val inFlight = store.nextBatch(owner)!!
        append(id, 1)
        store.closeRun(id, "test_stop")
        store.acknowledge(owner, inFlight, reply(inFlight))
        val remaining = store.nextBatch(owner)!!
        assertEquals(2, remaining.getJSONObject("run").getInt("revision"))
        assertEquals(1, remaining.getJSONArray("samples").length())
        assertEquals(1, remaining.getJSONArray("samples").getJSONObject(0).getInt("sequence"))
        store.acknowledge(owner, remaining, reply(remaining))
        assertNull(store.nextBatch(owner))
        assertEquals(0L, store.queueStats().first)
    }

    @Test
    fun ownersAndAcknowledgementsAreChecked()
    {
        val id = run()
        append(id, 0)
        assertTrue(store.hasPendingOtherOwner("another-owner"))
        assertNull(store.nextBatch("another-owner"))
        val pending = store.nextBatch(owner)!!
        val badReply = reply(pending).put("committed_sequences", JSONArray())
        assertTrue(runCatching { store.acknowledge(owner, pending, badReply) }.isFailure)
        assertEquals(1L, store.queueStats().second)
    }

    @Test
    fun encryptionAuthenticatesOwnerContext()
    {
        val ciphertext = vault.seal("private embedding bytes".toByteArray(), "owner-a/run-1/0")
        assertEquals("private embedding bytes", String(vault.open(ciphertext, "owner-a/run-1/0")))
        assertTrue(runCatching { vault.open(ciphertext, "owner-b/run-1/0") }.isFailure)
    }

    @Test
    fun crashRecoveryClosesPersistedRuns()
    {
        val id = run()
        append(id, 0)
        store.recoverInterruptedRuns()
        val metadata = store.nextBatch(owner)!!.getJSONObject("run")
        assertEquals("process_interrupted", metadata.getString("end_reason"))
        assertEquals(1, metadata.getInt("expected_samples"))
    }
}
