package ro.ubb.uicollector

import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.util.UUID

/** Run only on a test install: setUp/tearDown intentionally empty its local outbox. */
class OutboxInstrumentedTest
{
    private lateinit var store: LocalStore
    private lateinit var vault: Vault
    private val owner = "test-owner"

    @Before
    fun prepare()
    {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        vault = Vault(context)
        store = LocalStore(context, vault)
        store.purgePending()
    }

    @After
    fun clean() { store.purgePending(); store.close() }

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
