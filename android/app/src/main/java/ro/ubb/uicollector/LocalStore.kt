package ro.ubb.uicollector

import android.content.Context
import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.json.JSONObject
import org.json.JSONArray

class LocalStore(context: Context, private val vault: Vault, databaseName: String = "outbox.db") : SQLiteOpenHelper(context, databaseName, null, 1)
{
    private val directory = context.filesDir

    init
    {
        setWriteAheadLoggingEnabled(true)
    }

    override fun onConfigure(db: SQLiteDatabase)
    {
        db.setForeignKeyConstraintsEnabled(true)
        db.execPerConnectionSQL("PRAGMA synchronous=FULL", null)
        // This PRAGMA returns a row; Android's execSQL rejects row-returning statements.
        db.execPerConnectionSQL("PRAGMA wal_autocheckpoint=256", null)
        db.rawQuery("PRAGMA wal_autocheckpoint", null).use {
            check(it.moveToFirst() && it.getInt(0) == 256) { "Could not configure WAL checkpoint interval" }
        }
    }

    override fun onCreate(db: SQLiteDatabase)
    {
        db.execSQL("CREATE TABLE runs(id TEXT PRIMARY KEY, owner TEXT NOT NULL, metadata BLOB NOT NULL, revision INTEGER NOT NULL, acked_revision INTEGER NOT NULL DEFAULT 0, total_samples INTEGER NOT NULL DEFAULT 0, last_wall_ms INTEGER NOT NULL, last_elapsed_ns INTEGER NOT NULL, closed INTEGER NOT NULL DEFAULT 0)")
        db.execSQL("CREATE TABLE samples(run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, sequence INTEGER NOT NULL, payload BLOB NOT NULL, size_bytes INTEGER NOT NULL, PRIMARY KEY(run_id, sequence))")
        db.execSQL("CREATE TABLE reports(id TEXT PRIMARY KEY, owner TEXT NOT NULL, payload BLOB NOT NULL)")
        db.execSQL("CREATE TABLE counters(id INTEGER PRIMARY KEY CHECK(id=1), bytes INTEGER NOT NULL, samples INTEGER NOT NULL)")
        db.execSQL("INSERT INTO counters VALUES(1,0,0)")
        db.execSQL("CREATE TABLE benchmark_probe(id INTEGER PRIMARY KEY, payload BLOB NOT NULL)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int)
    {
        error("Explicit outbox migration required; refusing to discard pending data")
    }

    private fun encrypted(value: JSONObject, aad: String): ByteArray = vault.seal(value.toString().toByteArray(Charsets.UTF_8), aad)
    private fun decoded(value: ByteArray, aad: String): JSONObject = JSONObject(String(vault.open(value, aad), Charsets.UTF_8))

    @Synchronized
    fun createRun(owner: String, metadata: JSONObject)
    {
        val id = metadata.getString("id")
        writableDatabase.insertOrThrow("runs", null, ContentValues().apply {
            put("id", id)
            put("owner", owner)
            put("metadata", encrypted(metadata, "$owner/run/$id"))
            put("revision", metadata.getInt("revision"))
            put("last_wall_ms", metadata.getLong("start_wall_ms"))
            put("last_elapsed_ns", metadata.getLong("start_elapsed_ns"))
        })
    }

    @Synchronized
    fun append(runId: String, owner: String, payload: JSONObject, quota: Long): Boolean
    {
        val db = writableDatabase
        db.beginTransaction()
        try
        {
            val sequence = payload.getInt("sequence")
            val bytes = encrypted(payload, "$owner/sample/$runId/$sequence")
            if (queueStats().first + bytes.size > quota || directory.usableSpace < 64L * 1024 * 1024)
            {
                return false
            }
            db.rawQuery("SELECT owner, total_samples, closed FROM runs WHERE id=?", arrayOf(runId)).use {
                check(it.moveToFirst() && it.getString(0) == owner && it.getInt(1) == sequence && it.getInt(2) == 0)
            }
            db.insertOrThrow("samples", null, ContentValues().apply {
                put("run_id", runId)
                put("sequence", sequence)
                put("payload", bytes)
                put("size_bytes", bytes.size)
            })
            db.execSQL("UPDATE runs SET total_samples=total_samples+1,last_wall_ms=?,last_elapsed_ns=? WHERE id=?", arrayOf(payload.getLong("wall_ms"), payload.getLong("elapsed_ns"), runId))
            db.execSQL("UPDATE counters SET bytes=bytes+?,samples=samples+1 WHERE id=1", arrayOf(bytes.size))
            db.setTransactionSuccessful()
            return true
        }
        finally
        {
            db.endTransaction()
        }
    }

    @Synchronized
    fun closeRun(runId: String, reason: String, atLastSample: Boolean = false)
    {
        val db = writableDatabase
        db.beginTransaction()
        try
        {
            db.rawQuery("SELECT owner,metadata,revision,total_samples,last_wall_ms,last_elapsed_ns,closed FROM runs WHERE id=?", arrayOf(runId)).use {
                if (it.moveToFirst() == false || it.getInt(6) == 1)
                {
                    db.setTransactionSuccessful()
                    return
                }
                val owner = it.getString(0)
                val value = decoded(it.getBlob(1), "$owner/run/$runId")
                value.put("revision", it.getInt(2) + 1)
                value.put("expected_samples", it.getInt(3))
                value.put("end_wall_ms", if (atLastSample) it.getLong(4) else System.currentTimeMillis())
                value.put("end_elapsed_ns", if (atLastSample) it.getLong(5) else android.os.SystemClock.elapsedRealtimeNanos().coerceAtLeast(it.getLong(5)))
                value.put("end_reason", reason)
                db.update("runs", ContentValues().apply {
                    put("closed", 1)
                    put("revision", value.getInt("revision"))
                    put("metadata", encrypted(value, "$owner/run/$runId"))
                }, "id=?", arrayOf(runId))
            }
            db.setTransactionSuccessful()
        }
        finally
        {
            db.endTransaction()
        }
    }

    @Synchronized
    fun recoverInterruptedRuns()
    {
        val ids = mutableListOf<String>()
        readableDatabase.rawQuery("SELECT id FROM runs WHERE closed=0", null).use {
            while (it.moveToNext()) { ids.add(it.getString(0)) }
        }
        ids.forEach { closeRun(it, "process_interrupted", true) }
        writableDatabase.delete("benchmark_probe", null, null)
    }

    @Synchronized
    fun nextBatch(owner: String): JSONObject?
    {
        val db = readableDatabase
        db.rawQuery("SELECT id,metadata FROM runs WHERE owner=? AND (acked_revision < revision OR EXISTS(SELECT 1 FROM samples WHERE run_id=runs.id)) ORDER BY last_wall_ms,id LIMIT 1", arrayOf(owner)).use {
            if (it.moveToFirst() == false) { return null }
            val id = it.getString(0)
            val metadata = decoded(it.getBlob(1), "$owner/run/$id")
            val samples = JSONArray()
            db.rawQuery("SELECT sequence,payload FROM samples WHERE run_id=? ORDER BY sequence LIMIT 128", arrayOf(id)).use { rows ->
                while (rows.moveToNext())
                {
                    samples.put(decoded(rows.getBlob(1), "$owner/sample/$id/${rows.getInt(0)}"))
                }
            }
            return JSONObject().put("run", metadata).put("samples", samples)
        }
    }

    @Synchronized
    fun acknowledge(owner: String, sent: JSONObject, reply: JSONObject)
    {
        val run = sent.getJSONObject("run")
        val id = run.getString("id")
        require(reply.getString("run_id") == id)
        val submitted = sent.getJSONArray("samples")
        val expected = (0 until submitted.length()).map { submitted.getJSONObject(it).getInt("sequence") }.toSet()
        val acknowledged = reply.getJSONArray("committed_sequences")
        val actual = (0 until acknowledged.length()).map { acknowledged.getInt(it) }.toSet()
        require(actual == expected && acknowledged.length() == expected.size)
        require(reply.getInt("committed_revision") >= run.getInt("revision"))
        val db = writableDatabase
        db.beginTransaction()
        try
        {
            db.rawQuery("SELECT owner FROM runs WHERE id=?", arrayOf(id)).use {
                check(it.moveToFirst() && it.getString(0) == owner)
            }
            for (sequence in expected)
            {
                db.rawQuery("SELECT size_bytes FROM samples WHERE run_id=? AND sequence=?", arrayOf(id, sequence.toString())).use {
                    if (it.moveToFirst())
                    {
                        db.execSQL("UPDATE counters SET bytes=bytes-?,samples=samples-1 WHERE id=1", arrayOf(it.getLong(0)))
                        db.delete("samples", "run_id=? AND sequence=?", arrayOf(id, sequence.toString()))
                    }
                }
            }
            // A concurrent close may have raised revision since this request was sent.
            db.execSQL("UPDATE runs SET acked_revision=MAX(acked_revision,?) WHERE id=?", arrayOf(run.getInt("revision"), id))
            db.execSQL("DELETE FROM runs WHERE id=? AND closed=1 AND acked_revision>=revision AND NOT EXISTS(SELECT 1 FROM samples WHERE run_id=runs.id)", arrayOf(id))
            db.setTransactionSuccessful()
        }
        finally
        {
            db.endTransaction()
        }
    }

    @Synchronized
    fun queueStats(): Pair<Long, Long>
    {
        readableDatabase.rawQuery("SELECT bytes,samples FROM counters WHERE id=1", null).use {
            check(it.moveToFirst())
            return Pair(it.getLong(0), it.getLong(1))
        }
    }

    @Synchronized
    fun ownerStats(owner: String): Pair<Long, Long> {
        readableDatabase.rawQuery("SELECT COALESCE(SUM(size_bytes),0),COUNT(*) FROM samples JOIN runs ON samples.run_id=runs.id WHERE owner=?", arrayOf(owner)).use {
            check(it.moveToFirst())
            return Pair(it.getLong(0), it.getLong(1))
        }
    }

    /** Separate records awaiting an account from the current account's automatic upload queue. */
    @Synchronized
    fun pendingUploads(localOwner: String, accountOwner: String?): PendingUploads {
        var result = PendingUploads()
        readableDatabase.rawQuery("""
            SELECT owner,SUM(samples),SUM(reports) FROM (
                SELECT r.owner AS owner,COUNT(s.sequence) AS samples,0 AS reports
                FROM runs r LEFT JOIN samples s ON s.run_id=r.id GROUP BY r.owner
                UNION ALL SELECT owner,0,COUNT(*) FROM reports GROUP BY owner
            ) GROUP BY owner
        """.trimIndent(), null).use { rows ->
            while(rows.moveToNext()) {
                val samples=rows.getLong(1)
                val reports=rows.getLong(2)
                result=when(rows.getString(0)) {
                    localOwner -> result.copy(localSamples=samples,localReports=reports)
                    accountOwner -> result.copy(accountSamples=samples,accountReports=reports)
                    else -> result.copy(otherSamples=result.otherSamples+samples,otherReports=result.otherReports+reports)
                }
            }
        }
        return result
    }

    /** Rebind only the installation's unbound local data. AAD and ciphertext change atomically. */
    @Synchronized
    fun adoptLocal(source: String, destination: String) {
        require(source.startsWith("local:") && !destination.startsWith("local:") && destination.matches(Regex("[a-f0-9]{64}")))
        check(directory.usableSpace >= queueStats().first * 2 + 64L * 1024 * 1024) { "Insufficient free space for an atomic transfer" }
        val db = writableDatabase
        db.beginTransaction()
        try {
            db.rawQuery("SELECT 1 FROM runs WHERE owner=? AND closed=0",arrayOf(source)).use { check(!it.moveToFirst()) { "Close local recording sessions before transferring" } }
            val runIds=mutableListOf<String>()
            db.rawQuery("SELECT id FROM runs WHERE owner=?",arrayOf(source)).use { while(it.moveToNext()) runIds.add(it.getString(0)) }
            // Snapshot IDs first: changing an owner must not shift later CursorWindow offsets.
            for(id in runIds) {
                db.rawQuery("SELECT metadata FROM runs WHERE id=? AND owner=?",arrayOf(id,source)).use { runs ->
                    check(runs.moveToFirst())
                    val metadata=vault.open(runs.getBlob(0),"$source/run/$id")
                    db.rawQuery("SELECT sequence,payload FROM samples WHERE run_id=?",arrayOf(id)).use { samples ->
                        while(samples.moveToNext()) {
                            val sequence=samples.getInt(0)
                            val value=vault.open(samples.getBlob(1),"$source/sample/$id/$sequence")
                            val sealed=vault.seal(value,"$destination/sample/$id/$sequence")
                            db.update("samples",ContentValues().apply { put("payload",sealed);put("size_bytes",sealed.size) },"run_id=? AND sequence=?",arrayOf(id,sequence.toString()))
                        }
                    }
                    db.update("runs",ContentValues().apply { put("owner",destination);put("metadata",vault.seal(metadata,"$destination/run/$id")) },"id=?",arrayOf(id))
                }
            }
            val reportIds=mutableListOf<String>()
            db.rawQuery("SELECT id FROM reports WHERE owner=?",arrayOf(source)).use { while(it.moveToNext()) reportIds.add(it.getString(0)) }
            for(id in reportIds) {
                db.rawQuery("SELECT payload FROM reports WHERE id=? AND owner=?",arrayOf(id,source)).use {
                    check(it.moveToFirst())
                    val plaintext=vault.open(it.getBlob(0),"$source/report/$id")
                    db.update("reports",ContentValues().apply { put("owner",destination);put("payload",vault.seal(plaintext,"$destination/report/$id")) },"id=?",arrayOf(id))
                }
            }
            db.execSQL("UPDATE counters SET bytes=(SELECT COALESCE(SUM(size_bytes),0) FROM samples),samples=(SELECT COUNT(*) FROM samples) WHERE id=1")
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
    }

    @Synchronized
    fun hasPendingOtherOwner(owner: String, unboundLocalOwner: String = ""): Boolean
    {
        readableDatabase.rawQuery("SELECT 1 FROM runs WHERE owner<>? AND owner<>? UNION ALL SELECT 1 FROM reports WHERE owner<>? AND owner<>? LIMIT 1", arrayOf(owner, unboundLocalOwner, owner, unboundLocalOwner)).use { return it.moveToFirst() }
    }

    @Synchronized
    fun hasPending(owner: String): Boolean
    {
        return nextBatch(owner) != null || nextReport(owner) != null
    }

    @Synchronized
    fun saveReport(owner: String, value: JSONObject)
    {
        val id = value.getString("id")
        writableDatabase.insertOrThrow("reports", null, ContentValues().apply {
            put("id", id)
            put("owner", owner)
            put("payload", encrypted(value, "$owner/report/$id"))
        })
    }

    @Synchronized
    fun nextReport(owner: String): JSONObject?
    {
        readableDatabase.rawQuery("SELECT id,payload FROM reports WHERE owner=? ORDER BY rowid LIMIT 1", arrayOf(owner)).use {
            if (it.moveToFirst() == false) { return null }
            return decoded(it.getBlob(1), "$owner/report/${it.getString(0)}")
        }
    }

    @Synchronized
    fun acknowledgeReport(owner: String, id: String)
    {
        writableDatabase.delete("reports", "id=? AND owner=?", arrayOf(id, owner))
    }

    @Synchronized
    fun benchmarkWrite(value: JSONObject)
    {
        writableDatabase.insertWithOnConflict("benchmark_probe", null, ContentValues().apply {
            put("id", 1)
            put("payload", encrypted(value, "benchmark-probe"))
        }, SQLiteDatabase.CONFLICT_REPLACE)
    }

    @Synchronized
    fun clearBenchmarkProbe() { writableDatabase.delete("benchmark_probe", null, null) }

    @Synchronized
    fun purgePending()
    {
        val db = writableDatabase
        db.beginTransaction()
        try
        {
            db.delete("samples", null, null)
            db.delete("runs", null, null)
            db.delete("reports", null, null)
            db.execSQL("UPDATE counters SET bytes=0,samples=0 WHERE id=1")
            db.setTransactionSuccessful()
        }
        finally { db.endTransaction() }
    }
}
