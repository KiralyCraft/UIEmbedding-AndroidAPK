package ro.ubb.uicollector

import android.content.Context
import android.os.SystemClock
import androidx.work.Worker
import androidx.work.WorkerParameters
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.HttpUrl.Companion.toHttpUrl
import org.json.JSONObject
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min
import kotlin.random.Random

fun ownerKey(identity: JSONObject): String
{
    val bytes = (identity.getString("server") + "\n" + identity.getString("user_id")).toByteArray(Charsets.UTF_8)
    return MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}

class Uploader(private val app: RecorderApp)
{
    private val client = OkHttpClient.Builder().connectTimeout(10, TimeUnit.SECONDS).readTimeout(25, TimeUnit.SECONDS).writeTimeout(25, TimeUnit.SECONDS).callTimeout(30, TimeUnit.SECONDS).followRedirects(false).followSslRedirects(false).build()
    private val running = AtomicBoolean(false)

    fun login(server: String, username: String, password: String)
    {
        check(app.captureActive == false) { "Stop capture before changing authentication" }
        val normalized = (server.trim().trimEnd('/') + "/").toHttpUrl()
        require(normalized.isHttps && normalized.username.isEmpty() && normalized.password.isEmpty() && normalized.query == null && normalized.fragment == null) { "Use an HTTPS server URL without credentials or query parameters" }
        val body = JSONObject().put("username", username).put("password", password)
        val reply = request(normalized.toString(), "v1/auth/login", body, null)
        reply.put("server", normalized.toString())
        val owner = ownerKey(reply)
        check(app.store.hasPendingOtherOwner(owner, app.localOwner) == false) { "Queued data belongs to another account/server. Sign in to that account to upload it, or explicitly delete it first." }
        app.vault.saveIdentity(reply)
        retryNow()
        app.uploadStatus = "Signed in as ${reply.getString("username")}" 
        app.requestUpload()
    }

    fun retryNow()
    {
        app.preferences.edit().remove("upload_retry_at").remove("upload_failures").remove("upload_blocked").apply()
    }

    fun flush(maximumMs: Long = 20000): Boolean
    {
        if (running.compareAndSet(false, true) == false) { return false }
        try
        {
            val identity = app.vault.identity() ?: return true
            val owner = ownerKey(identity)
            if (app.preferences.getBoolean("upload_blocked", false))
            {
                app.uploadStatus = "Upload blocked by a server rejection. Data is retained; resolve the error and press Retry upload."
                return false
            }
            if (System.currentTimeMillis() < app.preferences.getLong("upload_retry_at", 0L)) { return false }
            val deadline = SystemClock.elapsedRealtime() + maximumMs
            do
            {
                val report = app.store.nextReport(owner)
                if (report != null)
                {
                    val reply = request(identity.getString("server"), "v1/benchmarks", report, identity.getString("token"))
                    check(reply.getBoolean("committed") && reply.getString("id") == report.getString("id")) { "Invalid benchmark acknowledgment" }
                    app.store.acknowledgeReport(owner, report.getString("id"))
                }
                else
                {
                    val batch = app.store.nextBatch(owner) ?: break
                    val reply = request(identity.getString("server"), "v1/ingest", batch, identity.getString("token"))
                    app.store.acknowledge(owner, batch, reply)
                }
                app.preferences.edit().putInt("upload_failures", 0).apply()
                app.uploadStatus = "Server acknowledged uploaded data"
            } while (SystemClock.elapsedRealtime() < deadline)
            val complete = !app.store.hasPending(owner)
            if(complete) app.uploadStatus = "Account upload queue is clear"
            return complete
        }
        catch (error: Exception)
        {
            val failures = (app.preferences.getInt("upload_failures", 0) + 1).coerceAtMost(12)
            val delay = min(900000L, (1L shl failures) * 1000L) + Random.nextLong(1000L)
            val blocked = error is HttpFailure && error.status in setOf(400, 403, 404, 409, 413, 422)
            app.preferences.edit().putInt("upload_failures", failures).putLong("upload_retry_at", System.currentTimeMillis() + delay).putBoolean("upload_blocked", blocked).apply()
            app.uploadStatus = if (error is HttpFailure && error.status == 401) "Sign in again to resume uploads. Capture can continue into the local queue." else "Upload retained locally: ${error.message?.take(200)}"
            return false
        }
        finally
        {
            running.set(false)
        }
    }

    private fun request(server: String, path: String, body: JSONObject, token: String?): JSONObject
    {
        val url = server.toHttpUrl().resolve(path) ?: error("Invalid server path")
        check(url.isHttps)
        val request = Request.Builder().url(url).post(body.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
        if (token != null) { request.header("Authorization", "Bearer $token") }
        client.newCall(request.build()).execute().use { response ->
            val text = response.body?.string().orEmpty()
            if (response.isSuccessful == false)
            {
                val detail = runCatching { JSONObject(text).optString("detail") }.getOrDefault("HTTP ${response.code}").take(200)
                throw HttpFailure(response.code, detail)
            }
            return JSONObject(text)
        }
    }
}

class HttpFailure(val status: Int, message: String) : Exception("HTTP $status: $message")

class UploadWorker(context: Context, parameters: WorkerParameters) : Worker(context, parameters)
{
    override fun doWork(): Result
    {
        val app = applicationContext as RecorderApp
        return if (app.uploader.flush()) Result.success() else Result.retry()
    }
}
