package ro.ubb.uicollector

import android.app.Application
import android.content.Context
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import java.util.UUID
import java.util.concurrent.TimeUnit

class RecorderApp : Application()
{
    lateinit var vault: Vault
        private set
    lateinit var store: LocalStore
        private set
    lateinit var labels: LabelTracker
        private set
    lateinit var uploader: Uploader
        private set
    @Volatile var captureActive = false
    @Volatile var calibrating = false
    @Volatile var captureStatus = "Stopped"
    @Volatile var uploadStatus = "Not signed in"
    @Volatile var currentFps = 0.0
    val preferences by lazy { getSharedPreferences("settings", Context.MODE_PRIVATE) }
    val deviceId: String by lazy {
        preferences.getString("device_id", null) ?: UUID.randomUUID().toString().also { preferences.edit().putString("device_id", it).commit() }
    }

    override fun onCreate()
    {
        super.onCreate()
        vault = Vault(this)
        store = LocalStore(this, vault)
        store.recoverInterruptedRuns()
        labels = LabelTracker(this)
        uploader = Uploader(this)
        val constraints = Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()
        val periodic = PeriodicWorkRequestBuilder<UploadWorker>(15, TimeUnit.MINUTES).setConstraints(constraints).build()
        WorkManager.getInstance(this).enqueueUniquePeriodicWork("embedding-periodic-upload", ExistingPeriodicWorkPolicy.KEEP, periodic)
    }

    fun requestUpload()
    {
        val constraints = Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()
        val work = OneTimeWorkRequestBuilder<UploadWorker>().setConstraints(constraints).build()
        WorkManager.getInstance(this).enqueueUniqueWork("embedding-upload-now", ExistingWorkPolicy.KEEP, work)
    }
}
