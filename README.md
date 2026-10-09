# UI Embedding Collector for Android

An Android research collector and optional authenticated Python/MySQL ingestion service for the trained F6 encoder. Full-screen views become 384-dimensional embeddings locally. Captured pixels stay in memory; the app never saves or uploads screenshots or video. Embeddings and application metadata are encrypted in an Android Keystore-backed local queue.

## Phone workflow

The Recording, Applications and Settings tabs separate everyday controls from configuration. First-run onboarding gates recording on Notifications, Accessibility and Usage Access. Android's permission state is shown explicitly; granting Accessibility is distinct from its temporary service connection state. Sideloaded applications may require **App info → Allow restricted settings** before enabling Accessibility.

All eligible installed applications and future installations are included by default. Applications provides searchable exclusions, including an optional system-app filter. The collector and system UI are always excluded. An existing nonempty legacy selected-app list remains in effect until the user explicitly switches to all applications.

While viewing the collector, an active session says **Ready · open another app to record**; the collector's own screen is intentionally excluded. After **App info → Force Stop**, Android may disable Accessibility or leave it disconnected. The Recording and Permissions menus explain the actual state and link directly to Accessibility settings. Re-enable UI Embedding Collector if needed, return to the app and start recording. Window labeling refreshes immediately when the service reconnects.

Recording works without an account or server. First use calibrates the complete capture, preprocessing, inference and encrypted-storage pipeline. Model/backend/driver/capture-method changes invalidate calibration. Maximum Detail uses the measured sustainable rate; Balanced uses half that rate, capped at 5/s; Battery Saver uses one quarter, capped at 1/s, and batches uploads. These modes remain distinct even on slower capture methods. The menu shows each effective rate. The dashboard shows actual committed throughput, stage timings, queue size, thermal state and session duration. The persistent notification has a Stop action.

### Continuous capture, lock/unlock and reboot

The default **Continuous** method uses Android's Accessibility full-display screenshot API. Recording pauses while locked or off and resumes after unlock without another screen-sharing dialog. A started session stays requested until explicitly stopped. A boot receiver and Accessibility reconnection restore the requested session after first unlock while all permissions remain granted. Credentials and encrypted storage are never opened before first unlock. Explicit Stop disables resumption. Android can still stop apps or require re-enabling a sideloaded service after an update; this is not a guarantee against force-stop, OS termination or hardware failure. Reboot recovery was reviewed in code and the merged manifest, without rebooting the test phone.

Calibration probes increasing capture rates up to the app's 30/s test ceiling, including 0.5/s steps around the usual Accessibility bottleneck. Android screenshot-rate rejections stop escalation immediately, are retained in the report, and add retry spacing based on the rejected interval. The last passing rate must sustain capture, inference and encrypted storage for 30 seconds; a failed confirmation steps down and confirms again. This replaces the original imposed 2.5/s ceiling and invalidates its saved calibration. Raw pixels are processed in memory and discarded. Identical consecutive screenshots are suppressed outside calibration. Keyboard-visible frames remain full-screen views of the host app (including Termux and Termux:X11). Notification panels are labeled as System UI; focused split-screen apps retain their own label. Samples include keyboard/window context. Locked/off, protected, unidentified screens and explicitly excluded visible apps pause collection. No Accessibility text nodes or view hierarchies are recorded.

Settings → Capture and automatic resume also offers **Fast session**, using MediaProjection. It can sample faster, but Android may end the session on lock and requires fresh screen-sharing consent. Android 15 may hide private notification content while MediaProjection is active; stopping the session restores normal notification presentation. Continuous Accessibility capture does not create a MediaProjection session. See [Android's screen-sharing protections](https://developer.android.com/about/versions/15/behavior-changes-all#screenshare-protection) and [projection lifecycle](https://developer.android.com/media/grow/media-projection#resource-recovery).

## Inference and OpenCL

The deployment reproduces the trained MobileNetV4 backbone, global pooling and original `960 → 384 → 384` head with LayerNorm, exact GELU and L2 normalization. The preferred LiteRT model runs the complete encoder on OpenCL. If the complete model is unsupported, a clearly reported compatibility backend uses the OpenCL backbone and original CPU head. There is no deliberate CPU-only CNN fallback.

Preprocessing preserves the full screen: aspect-preserving fit/pad to RGB 384×800, AREA shrink or LINEAR enlargement, `(124,116,104)` padding and ImageNet normalization. Native OpenCL preprocessing competes with OpenCV on the actual source dimensions. Alternating repeated trials include fresh ingestion/upload, transform/readback and inference. OpenCL is selected only with at most one RGB-level resize difference, embedding cosine ≥0.9999 and at least 5% pipeline p95 improvement. Selection is cached by deployment, driver, capture method and dimensions.

Preprocessing telemetry measures ingestion plus transform, excluding unrelated permission/label checks. Sustainable calibration uses full end-to-end latency. USB-connected timing is not a battery-energy measurement; no power saving is inferred from GPU use alone.

## Build

This is a source-only repository. Privately provision the authorized F6 student checkpoint as `models/f6_weights.pt`; weights, generated model assets, APKs and recordings are ignored by Git. `models/provenance.json` identifies the expected checkpoint. No DINOv3 weights or source are included.

```bash
./00_export_model.sh                 # Python 3.11; or add --docker
export ANDROID_HOME="$HOME/Android/Sdk"
export JAVA_HOME=/path/to/full/jdk    # JDK 17+ including javac
./01_build_apk.sh
adb -s ADB_SERIAL install -r android/app/build/outputs/apk/debug/app-debug.apk
```

Android requirements: API 30+, compatible OpenCL GPU, SDK platform/build tools 35, NDK 27.2.12479018 and CMake 3.22.1. Gradle 8.11.1/AGP 8.9.2/Kotlin 2.0.21, LiteRT 1.4.1 and OpenCV 4.10.0 are pinned. The build downloads pinned OpenCL headers. `tools/gradle.sh` verifies the Gradle distribution checksum. Retain the app ID, signing key and Keystore data across updates.

The exporter runs 36 numerical comparisons across nine synthetic full-screen inputs and produces a content-hashed manifest. `tools/check_assets.py` rejects missing or altered assets before a build. Each GPU initialization also checks a synthetic golden embedding. Generated assets are local build products, not Git commits.

## Device and host tests

```bash
./04_test_server.sh
./05_test_device_inference.sh ADB_SERIAL
./01_build_apk.sh :fixture:assembleDebug :app:assembleDebugAndroidTest
adb -s ADB_SERIAL install -r android/fixture/build/outputs/apk/a/debug/fixture-a-debug.apk
adb -s ADB_SERIAL install -r android/fixture/build/outputs/apk/b/debug/fixture-b-debug.apk
python3 tools/continuous_capture_smoke.py ADB_SERIAL
```

The controlled capture helper grants only collector prerequisites, preserves other enabled Accessibility services, uses two synthetic fixture apps, tests A→B→A visits, screen lock/unlock and Battery Saver, and stops in `finally`. It requires an unlocked test phone; a secure keyguard requires the owner to unlock it. Outbox instrumentation uses separate randomly named test databases and preserves the production queue. Private logs stay under ignored `reports/work/`. See [validation](reports/VALIDATION.md) and [measured device results](reports/DEVICE_RESULTS.md).

## Optional server

The deployed administrator console and capture endpoint are [kiralycraft.com/projects/uiembeddings](https://kiralycraft.com/projects/uiembeddings/). See [deployment and administration](docs/SERVER_DEPLOYMENT.md) for the Debian VM, user/device management, operations and validation.

```bash
cp .env.example .env
# Set independent MYSQL_PASSWORD and MYSQL_ROOT_PASSWORD secrets.
./02_start_server.sh
docker compose exec api python -m collector.cli create-user alex
```

Docker Compose starts MySQL 8.4 and FastAPI behind `127.0.0.1:8000`; terminate HTTPS at your reverse proxy using `docs/nginx.conf`. Set `TRUSTED_PROXY_IPS` to the exact proxy source. The phone requires HTTPS, normal certificate validation and no redirects. MySQL is not publicly published. The native deployment has also been verified through the public Apache proxy and local MariaDB; see the deployment report above.

In Settings → Server and account, sign in to an upload destination. Existing local recordings remain local until you explicitly confirm their transfer. Transfer decrypts/reseals ownership-bound data atomically and refuses active runs or insufficient space. Account-bound data cannot be rebound to another account. Server tokens expire/revoke without deleting the phone queue.

The encrypted SQLite outbox uses WAL and FULL synchronization on every connection. Uploads are independent of capture, batched and idempotent by run UUID/sequence. Data is deleted locally only after acknowledged server commit. Queue limits pause collection without evicting old samples. Losing app data, its signing identity or its Keystore key loses access to local recordings.

Each return to an application creates a distinct run; transitions, locks, exclusions and ambiguous labels can split visits. Samples preserve acquisition timestamps, Activity source, sequence, mode, preprocessing/backend and capture method. The server only exports closed, complete, nonempty runs as `embeddings.npy`, `samples.jsonl` and `run.json`; run numbers are identifiers, not chronological rankings. No arbitrary crop is treated as a whole-screen embedding.

This is a sideloaded research application. Store-distribution policy review, 16-KB-page devices, prolonged battery/thermal endurance and real MySQL concurrency remain separate validation work.

Signed-in participants can create accounts under **People & devices**. Each creation records its creator and time; administrators see that history alongside per-device statistics. Participants see their own recordings and their direct account creations. See [deployment and migration notes](docs/SERVER_DEPLOYMENT.md#account-creation-history).


The Android **Uploads and local storage** panel separates automatic uploads to the signed-in account from recordings made before sign-in and data belonging to another account/server. **Settings → Server and account → Sync older local recordings to this account** explicitly assigns the older local records, preserving their application labels, timestamps and embeddings. This works while an account-bound recording continues; open local-only runs must first be closed. Local copies are removed only after the server acknowledges them. A small changing automatic-upload queue during recording is normal.
