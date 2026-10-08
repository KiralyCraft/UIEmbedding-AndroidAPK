# Continuous Android F6 embedding collector

An Android screen-to-embedding collector and authenticated Python/MySQL ingestion service built around the supplied **F6 checkpoint, step 26,000**. The device produces 384-dimensional embeddings. Raw captured pixels remain in RAM; no screenshot/video files or screen pixels are uploaded.

**Current validation (2026-10-08):** the F6 model was converted with all 27 export parity checks passing, the Android debug APK compiled, and server-free inference was tested on a Sony Xperia XQ-DQ72 running Android 15. All 62 backbone nodes delegated to OpenCL. One hundred repeated embeddings and the production RGBA/OpenCV preprocessing path passed numerical checks; two JVM unit tests and five outbox device tests also passed. A database startup crash discovered on the device was fixed. See `reports/device-20261008/RESULTS.md` for measurements and evidence. MediaProjection collection, sustained capture rates, real-server integration and MySQL concurrency remain unvalidated.

## Architecture

```text
Android consent + visible foreground capture service
  → MediaProjection / latest ImageReader frame (RAM only)
  → OpenCV: aspect-preserving fit/pad → RGB 384×800 → normalize
  → LiteRT GPU: trained MobileNetV4 convolutional backbone + pooling
  → CPU: trained 960→384→384 embedding head + L2 normalization
  → attach foreground package/activity/timestamps/model identity
  → encrypted durable SQLite outbox
  → authenticated HTTPS batch upload, idempotent retry
  → your reverse proxy terminates TLS
  → HTTP FastAPI service
  → MySQL: users / devices-by-ID / applications / runs / samples / benchmarks
```

### What “continuous” means

Capture remains active across foreground app changes until explicitly stopped or Android ends the projection. It does not stop after one video, one app visit, a successful upload, or loss of Internet connectivity. Uploading is independent of inference. A switch from `X → Y → X` creates separate runs for both visits to X.

This is **continuous sampling**, not an MP4 recorder, lossless compositor-frame recorder, or a guarantee of uninterrupted data. The sampler consumes the newest available frame at the calibrated target rate, discards capture backlog and never builds an unbounded inference queue. It does not fabricate extra samples for static screens. Each stored embedding retains its acquisition wall time, monotonic time and raw image timestamp. Gaps are real and must not be treated as fixed-FPS video.

Pixels may be retained briefly in RAM while the foreground label settles, then are processed at most once. Foreground association is a best-effort OS-event label, not a privileged atomic proof that every pixel belongs to one Activity. Transitions, overlays, lock screens and unsupported cases are deliberately conservative.

### Actual F6 model

The original global encoder is reproduced, not its ImageNet classifier or dense-patch head:

* `mobilenetv4_conv_small.e2400_r224_in1k`, trained `forward_features` output with **960** channels. The checkpoint config says feature_dim=1280, but its embedding-head weight is `[384, 960]`; using 1280 would be wrong.
* Spatial global average pooling, then `Linear(960,384,bias) → LayerNorm(eps=1e-5) → GELU → Linear(384,384,no bias) → L2 normalization`. Dropout is disabled at inference.
* GPU runs the convolutional backbone and pooling. The small original head runs in Kotlin on CPU. This avoids silently replacing LayerNorm/GELU or quantizing the trained model. It is **not an all-operations-on-GPU deployment**.
* NHWC float32 input is RGB, width 384 and height 800. Fit-pad is aspect preserving, with RGB padding `(124,116,104)`, OpenCV AREA when shrinking and LINEAR when growing, ImageNet mean/std, and no training augmentation.

`models/f6_weights.pt` contains the supplied trained student tensors plus necessary config, with optimizer/RNG state removed. `models/provenance.json` records the original ZIP/checkpoint hashes. No retraining or online pretrained-weight download is required.

## 1. Export the deployment model

Run on a machine with Python 3.11 and network access to install the build dependencies:

```bash
./00_export_model.sh
```

Or build/run the pinned converter container:

```bash
./00_export_model.sh --docker
```

The exporter uses the original vendored model implementation and TIMM 1.0.20, strictly loads the checkpoint, traces the convolutional backbone, translates a narrow reviewed operation set to NHWC TensorFlow, and converts it to builtin-only float32 LiteRT. It explicitly preserves PyTorch symmetric padding, TIMM BatchNormAct activations, depthwise convolutions, residuals and layer scale. It aborts on unexpected operations rather than guessing.

Nine deterministic synthetic inputs are used for PyTorch/split-model, TensorFlow and LiteRT parity. Optional local screenshots can extend **build-time** validation without being copied to the app or server:

```bash
./00_export_model.sh --parity-images /path/to/local/validation-images
```

Only after every check passes are these assets committed:

```text
android/app/src/main/assets/
  f6_backbone.tflite
  f6_head.bin
  golden_input.bin         # synthetic normalized test input, not a captured screenshot
  golden_embedding.bin
  f6_manifest.json         # content-hashed deployment identity and parity results
```

A success manifest is intentionally absent from this archive until you run the converter. The head binary is already extracted from the actual trained weights. Conversion failure removes the old success manifest so a stale export cannot silently be accepted.

The phone validates asset hashes and compares its GPU+Kotlin output against the synthetic PyTorch golden embedding on each capture start. Cosine must be at least **0.9999**. It does not switch into a deliberate CPU-only mode on failure. LiteRT's Java delegate API does not expose a robust public partition-coverage attestation; confirm actual GPU delegation with the profiler/log checklist in `docs/DEVICE_ACCEPTANCE.md`.

## 2. Build/install Android

Requirements: Android SDK platform 35/build-tools 35.0.0, Java 17 or newer, and an Android **11/API 30+** device with a functioning compatible GPU delegate. Project versions are AGP 8.9.2, Gradle 8.11.1, Kotlin 2.0.21, LiteRT 1.4.1. This is a sideloaded research app, not a claim of current Play Store policy compliance.

```bash
export ANDROID_HOME="$HOME/Android/Sdk"  # or use android/local.properties
./01_build_apk.sh
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

`tools/gradle.sh` downloads Gradle and verifies the distribution against its HTTPS-published SHA256; no unverified wrapper JAR is bundled. Android Studio can alternatively open `android/`. The debug APK is for testing. Configure your own release signing key for long-term collection; retain the same key/application ID for updates so app data is not lost.

OpenCV native dependencies are pinned to 4.10.0. The native library/16-KB-page-size compatibility of this APK has **not** been validated. A successful compile is not enough to establish support on newer 16-KB devices; inspect/test the native libraries as described in the acceptance checklist.

### Try inference before configuring a server

After exporting the model, run the two server-free instrumentation checks on one explicitly selected device:

```bash
export ANDROID_HOME="$HOME/Android/Sdk"
# This host's default Java 17 is a JRE; select an installed JDK with javac.
export JAVA_HOME=/path/to/your/jdk
./05_test_device_inference.sh ADB_SERIAL
```

The script builds and installs the debug app and test APK, then tests 100 GPU-backed model invocations plus a synthetic full-resolution RGBA frame through the production preprocessor and embedding engine. It needs no account, server, accessibility permission or screen-sharing consent. It does not run the outbox tests that purge pending data. Reports include numerical parity, model-only p50/p95 latency, preprocessing error and GPU delegation logs. Do not interpret model-only latency as sustainable capture FPS.

The preprocessing check requires exact source RGB pixels and allows at most one RGB intensity level of difference after uint8 resizing, plus floating-point tolerance. It separately requires the resulting embedding cosine to be at least 0.9999 against the PyTorch reference. Android OpenCV 4.10 and host OpenCV 4.11 were not byte-identical after resizing the tested fixture; the observed embedding cosine was 0.999999999936.

## 3. Start MySQL and Python behind your reverse proxy

```bash
cp .env.example .env
# Edit .env: independent random hexadecimal MYSQL_PASSWORD and MYSQL_ROOT_PASSWORD.
# Example secret generator: python3 -c 'import secrets; print(secrets.token_hex(32))'
./02_start_server.sh

docker compose exec api python -m collector.cli create-user alex
# CLI prompts for a password; it is not passed on the command line.
```

The script starts MySQL 8.4, creates the initial schema, registers the validated model manifest, then starts FastAPI on **127.0.0.1:8000**. MySQL is not published externally. Docker Compose is the deployment path; SQLite is used only by local development/tests.

Configure a dedicated HTTPS hostname using `docs/nginx.conf`, or apply the equivalent settings to your existing proxy. The proxy sends HTTP to `127.0.0.1:8000`. Set `TRUSTED_PROXY_IPS` to the exact proxy source address as observed inside the API container; a host proxy may appear as the Docker bridge gateway, not 127.0.0.1. Never use `*` on an exposed backend. Proxy forwarding headers must be overwritten, not blindly passed from the client.

The app requires HTTPS, validates the certificate normally, rejects embedded URL credentials and redirects, and supports a path-prefix URL when the proxy strips that prefix. Development self-signed TLS requires a proper trusted certificate arrangement; this project deliberately has no trust-all switch.

`init-db` creates schema v1; it is **not** an automatic future migration engine. Back up the database before schema changes. `/healthz` is process liveness, not proof that MySQL is writable. MySQL volume data and exports are not application-encrypted; protect the host, access, backups and storage encryption.

User sessions use Argon2id password hashes and random opaque bearer tokens, stored hashed on the server. Tokens expire after 90 days by default. No public self-registration is exposed. Administrative actions:

```bash
docker compose exec api python -m collector.cli reset-password alex
docker compose exec api python -m collector.cli revoke-sessions alex
```

Both revoke existing sessions. The phone retains pending data after 401 and requires reauthentication to the **same server/account** before it can drain that queue.

## 4. Start collection on the phone

Enter the HTTPS server URL and sign in. Select the application packages to include. Enable the collector's Accessibility service and Usage Access. Some sideloaded builds may require Android's explicit restricted-settings approval before enabling Accessibility.

Choose **Calibrate GPU and start continuous capture**, and grant **entire-screen** projection, not single-app sharing. A dedicated activity shows a moving synthetic pattern while the app measures capture/resize/inference/head/encrypted-storage throughput. Keep it visible. Calibration then transitions into continuous collection; open an allowed app.

Later use **Start continuous capture with saved calibration**. You can stop from the app or the persistent notification. The sampler will not record the collector itself, the system UI, excluded apps or ambiguous windows.

### Benchmark and runtime adaptation

The calibration sweep tests 0.25, 0.5, 1, 2, 5, 10, 15, 20 and 30 samples/second. It uses warmups, at least eight-second candidate windows, followed by a thirty-second sustained confirmation at the best passing rate. Failed confirmation tries a lower previously passing rate and confirms that rate separately.

Passing requires p95 processing time within 75% of the frame budget, at least 90% of requested throughput, at most 5% missed deadlines, and at least 80% fresh frames. The benchmark includes encrypted SQLite writes, not just the model invocation. It excludes network latency because uploads are asynchronous. Reports contain achieved rates, p50/p95, fresh counts, missed deadlines, thermal status and GPU golden parity.

During collection the controller periodically reduces the target if processing p95 is too slow or Android reports thermal pressure. It never increases above the calibrated ceiling. Severe thermal/storage conditions pause sampling and resume when safe; they split runs. This controls **embedding sampling**, not the compositor/display refresh rate.

## 5. Offline durability and retry semantics

Samples are committed to an app-private SQLite WAL outbox (`synchronous=FULL`) before being eligible for upload. Sample/run/report payloads are AES-GCM encrypted with an Android Keystore key and authenticated contexts bound to account/run/sequence. Capture does not wait for the network.

The uploader sends batches of up to 128 samples. Local samples are deleted only after the server commits and explicitly acknowledges the corresponding sequence numbers. `(run UUID, sequence)` is unique in MySQL; an identical retry is acknowledged without inserting a duplicate, while conflicting data is rejected. A concurrent run-close cannot be erased by the acknowledgment of an earlier open-run batch.

Transient failures back off with jitter. Uploads retry while capture is active and via network-constrained WorkManager afterward. A 401, model mismatch or conflicting payload leaves data intact and shows an action-required status. **Retry uploads now** clears the retry block after the underlying issue is corrected.

The default outbox quota is **512 MiB of encrypted sample payload**, configurable from 64 to 8192 MiB. SQLite metadata/WAL/file overhead is additional. There is a separate low-free-storage guard. No oldest-data eviction occurs: at the limit sampling pauses; existing data continues uploading; collection resumes after sufficient space is freed. SQLite may reuse freed pages without shrinking its file immediately.

Server URL + user UUID define queue ownership. Signing into another identity is rejected while a different owner's data remains. Explicitly deleting the pending queue is destructive, may leave incomplete server runs, and is not a remote server deletion or guaranteed physical secure erase. App uninstall/data clearing or loss of the Keystore key loses the local queue. Sudden process death can lose a currently in-flight frame before its SQLite commit, not already committed samples. At next process start, unfinished local runs are closed at their last committed sample with `process_interrupted`.

## Dataset grouping and export

An uninterrupted confidently labeled foreground visit is one run. Package switches, excluded/ambiguous windows, visible keyboard, critical thermal/storage pause, projection end and display resizing can split visits. Activity changes within the same package may remain in the same run; activity is a per-sample label. A keyboard pause therefore can produce two runs even without leaving an app.

```text
com.example.appX / device UUID / run_000001_<run UUID>/
  embeddings.npy   # shape (T, 384), float32, no pickle
  samples.jsonl    # corresponding row-level timing/labels/processing metadata
  run.json        # model, package, version, device, session, bounds, end reason
com.example.appY / device UUID / run_000001_<run UUID>/
com.example.appX / device UUID / run_000002_<run UUID>/
```

Run numbers are allocated per **user + device + package** on first server receipt. Offline/reordered delivery means run numbers are identifiers, not guaranteed chronological ranks. Sort by capture timestamps/session as appropriate. Samples inside each run are ordered by sequence and strictly increasing monotonic acquisition time. Wall clocks can change; monotonic times are comparable only within the relevant device boot/session.

A run is `complete` only when the server has its closure and every sample in `0..expected_samples-1`. Export excludes open/incomplete and empty runs:

```bash
./03_export_dataset.sh alex
./03_export_dataset.sh alex com.example.appX
```

Every export uses a new timestamped folder under `exports/` and refuses overwrites. Dataset ingestion example:

```python
from pathlib import Path
import json
import numpy as np

run_dir = next(Path("exports").rglob("run.json")).parent
vectors = np.load(run_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False)
labels = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text().splitlines()]
assert vectors.shape == (len(labels), 384)
# Do not assume equal time spacing or pool vectors from different model_id values.
```

Package/activity are OS metadata, **not semantic category labels** such as shopping/social/banking. Add a versioned package-to-category labeling layer later. No app-category inference from names is fabricated here.

## API and tests

Authenticated endpoints: `POST /v1/ingest`, `POST /v1/benchmarks`, `GET /v1/models`, `GET /v1/runs`, `GET /v1/runs/{id}/samples`; login/logout are under `/v1/auth/`. OpenAPI is at `/docs` on the backend. The strict schemas reject unexpected fields, screenshots, wrong dimensions, nonfinite/unnormalized embeddings, invalid package paths and ordering conflicts. Request bodies are capped at 2,000,000 bytes.

```bash
./04_test_server.sh                         # isolated SQLite-backed API/contract tests
python tools/test_kotlin_head.py             # needs torch/numpy/Kotlin CLI; exact trained head
./tools/gradle.sh -p android :app:testDebugUnitTest
./tools/gradle.sh -p android :app:connectedDebugAndroidTest
```

**Instrumentation tests intentionally clear the test installation's outbox. Never run them against an installation holding valuable pending samples.** GPU instrumentation requires a real supported device and generated deployment assets. Full device validation, network-loss tests, Android limitations and primary documentation are in `docs/DEVICE_ACCEPTANCE.md`, `docs/DESIGN.md` and `docs/SOURCES.md`.
