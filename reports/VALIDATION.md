# Validation status of this delivery

## Current device verification: 2026-10-08

The original delivery status below is historical. Subsequent verification on the local build host and Sony Xperia XQ-DQ72 completed the full model conversion, APK build, two JVM unit tests and seven Android instrumentation tests. The device logs show all 62 backbone nodes delegated to OpenCL. GPU inference and synthetic RGBA/OpenCV preprocessing passed parity gates. A startup crash in `LocalStore.onConfigure` was fixed by executing the row-returning WAL checkpoint PRAGMA through `rawQuery`; the device regression test verifies WAL, FULL synchronization and the checkpoint interval.

See [the measured results](device-20261008/RESULTS.md). Continuous MediaProjection behavior, calibration/thermal endurance, activity labels, real-server/MySQL operation and 16-KB devices still require acceptance testing.

## Original delivery evidence

## Executed successfully

| Check | Result | Evidence |
|---|---|---|
| Cleaned F6 checkpoint | Every retained student tensor compared equal to the supplied original; optimizer/RNG state excluded | `models/provenance.json` |
| Actual trained Kotlin embedding head vs original PyTorch head | 82 vectors; minimum cosine 0.999999999999829; maximum elementwise absolute error 9.685754776000977e-08 | `reports/kotlin_head_parity.json`, `tools/test_kotlin_head.py` |
| Python export preprocessing vs original deterministic transform | Seven resolutions/aspect ratios; all maximum differences exactly 0.0 in this environment | `reports/preprocessing-parity.json` |
| Python backend API/contract tests | 30 passed, SQLite-backed | `reports/server-tests.xml`, `reports/server-tests.txt` |
| Kotlin/Gradle source parsing | No syntax errors in the parsed source set | `reports/kotlin-syntax.txt` |
| Python syntax | `compileall` successful for exporter, backend and helper scripts | Validation execution |

The head comparison compiles and executes the **same Core.kt shipped in the app**, with actual trained head tensors. It also exercises rate reduction, thermal backoff, ceiling bounds, percentile logic and nonfinite/size rejection. It is not a reimplementation-only comparison.

Server tests include authentication/session revocation, account isolation, login throttling, immutable run identity, idempotent retry/conflict rollback, out-of-order delivery, gap-aware completion, late open-run retries, app-return run numbering, monotonic ordering, closure constraints, vector validation, prohibited screenshot fields, package/path validation, request-body size and complete-only array export.

## Not executed; required before deployment

- Complete PyTorch backbone instantiation with TIMM and conversion to TensorFlow/LiteRT.
- Full end-to-end PyTorch/LiteRT and actual phone GPU parity; export and startup gates exist but were not passed here.
- Android Gradle dependency resolution, APK compilation, Android type checking, unit/instrumentation execution on Android.
- GPU delegate partition coverage, OpenCL/OpenGL vendor behavior, measured phone FPS, battery/thermal endurance.
- Android OpenCV preprocessing parity, screen capture and Activity attribution under real window/lifecycle behavior.
- Real MySQL/InnoDB deployment, concurrency/deadlock and proxy integration tests.
- Native-library compatibility with 16-KB memory pages.

The environment lacked the Android SDK/Gradle, TIMM, TensorFlow, PyMySQL, Docker and a connected Android GPU device; outbound dependency downloads from the execution container failed. No APK, backbone `.tflite`, successful deployment manifest, observed GPU FPS or claim of device validation is supplied. The trained weights and extracted head binary are supplied; executable build/test/export procedures are included.

## Test environment versus pinned build environments

Host verification used Python 3.13, PyTorch 2.10.0+cpu and Kotlin CLI 1.9.0 on Java 21. The host backend used the packages available in the delivery environment, not a freshly installed pinned container. `reports/environment.json` records their versions. The converter intentionally uses its own Python 3.11/TIMM/TensorFlow pins, and Android uses Kotlin 2.0.21. A Kotlin parser check is not an APK compile. Passing host tests is not evidence that every pinned Android/converter/server dependency set has already been built.

Use `docs/DEVICE_ACCEPTANCE.md` as the release gate. Retain the generated manifest/parity reports and test results from the actual target phone/build host with any collected dataset.
