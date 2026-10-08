# Acceptance checklist before a dataset collection campaign

These tests are **instructions, not claims that the supplied source has passed them**. Use a dedicated test install/account. Preserve your deployed model manifest, APK, signing key and dependency versions once validated.

See [measured Xperia results](../reports/DEVICE_RESULTS.md) for completed checks. `05_test_device_inference.sh ADB_SERIAL` reproduces GPU and synthetic preprocessing checks. `tools/continuous_capture_smoke.py` covers the controlled continuous-capture workflow; the remaining deployment checks below must be assessed separately.

Use `python3 tools/continuous_capture_smoke.py ADB_SERIAL --rates-only` to run fresh calibration and compare all three modes on the animated fixture without locking or rebooting. It stops collection in `finally`; its read-only audit requires new samples after that calibration and checks both saved targets and actual acquisition intervals. Calibration should retain any Android screenshot-rate rejection, stop escalation, and sustain the selected lower rate for 30 seconds. The three displayed limits must be distinct even below 5/s. Missing screenshots must not count as successful inference.

## Build and numeric checks

Run `00_export_model.sh` and retain its complete output. Require all nine synthetic inputs to pass original/split/TF/LiteRT parity. Optionally add representative local screenshot files through `--parity-images`; they are not packaged. Inspect the generated manifest for 960 backbone features, 384 output dimensions, NHWC [1,800,384,3], no quantization and only the reviewed GPU operator set.

Run `01_build_apk.sh`. No successful build or APK is included in the delivery archive. Run host tests and inspect warnings/errors rather than skipping them. In Android Studio, inspect the merged manifest for mediaProjection foreground service, usage/accessibility permissions, disabled backups, no cleartext traffic and the optional OpenCL native library declaration.

Outbox instrumentation uses isolated randomly named databases and removes only those test databases. Permission UI tests require their documented initial permission/onboarding state. The recorded-visits test requires the controlled capture helper first. Select each suite explicitly on the intended device. A real GPU is required for GPU parity; an emulator using CPU rendering is not evidence of target-phone GPU performance.

```bash
adb shell getprop ro.build.version.release
adb shell getprop ro.product.model
adb shell getconf PAGE_SIZE
adb logcat -c
./tools/gradle.sh -p android :app:connectedDebugAndroidTest
adb logcat -d | grep -Ei 'tflite|litert|delegate|opencl|opengl|gpu'
```

Inspect logs for successful GPU delegate creation/partitioning and any unsupported-operation warnings. Do not accept CPU-only timing as GPU timing. Use the LiteRT benchmark/profiling tools with the same exported backbone when partition coverage cannot be determined from Java logs. Record which backend/driver actually ran. The in-app synthetic GPU cosine must be at least 0.9999; this alone proves numerical similarity, not delegation coverage.

Check all APK native dependencies on 16-KB-page devices using Android's documented ELF/ZIP alignment tests. The pinned OpenCV 4.10 native artifact may need replacement/rebuilding for that target. Updating native dependencies requires rebuilding, repeating numeric/latency tests and recording the changed APK. AGP packaging alignment cannot repair an incompatible prebuilt ELF library.

## Capture, timing and labels

Select two non-sensitive test applications and verify `X → Y → X` creates distinct X runs. Within X, move between Activities, open a dialog and compare UsageEvents activity vs separate Accessibility window_class. Missing/ambiguous activity should remain null rather than a dialog widget class being mislabeled as an Activity.

Test portrait/landscape changes on API 30–33 and API 34+ separately. Older APIs use periodic display-size inspection; newer APIs use the capture resize callback. Assert preprocessing sees the actual full frame without extra unexpected letterbox bars. Check source dimensions/rotation in sample metadata and compare a controlled local golden image through the Android preprocessing implementation against the original Python transform. The synthetic 1080×1920 RGBA ImageReader fixture passed on the Xperia; real MediaProjection frames and other resolutions still need coverage.

Run calibration with the full-screen animation visible, then inspect **View last benchmark**. Repeat once after the phone is warm, then collect while scrolling in representative apps. Confirm p95 stays within its frame budget and the target reduces under pressure. Do not use a cool synthetic benchmark alone to declare long-session FPS sustainable.

Leave an app static. The recorder should avoid synthesizing duplicate time-grid samples and keep the run open. Its next real frame or eventual run end should preserve elapsed dwell through timestamps. A first stable frame should be processed even when it was held briefly during label settling.

Open a keyboard, notification shade, split-screen, PiP and excluded apps. This build deliberately pauses/splits rather than assigning mixed screen content to an app. Verify it resumes when the unambiguous allowed app returns. Check task-switch recents thumbnails and protected/blank surfaces are not assumed to be ordinary application screens.

For Continuous Accessibility capture, lock/off must pause collection and unlock must resume the requested session without a screen-sharing dialog. Explicit Stop must clear the resume request. For Fast MediaProjection sessions, lock or the projection chip ends consent and requires a new grant. Revoke Usage Access/Accessibility, stop from the foreground notification, and kill the process. Recording requires valid grants and a prior explicit start. Queued data must survive all cases except explicit app data clearing/uninstall/key loss.

## Offline and database tests

1. Start collection, disable Wi-Fi/mobile data, switch X/Y/X, stop and reopen the app. Pending counts must persist. Reconnect, wait for the scheduled/upload-now path and confirm the server receives every committed sample and distinct run boundary.
2. Interrupt the connection after the server commits but before the phone sees the response. Retry must not increase the database row count for an existing `(run_id, sequence)`.
3. Close a run while a batch is in flight. The closure must remain queued until separately acknowledged; the server's complete flag must stay false until every expected sequence is present.
4. Set a small queue limit and fill it offline. Capture must pause without evicting old samples, and automatically resume after upload frees sufficient payload/free-disk space. Check available disk as well as the payload counter.
5. Revoke the session token while recording offline. A 401 must retain the queue. Stop capture, reauthenticate to the same account and confirm delivery. Attempt another account/URL and confirm cached data is not uploaded under it.
6. Submit the same sample with changed metadata/vector bytes: expect 409 and no partial commit. Submit unknown model, malformed vector, NaN, non-normalized vector, foreign run ID, too-large body and unexpected screenshot field. Check that errors do not echo sensitive payloads.
7. Test concurrent uploads against a **separate disposable MySQL 8.4 instance**, including simultaneous first runs for the same package and identical retries. The 32 executed server tests use SQLite, not InnoDB; MySQL locking/deadlock behavior still needs this integration test.
8. Export complete runs. Check arrays are float32 `(T,384)`, row count equals metadata count, sequence has no gaps, and times increase within each run. Verify incomplete runs are absent from default export and cross-account data is inaccessible.

## Storage and operations

Inspect app-private storage on a test/debug build: it should contain model assets, encrypted outbox payloads, preferences and WorkManager state, not captured PNG/JPEG/MP4 files or raw frame archives. Inspect HTTP bodies at your own test TLS endpoint and confirm only embeddings and metadata are sent. Do not enable production body logging to perform this test.

Back up/restore the MySQL volume in a disposable environment. Validate reverse-proxy forwarded-header handling, HTTPS certificates, host firewall, non-public MySQL, login rate limiting and administrative session revocation. Protect exports as sensitive data. Define retention and deletion operations before enrolling other people; this research implementation does not claim a formal compliance/security audit.
