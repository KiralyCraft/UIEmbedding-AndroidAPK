# Xperia validation — 2026-10-08

Device: Sony Xperia XQ-DQ72, Android 15/API 35, 4-KB memory pages. Actual display/capture dimensions: 1096×2560. USB charging was connected, so these results do not measure energy use or unplugged endurance.

## Passed

- APK and instrumentation build; six JVM tests.
- 36 export comparisons across nine synthetic full-screen inputs using the unchanged trained F6 student weights.
- Full encoder GPU: all 80/80 nodes delegated to a single OpenCL partition. Nine synthetic GPU inputs passed, and 100 repeated golden invocations had minimum cosine 0.9999999999993476 and maximum absolute error 1.7508864402770996e-7. Latest model-only median 11.00 ms, p95 13.21 ms. The trained head is included on GPU.
- RGBA preprocessing golden parity and native OpenCL parity over eight sizes/aspect ratios, including actual Xperia dimensions. Maximum resized RGB difference approximately one intensity level; every embedding cosine ≥0.9999.
- Corrected live preprocessing comparison selected OpenCL: complete pipeline p95 11.62 ms versus 14.91 ms for CPU. Upload/ingestion p95 1.44 ms, GPU transform/readback p95 1.66 ms in that calibration probe. These are timing comparisons, not power measurements.
- Continuous Accessibility calibration confirmed 2.5/s over 30.097 seconds: 75 processed/fresh frames, achieved 2.492/s, complete capture-to-storage p95 160.46 ms, no missed deadlines, Android thermal status 0. Capture/readback and scheduling explain the difference from model-only timing.
- Actual encrypted A→B→A fixture visits with contiguous sequences and monotonic acquisition timestamps. After the owner manually unlocked the phone, a distinct A visit committed 14 samples without new screen-sharing consent. The preceding visit closed with `screen_locked`.
- Battery Saver committed fixture samples with target ≤1/s. Typical live preprocessing medians were 5.6–6.4 ms, inference medians 9.8–10.7 ms.
- Stage 3 grant remains stable through Accessibility inspection/rebinding. Notification denial blocks Next; Applications and Settings remain accessible. The final permission/UI plus GPU/preprocessing regression batch passed four device tests.
- Eight isolated outbox device tests passed: WAL/FULL synchronization across connections, ownership authentication, acknowledgement races, atomic local transfer/rollback and traversal of more than one CursorWindow.
- 32 SQLite-backed server API tests; source/shell/Python checks passed.

## Corrections made during validation

Accessibility grant was originally conflated with transient service connection, causing Stage 3 flicker. Permission state now follows Android's grant; runtime labeling still waits for a bound, valid service. Old service-instance callbacks cannot disconnect a newer instance.

The old preprocessing counter included the entire pre-inference capture tick. It now times only ingestion and transformation. The first CPU/OpenCL selector also incorrectly charged every GPU trial the initial allocation/upload latency; repeated fresh uploads and alternating order reversed that misleading result. CPU tensor copies were eliminated.

Android 15 hides private notification content during MediaProjection. Continuous mode uses Accessibility capture instead; no MediaProjection session was active during its accepted run. Fast sessions retain Android's screen-sharing behavior and explain it in the UI.

The capture helper initially assumed ADB could dismiss a secure keyguard and that an enabled service was necessarily bound. It now waits for the owner to unlock, verifies the bound screenshot capability and foreground recording service before fixtures, and stops in `finally`. A later helper click failed on an already-selected Battery Saver radio; the stored-data audit independently passed unlock recovery and the helper now handles that state. Raw failed attempts remain private; they are not counted as successful complete helper runs.

## Reviewed, not exercised

Reboot recovery: explicit RECEIVE_BOOT_COMPLETED permission and non-exported receiver are present in the built merged manifest. The receiver and Accessibility connection share the resume path; first unlock, persisted user request, all grants and screenshot capability are required. Stop clears the request. The foreground type is specialUse, not mediaProjection. The phone was **not rebooted**, as requested.

## Remaining scope

No 24-hour endurance guarantee, unplugged energy comparison, real MySQL/proxy deployment/concurrency test, or 16-KB-page device test. Continuous mode deliberately caps requests at 2.5/s before calibration because Android limits Accessibility screenshots; Fast sessions offer higher rates with fresh consent after lock. OEM restrictions and force-stop can still require user intervention.
