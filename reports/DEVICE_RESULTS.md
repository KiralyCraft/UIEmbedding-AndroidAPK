# Xperia validation — 2026-10-08

Device: Sony Xperia XQ-DQ72, Android 15/API 35, 4-KB memory pages. Actual display/capture dimensions: 1096×2560. USB charging was connected, so these results do not measure energy use or unplugged endurance.

## Passed

- APK and instrumentation build; eleven JVM tests, including adaptive calibration failure/confirmation and distinct low-rate mode coverage.
- 36 export comparisons across nine synthetic full-screen inputs using the unchanged trained F6 student weights.
- Full encoder GPU: all 80/80 nodes delegated to a single OpenCL partition. Nine synthetic GPU inputs passed, and 100 repeated golden invocations had minimum cosine 0.9999999999993476 and maximum absolute error 1.7508864402770996e-7. Latest model-only median 11.00 ms, p95 13.21 ms. The trained head is included on GPU.
- RGBA preprocessing golden parity and native OpenCL parity over eight sizes/aspect ratios, including actual Xperia dimensions. Maximum resized RGB difference approximately one intensity level; every embedding cosine ≥0.9999.
- Corrected live preprocessing comparison selected OpenCL: complete pipeline p95 11.62 ms versus 14.91 ms for CPU. Upload/ingestion p95 1.44 ms, GPU transform/readback p95 1.66 ms in that calibration probe. These are timing comparisons, not power measurements.
- Initial Continuous Accessibility calibration confirmed the then-imposed 2.5/s ceiling over 30.097 seconds: 75 processed/fresh frames, achieved 2.492/s, complete capture-to-storage p95 160.46 ms, no missed deadlines, Android thermal status 0. See the adaptive follow-up below for the measured capture limit.
- Actual encrypted A→B→A fixture visits with contiguous sequences and monotonic acquisition timestamps. After the owner manually unlocked the phone, a distinct A visit committed 14 samples without new screen-sharing consent. The preceding visit closed with `screen_locked`.
- Battery Saver committed fixture samples with target ≤1/s. Typical live preprocessing medians were 5.6–6.4 ms, inference medians 9.8–10.7 ms.
- Stage 3 grant remains stable through Accessibility inspection/rebinding. Notification denial blocks Next; Applications and Settings remain accessible. The final permission/UI plus GPU/preprocessing regression batch passed four device tests.
- Eight isolated outbox device tests passed: WAL/FULL synchronization across connections, ownership authentication, acknowledgement races, atomic local transfer/rollback and traversal of more than one CursorWindow.
- 34 SQLite-backed server API tests, including adaptive report upload/retry and compatibility with previously committed legacy benchmark hashes; source/shell/Python checks passed.

## Adaptive calibration follow-up

The fixed 400-ms request spacing and 2.5/s calibration cap were removed. The Xperia passed probes at 0.25, 0.5, 1, 2 and 2.5/s, then returned `ERROR_TAKE_SCREENSHOT_INTERVAL_TIME_SHORT` during the 3/s probe. Calibration stopped escalation and confirmed 2.5/s for 30.001 seconds: 75 fresh frames, achieved 2.4999/s, no missed deadlines, capture-to-storage p95 88.75 ms, thermal status 0. Both preprocessing and full inference used OpenCL. This is a sustainable tested rate on this phone, not a claim that all Android devices have the same ceiling or that every intermediate rate was probed.

`python3 tools/continuous_capture_smoke.py QV7713JCJD --rates-only` completed successfully without locking or rebooting the phone. Each mode recorded the animated fixture for 20 seconds; an independent read-only instrumentation audit decrypted only fixture samples newer than this calibration and verified their target rates and actual acquisition spacing:

| Mode | Target /s | Samples | Measured /s from median acquisition interval |
|---|---:|---:|---:|
| Maximum Detail | 2.5 | 50 | 2.519 |
| Balanced | 1.25 | 25 | 1.255 |
| Battery Saver | 0.625 | 13 | 0.627 |

The mode menu displays these effective targets. Old imposed-ceiling calibrations are invalidated. Test collection was stopped, encrypted queued data retained, and Maximum Detail restored after the check. USB charging prevents an energy-savings claim from these rate measurements alone.

## Corrections made during validation

Screenshot-diagnostic correction: the initial static-screen and 149-pixel scroll similarity figures (including 0.9976025672, 0.9970867950 and 0.9995943322) are **superseded and invalid as full-screen embedding comparisons**. `adb exec-in ... tee` could return before the complete PNG was written; an observed transfer retained only 12,285 of 181,570 bytes. Android's BitmapFactory still decoded partial PNGs, so image dimensions alone did not detect the problem. The diagnostic now transfers through `adb shell -T`, verifies source/device SHA-256 equality, requires the PNG IEND footer and requires an expected checksum before decoding. This affected only the file-based diagnostic; production Accessibility capture does not use PNG transfers. The original raw-image checks still established the 149-pixel scroll and restoration, but their embedding figures must not be used. Invalid private reports are retained separately for traceability.

Verified further-scroll comparison: three captures of the owner's new Catima position were each 181,570 bytes, SHA-256 identical and fully transferred; their RGBA and normalized tensors were identical. Native OpenCL preprocessing plus the full OpenCL encoder produced **bit-for-bit identical embeddings** for all three captures and 20 repeated invocations: cosine 1, L2 distance 0, maximum component difference 0. Against the most recent retained original-position Catima recording before the prior diagnostic (sample time 18:11:10.472), the new capture's cosine was **0.9927368950**, L2 distance 0.1205247277 and maximum component difference 0.0182778928. The reference was read from the encrypted outbox, restricted to the same owner/package and a closed visit before the diagnostic cutoff; it is not the exact original PNG probe vector. Its model, OpenCL inference/preprocessing backends and 1096×2560 geometry matched.

A fresh Accessibility API capture of the unchanged new position independently gave cosine **0.9930938508** against that original recorded reference, **0.9998367136** against the verified PNG capture and **0.9998365463** against the latest production sample of the new position. This cross-check supports the complete-frame result and capture-path agreement. Recording was resumed without changing the owner's new scroll position. Raw diagnostic screenshots were removed; the valid private report is `reports/work/further-scroll-check/verified-device-report.json`, with vectors retained locally for subsequent comparisons. No general same-place threshold is established by this one example.

Force Stop/relaunch follow-up: the original `Waiting: ro.ubb.uicollector` status represented the intentionally excluded collector screen; opening the animated fixture proved capture/inference were working. A separate `am force-stop` reproduction removed the collector from Android's enabled Accessibility list. The updated app showed `Accessibility is off`, disabled Start, and opened Android's Accessibility settings through the recovery button. After restoring the collector's service (preserving Termux:X11's service), a 12-second fixture visit committed 15 samples in Balanced mode. Returning to the collector displayed `Ready · open another app to record` and explicitly explained its own-screen exclusion. The APK build and all 11 JVM tests passed. Capture was stopped after testing, with Accessibility bound and no crashed services. UIAutomator itself temporarily disconnects Accessibility, so active-status text was verified using screenshots rather than hierarchy dumps; counts were then checked from the UI.

Accessibility grant was originally conflated with transient service connection, causing Stage 3 flicker. Permission state now follows Android's grant; runtime labeling still waits for a bound, valid service. Old service-instance callbacks cannot disconnect a newer instance.

The old preprocessing counter included the entire pre-inference capture tick. It now times only ingestion and transformation. The first CPU/OpenCL selector also incorrectly charged every GPU trial the initial allocation/upload latency; repeated fresh uploads and alternating order reversed that misleading result. CPU tensor copies were eliminated.

Android 15 hides private notification content during MediaProjection. Continuous mode uses Accessibility capture instead; no MediaProjection session was active during its accepted run. Fast sessions retain Android's screen-sharing behavior and explain it in the UI.

The capture helper initially assumed ADB could dismiss a secure keyguard and that an enabled service was necessarily bound. It now waits for the owner to unlock, verifies the bound screenshot capability and foreground recording service before fixtures, and stops in `finally`. A later helper click failed on an already-selected Battery Saver radio; the stored-data audit independently passed unlock recovery and the helper now handles that state. Raw failed attempts remain private; they are not counted as successful complete helper runs.

## Reviewed, not exercised

Reboot recovery: explicit RECEIVE_BOOT_COMPLETED permission and non-exported receiver are present in the built merged manifest. The receiver and Accessibility connection share the resume path; first unlock, persisted user request, all grants and screenshot capability are required. Stop clears the request. The foreground type is specialUse, not mediaProjection. The phone was **not rebooted**, as requested.

## Remaining scope

No 24-hour endurance guarantee, unplugged energy comparison, real MySQL/proxy deployment/concurrency test, or 16-KB-page device test. Continuous mode measures the screenshot/pipeline bottleneck; Fast sessions offer a different capture path with fresh consent after lock. OEM restrictions and force-stop can still require user intervention.

## Keyboard and foreground overlays — APK 1.1.2 (2026-10-09)

The Xperia XQ-DQ72/API 35 now captures keyboard-visible full-display frames with their host application label. Live uploads from the controlled keyboard fixture and ordinary Termux (`com.termux`) contained `keyboard_visible=true` and unchanged 1096×2560 source dimensions. Foreground notification drawer/quick-settings uploads were labeled `com.android.systemui` with `screen_kind=system_overlay`; they did not inherit an underlying app Activity. Pixels remain transient and the encoder/preprocessing path is unchanged. Termux:X11's disconnected launcher/preferences were also captured, but no connected X11 desktop session was exercised.

Android unit tests: 16 passed, covering IME focus, Termux/Termux:X11 host labels, drawer transitions, passive system bars, focused multiple-window selection, missing identities and explicit visible-app exclusions. Server suite: 51 passed on SQLite and 51 on isolated MariaDB, including old sample retry-hash compatibility and new context persistence. The compatible server update was deployed before installing the APK; no database migration was needed. Screen lock/off, protected screens, unknown identities and explicit exclusions remain pause conditions.
