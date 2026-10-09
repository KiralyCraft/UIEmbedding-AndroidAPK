# Design invariants and limitations

## Capture and labels

Continuous mode requests one Accessibility full-display screenshot at a time, copies its borrowed hardware buffer into a reusable RGBA buffer, and closes/recycles the platform objects. The user explicitly starts/stops recording; lock/off pauses it. Fast mode owns one MediaProjection token and VirtualDisplay per user-granted session, acquires the latest ImageReader frame, and resizes the existing display on rotation. Both methods preserve complete-screen geometry. GPU delegate creation/invoke/close use one dedicated HandlerThread.

Calibration increases the complete pipeline rate from 0.25/s to at most 30/s, with 0.5/s steps from 2 through 4/s. It stops escalation at the first failed timing/freshness probe or Android screenshot interval rejection, then confirms the last passing rate for 30 seconds. Failed confirmation steps down to another passing rate and repeats confirmation. Missing frames count as unsuccessful attempts; cached pixels cannot satisfy calibration throughput. Android interval errors also establish a minimum request spacing from the rejected interval plus 10% (at least 20 ms) headroom. This is measured on the device rather than assuming a universal Accessibility ceiling. The API documents the rejection without promising a fixed interval: [Accessibility screenshot rate error](https://developer.android.com/reference/android/accessibilityservice/AccessibilityService#ERROR_TAKE_SCREENSHOT_INTERVAL_TIME_SHORT).

Maximum Detail uses the calibrated rate, Balanced uses half (at most 5/s), and Battery Saver uses one quarter (at most 1/s). Thermal/processing throttling never raises a mode above its effective ceiling, including ceilings below 0.25/s. Mode menus show their effective rates for the current capture configuration; obsolete saved calibration is not presented as current.

The capture pipeline is synchronous within the GPU thread but does not run on the UI thread. A finite RAM candidate frame may wait for a 300 ms stable foreground generation; a changed generation invalidates that frame. The post-inference label generation is checked before committing a sample. This reduces, but cannot eliminate, OS-event/render race mislabels. No API in this unprivileged implementation atomically binds a compositor frame to an Activity. Android permissions/OEM behavior can make exact activity identification unavailable.

The Accessibility service inspects visible-window types/focus/bounds and only the root packageName. It does not read node text, event text, descriptions or traverse children. Accessibility event.className is saved separately as window_class, never claimed to be an Activity name. UsageEvents supplies the activity when its foreground package agrees with Accessibility. Missing activity is null with activity_source=unknown. Accessibility supplies the screen package even when UsageEvents lags or still reports an underlying app; a mismatch makes Activity unknown rather than blocking the screen. The keyboard is included in its host app's full-display image, never used as the primary app label. Foreground system/Accessibility overlays (including the notification drawer) are labeled with their own package. In multiple application windows, the highest focused/active non-IME window supplies the primary label. A passive status/navigation bar does not replace an app label. `window_context` records screen kind, keyboard visibility and visible package IDs, without text or cropped imagery. Opt-out of any visible app still blocks that composite frame; locks, protected screens and unidentifiable windows remain paused.

Fast MediaProjection consent is single-session and cannot silently resume after revocation; Android 15 QPR1+ stops it on lock. Continuous Accessibility mode retains the explicit recording-request flag across lock and process/service reconnection. A BOOT_COMPLETED receiver and Accessibility onServiceConnected both check that flag, first-unlock state, screenshot capability and required grants before restoring the specialUse foreground service. Stop clears the flag. No direct-boot capture or reboot was tested. Force-stop suppresses background work until the app is launched again. Protected content remains subject to Android's screenshot restrictions.

The recorder is whole-display oriented. The request opts into the default display on API 34+, but the user/OEM may still restrict sharing. Select entire display. Single-app capture cannot follow all foreground apps correctly and is not the intended mode. Split-screen/PiP frames remain full-display composites; the primary label is the focused app and visible packages are retained as context. OEM window reporting can still leave a screen unidentified.

## Durable outbox state machine

1. Create a UUID run with revision=1 and immutable owner/model/device/package/start metadata.
2. Commit sample sequence N and increment local total_samples in one SQLite transaction.
3. Background uploader snapshots the run revision plus pending samples 0..128 at a time.
4. Server transaction locks/creates the run, validates metadata and sample identity, inserts unseen samples and commits.
5. A lost response causes an unchanged retry. Hash and composite-key checks make that idempotent.
6. Client verifies acknowledged sequence set and revision, deletes only that set, and marks only its SENT revision acknowledged.
7. A package boundary closes local metadata with revision=2 and final expected_samples. It does not mutate previously stored sample payloads.
8. The server marks complete when closure exists and received_samples==expected_samples, with all unique nonnegative indices below that length.
9. Client deletes closed local run metadata only when closure is acknowledged and no samples remain.

UUIDs isolate rebooted sessions; database uniqueness establishes idempotency, not just request-order assumptions. A concurrent old upload cannot mark a newly closed run fully acknowledged. Explicit queue deletion may abandon a server run; the server will not invent missing samples to complete it.

The effective delivery guarantee is: committed local records remain pending until acknowledged, provided app data/keys/storage survive and an authorized matching server is eventually reachable. It is not a distributed exactly-once network guarantee or indefinite storage guarantee. Database/server rollback after acknowledgment requires backups/replication outside this application; the client is permitted to delete the acknowledged copy.

## Authentication and deployment

There is no public enrollment endpoint. An administrator creates users and registers the exact exported model. Passwords use Argon2id. Server tokens are random, user-scoped and stored by hash with expiry. Phone credentials and queue payloads use AES-GCM under a non-exported Android Keystore key; authenticated contexts distinguish identity, run, report and sample payloads. Plain queue columns include UUIDs, owner hash and counters, not package/activity strings or vectors.

HTTPS is mandatory on the phone. No permissive certificate verifier or hostname verifier is used. The backend intentionally serves HTTP on a private interface behind the TLS proxy. Only explicitly trusted proxy addresses may supply forwarded information. A reverse proxy may need additional deployment-specific authentication/rate limits, backup policy and operational monitoring.

Embeddings are not anonymous: their structure may reveal app usage and screen semantics. Server operators can read stored vectors/metadata. User consent, minimization, appropriate retention and access policy remain deployment responsibilities. No legal-compliance certification or formal security audit is claimed.

## Model compatibility and limits

The exporter creates the trained full LiteRT encoder plus a compatibility backbone/binary head. Deployment identities hash the assets and preprocessing version; regenerated assets may get a new model_id even from the same weights. Preserve a deployed export for a collection campaign.

FP32 preserves the trained LayerNorm epsilon and exact GELU in the preferred full OpenCL model. The compatibility CPU head uses double intermediates and an erf approximation. Export, GPU and preprocessing parity are separate gates; head-only parity is insufficient.

The exporter is F6-specific and intentionally rejects arbitrary architecture changes. It uses only ordinary/depthwise convolution, fused affine/activation, optional layer scale, residual addition and fixed global pooling. Unexpected grouped convolutions, traced functions, dynamically shaped operators or unsupported activations stop export. A GPU-compatible operator list is necessary but not sufficient for full delegation on every vendor driver. Review delegate logs/profiling on the target phone; lack of deliberate CPU fallback is not proof every node is delegated.

Preprocessing compares complete repeated CPU and native OpenCL pipelines, including fresh ingestion/upload and readback. Trials alternate order and discard warmups; the first allocation/upload is not reused as a steady-state cost. OpenCL must pass RGB/embedding parity and improve p95 by 5%. It currently uses a separate context from LiteRT, so the GPU preprocessing output returns through a host buffer. The CPU path writes normalized data directly into the model's direct buffer. Zero-copy interop, FP16 or quantization require separate parity/performance validation.

## Dataset semantics

A sample records its capture/acquisition wall time, elapsedRealtimeNanos, Image.timestamp, source dimensions, rotation, target rate, preprocessing/GPU/head timings, thermal level, activity/source, window class and label age. Image.timestamp is retained as a raw source clock; no undocumented cross-OEM clock equivalence is assumed. Label age is measured at processing-time snapshot and can exceed frame hold time.

No sample is manufactured to fill a regular grid. Fast sessions detect freshness by ImageReader timestamps, so identical-content new buffers may be retained. Continuous screenshots suppress consecutive identical RGBA fingerprints and record monotonic acquisition time rather than claiming an ImageReader clock. Each sample identifies its capture method. Use timestamps to weight dwell or resample explicitly, preserving original sequences.

Activities in an app are contextual labels, not run IDs. A run can contain multiple activities of the same package. Opening the keyboard does not change the host package. Foreground overlays change the primary package, so entering/leaving a notification drawer produces explicit app/System UI run transitions; an exclusion pause still ends the current fragment. For few-shot evaluation, split by application/device/session as needed rather than randomly leaking neighboring frames across train/test.

The server accepts optional `window_context` on samples. When absent/null it is omitted from canonical sample metadata so historical samples retain their exact retry hashes. Deploy this compatible server update before APK 1.1.2. No database schema migration is required.
