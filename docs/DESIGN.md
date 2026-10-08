# Design invariants and limitations

## Capture and labels

The process owns one MediaProjection token and one VirtualDisplay per user-granted session. GPU delegate creation/invoke/close all use one dedicated HandlerThread. ImageReader acquisition uses acquireLatestImage and closes acquired Images in finally blocks. Resize callbacks replace the surface/ImageReader and resize the existing display, not reuse the token to create a second display.

The capture pipeline is synchronous within the GPU thread but does not run on the UI thread. A finite RAM candidate frame may wait for a 300 ms stable foreground generation; a changed generation invalidates that frame. The post-inference label generation is checked before committing a sample. This reduces, but cannot eliminate, OS-event/render race mislabels. No API in this unprivileged implementation atomically binds a compositor frame to an Activity. Android permissions/OEM behavior can make exact activity identification unavailable.

The Accessibility service inspects visible-window types/focus/bounds and only the root packageName. It does not read node text, event text, descriptions or traverse children. Accessibility event.className is saved separately as window_class, never claimed to be an Activity name. UsageEvents supplies the activity when its foreground package agrees with Accessibility. Missing activity is null with activity_source=unknown. A disagreement, screen lock, disconnected service, keyboard, system overlay or multiple application windows pauses collection.

Android 14+ user consent is single-session; this app cannot silently resume projection after a process restart or screen-lock stop. Android 15 QPR1 and later stop media projection on lock; the app also deliberately stops on SCREEN_OFF on older versions. Restarting capture requires the user's action and fresh consent. Force-stop suppresses background work until the app is launched again. Notifications, permissions and restrictions are not bypassed. DRM/FLAG_SECURE content may be blank; a GPU embedding of a blank image is not meaningful evidence about protected content. No attempt to defeat secure capture is included.

The recorder is whole-display oriented. The request opts into the default display on API 34+, but the user/OEM may still restrict sharing. Select entire display. Single-app capture cannot follow all foreground apps correctly and is not the intended mode. Split-screen/PiP/overlay-heavy cases are conservative, not guaranteed supported.

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

The F6 encoder cannot be loaded as a generic .pts file by this Android runtime. The export stage creates the trained LiteRT backbone and binary original head. Embedding identities hash the exported assets and preprocessing version; regenerated assets may get a new model_id even from the same weights. Preserve a deployed export for a collection campaign instead of repeatedly changing the converter environment.

FP32 is used initially. The CPU head computes linear algebra/LayerNorm in double intermediates and an erf approximation for GELU, then emits normalized float32. Its trained-weight JVM parity is measured in reports/kotlin_head_parity.json. That measurement does not establish preprocessing, backbone conversion or GPU numerical parity. Those are separate gates.

The exporter is F6-specific and intentionally rejects arbitrary architecture changes. It uses only ordinary/depthwise convolution, fused affine/activation, optional layer scale, residual addition and fixed global pooling. Unexpected grouped convolutions, traced functions, dynamically shaped operators or unsupported activations stop export. A GPU-compatible operator list is necessary but not sufficient for full delegation on every vendor driver. Review delegate logs/profiling on the target phone; lack of deliberate CPU fallback is not proof every node is delegated.

The first implementation prioritizes parity and durability over zero-copy throughput. Frames cross ImageReader to OpenCV CPU buffers, normalized input is copied into the GPU interpreter, pooled 960-float features return for the CPU head, and samples use durable SQLite commits. Benchmark the whole pipeline; do not infer FPS from a desktop F6 forward pass. GPU textures/AHardwareBuffer input, FP16 or quantization would require a separate measured compatibility and parity revision.

## Dataset semantics

A sample records its capture/acquisition wall time, elapsedRealtimeNanos, Image.timestamp, source dimensions, rotation, target rate, preprocessing/GPU/head timings, thermal level, activity/source, window class and label age. Image.timestamp is retained as a raw source clock; no undocumented cross-OEM clock equivalence is assumed. Label age is measured at processing-time snapshot and can exceed frame hold time.

No sample is manufactured simply to fill a regular grid. An unchanged display may produce no new buffer; a display may also produce repeated-content new buffers. This code detects freshness by buffer timestamps, not a content hash. Thus adjacent embeddings may be identical and both still correspond to legitimately supplied frames. Use timestamps to weight dwell, interpolate or resample explicitly in later work, and keep original sequences as provenance.

Activities in an app are contextual labels, not run IDs. A run can contain multiple activities of the same package. A return after keyboard/overlay/exclusion pause is a new fragment. For few-shot evaluation, split by application/device/session as needed rather than randomly leaking neighboring frames across train/test.
