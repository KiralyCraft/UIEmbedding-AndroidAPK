#!/usr/bin/env bash
# Server-free model and RGBA preprocessing checks; never invokes outbox-purge tests.
set -euo pipefail
cd "$(dirname "$0")"
SERIAL="${1:?Usage: ./05_test_device_inference.sh ADB_SERIAL [REPORT_DIRECTORY]}"
REPORT="${2:-reports/device-inference-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$REPORT"
REPORT="$(cd "$REPORT" && pwd)"
command -v adb >/dev/null
command -v rg >/dev/null
[[ "$(adb -s "$SERIAL" get-state)" == device ]]
python3 tools/check_assets.py
export ANDROID_HOME="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}"
./01_build_apk.sh :app:assembleDebugAndroidTest --console=plain > "$REPORT/build.log" 2>&1
adb -s "$SERIAL" install -r android/app/build/outputs/apk/debug/app-debug.apk
adb -s "$SERIAL" install -r -t android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
START="$(adb -s "$SERIAL" shell "date '+%m-%d %H:%M:%S.000'" | tr -d '\r')"
adb -s "$SERIAL" shell am instrument -w -r \
    -e class ro.ubb.uicollector.GpuParityInstrumentedTest,ro.ubb.uicollector.PreprocessingInstrumentedTest \
    ro.ubb.uicollector.test/androidx.test.runner.AndroidJUnitRunner | tee "$REPORT/instrumentation.txt"
adb -s "$SERIAL" logcat -d -T "$START" -s tflite UICollectorInference > "$REPORT/gpu-logcat.txt"
# `am instrument` can exit zero even when a test fails; require JUnit's success summary.
if ! rg -q '^OK \(3 tests\)' "$REPORT/instrumentation.txt"; then
    echo "Device inference failed; see $REPORT/instrumentation.txt" >&2
    exit 1
fi
adb -s "$SERIAL" shell run-as ro.ubb.uicollector cat cache/gpu-inference-report.json > "$REPORT/gpu-inference-report.json"
adb -s "$SERIAL" shell run-as ro.ubb.uicollector cat cache/preprocessing-report.json > "$REPORT/preprocessing-report.json"
sha256sum android/app/build/outputs/apk/debug/app-debug.apk > "$REPORT/apk-sha256.txt"
printf '\nDevice inference passed. Reports: %s\n' "$REPORT"
