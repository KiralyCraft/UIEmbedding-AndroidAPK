#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 tools/check_assets.py
if [[ -z "${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}" && ! -f android/local.properties ]]; then
    echo 'Set ANDROID_HOME to your Android SDK (platform 35, build-tools 35.0.0), or create android/local.properties.' >&2
    exit 1
fi
./tools/gradle.sh -p android :app:assembleDebug :app:testDebugUnitTest "$@"
printf '\nAPK: %s/android/app/build/outputs/apk/debug/app-debug.apk\n' "$PWD"
