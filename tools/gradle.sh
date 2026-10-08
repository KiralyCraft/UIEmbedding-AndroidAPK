#!/usr/bin/env bash
# Transparent Gradle bootstrap; no unverified Gradle wrapper JAR is bundled.
set -euo pipefail
VERSION=8.11.1
BASE="${GRADLE_USER_HOME:-$HOME/.gradle}/uicollector-bootstrap"
INSTALL="$BASE/gradle-$VERSION"
if [[ ! -x "$INSTALL/bin/gradle" ]]; then
    command -v curl >/dev/null
    command -v unzip >/dev/null
    mkdir -p "$BASE"
    TEMP="$(mktemp -d "$BASE/download.XXXXXX")"
    trap 'rm -rf "$TEMP"' EXIT
    curl --fail --location --proto '=https' --tlsv1.2 "https://services.gradle.org/distributions/gradle-$VERSION-bin.zip" -o "$TEMP/gradle.zip"
    curl --fail --location --proto '=https' --tlsv1.2 "https://services.gradle.org/distributions/gradle-$VERSION-bin.zip.sha256" -o "$TEMP/gradle.sha256"
    HASH="$(tr -d '\r\n ' < "$TEMP/gradle.sha256")"
    [[ "$HASH" =~ ^[0-9a-f]{64}$ ]]
    printf '%s  %s\n' "$HASH" "$TEMP/gradle.zip" | sha256sum --check -
    unzip -q "$TEMP/gradle.zip" -d "$BASE"
fi
exec "$INSTALL/bin/gradle" "$@"
