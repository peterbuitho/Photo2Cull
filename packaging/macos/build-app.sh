#!/usr/bin/env bash
# Stage a local Photo2Cull.app so it can be launched from Finder (double
# click, Spotlight, Dock) instead of opening in Terminal. A bare Unix
# executable like target/release/photo2cull isn't an application bundle, so
# Finder falls back to running it in Terminal -- wrapping it in a proper
# .app, as CI does for releases (.github/workflows/build.yml), fixes that.
#
# Usage: packaging/macos/build-app.sh
# Builds the release binary if needed, then stages ./Photo2Cull.app in the
# repo root (gitignored). Run it again after pulling changes to refresh it.

set -euo pipefail

cd "$(dirname "$0")/../.."

VERSION=$(grep '^version' Cargo.toml | head -1 | sed -E 's/version = "(.*)"/\1/')
APP="Photo2Cull.app"
BIN="target/release/photo2cull"

cargo build --release

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp "$BIN" "$APP/Contents/MacOS/photo2cull"
chmod +x "$APP/Contents/MacOS/photo2cull"
sed "s/__VERSION__/$VERSION/g" packaging/macos/Info.plist >"$APP/Contents/Info.plist"

# On a filesystem without native xattr support (e.g. an exFAT/NTFS external
# drive), macOS silently stores the binary's xattrs in a "._*" AppleDouble
# companion file next to it -- `cp` carries that over. `codesign --deep`
# below then trips trying to sign that companion as if it were part of the
# bundle, so strip any before signing.
find "$APP" -name '._*' -delete

# Ad-hoc sign (no Apple Developer account/certificate needed). This isn't
# notarization -- Gatekeeper still shows an "unidentified developer"
# warning on first launch (right-click > Open to bypass) -- but it avoids
# the misleading "is damaged" rejection unsigned binaries get on newer
# macOS.
codesign --force --deep --sign - "$APP"
codesign --verify --verbose "$APP"

echo "Staged $APP -- open it from Finder, or run: open $APP"
