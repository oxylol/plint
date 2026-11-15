#!/usr/bin/env bash
# Build the TV home screen into data/tv/plint-home.apk.   apps/tv-home/build.sh [--install]
# No Gradle: Google's build tools (aapt2, d8, zipalign, apksigner) and javac, run in a JDK container.
# The tools (about 1 GB, under Google's SDK licence) are fetched once and kept on the array. The key
# that signs the app is made on the first build and stays in data/tv: an update must carry the same one.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLINT_DIR="${PLINT_DIR:-/opt/plint}"
out="$PLINT_DIR/data/tv"
sdk=/mnt/storage/plint/tv-home/sdk
mkdir -p "$sdk"

docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$here:/src:ro" -v "$out:/out" -v "$sdk:/sdk" \
  eclipse-temurin:17-jdk-jammy bash -euc '
    bt=/sdk/build-tools/34.0.0; android=/sdk/platforms/android-34/android.jar
    if [ ! -x $bt/aapt2 ] || [ ! -f $android ]; then
      cd /tmp && curl -fsSL -o tools.zip https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
      jar xf tools.zip && chmod +x cmdline-tools/bin/*
      yes | cmdline-tools/bin/sdkmanager --sdk_root=/sdk "build-tools;34.0.0" "platforms;android-34" >/dev/null
    fi
    cd "$(mktemp -d)"
    $bt/aapt2 link -o app.apk -I $android --manifest /src/AndroidManifest.xml -A /src/assets
    javac -source 8 -target 8 -Xlint:-options -cp $android -d classes $(find /src/src -name "*.java")
    $bt/d8 --release --lib $android --min-api 26 --output . $(find classes -name "*.class")
    $bt/aapt add app.apk classes.dex >/dev/null
    $bt/zipalign -f 4 app.apk aligned.apk
    [ -f /out/home.keystore ] || keytool -genkeypair -keystore /out/home.keystore -storepass android -keypass android \
      -alias plint -keyalg RSA -keysize 2048 -validity 10000 -dname CN=Plint 2>/dev/null
    $bt/apksigner sign --ks /out/home.keystore --ks-pass pass:android --out /out/plint-home.apk aligned.apk
    rm -f /out/plint-home.apk.idsig
  '
ls -l "$out/plint-home.apk"
[[ "${1:-}" != --install ]] || "$PLINT_DIR/bin/plint" tv install "$out/plint-home.apk"
