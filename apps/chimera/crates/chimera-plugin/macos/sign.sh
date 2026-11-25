#!/usr/bin/env bash
# Developer ID signing and notarisation on CI. Without the secrets (a fork's pull request, or a Mac
# with none set) each step says so and does nothing, and build.sh signs ad hoc as before.
#
#   sign.sh keychain    import the certificate into a keychain of its own, and put the identity
#                       in $GITHUB_ENV as CHIMERA_SIGN_IDENTITY, which build.sh signs with
#   sign.sh notarize    send target/bundled's bundles to Apple, wait, and staple each one
#
# Secrets: MACOS_CERT_P12 (base64 of the .p12), MACOS_CERT_PASSWORD, NOTARY_KEY_P8 (the App Store
# Connect API key's text), NOTARY_KEY_ID, NOTARY_ISSUER_ID.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
tmp=${RUNNER_TEMP:-$(mktemp -d)}

case "${1:-}" in
keychain)
  if [ -z "${MACOS_CERT_P12:-}" ]; then
    echo "no MACOS_CERT_P12: the bundles are signed ad hoc"
    exit 0
  fi
  kc="$tmp/signing.keychain-db"
  pw=$(openssl rand -hex 16)
  security create-keychain -p "$pw" "$kc"
  security set-keychain-settings -lut 21600 "$kc"
  security unlock-keychain -p "$pw" "$kc"
  printf '%s' "$MACOS_CERT_P12" | tr -d '[:space:]' | base64 -d > "$tmp/cert.p12"
  security import "$tmp/cert.p12" -k "$kc" -P "$MACOS_CERT_PASSWORD" -T /usr/bin/codesign
  rm -f "$tmp/cert.p12"
  # the intermediate, in case the .p12 holds only the leaf
  curl -fsSL https://www.apple.com/certificateauthority/DeveloperIDG2CA.cer -o "$tmp/g2.cer"
  security import "$tmp/g2.cer" -k "$kc" || true
  # let codesign use the key without a prompt
  security set-key-partition-list -S apple-tool:,apple: -s -k "$pw" "$kc" > /dev/null
  # shellcheck disable=SC2046 # the existing list, one path per word
  security list-keychains -d user -s "$kc" $(security list-keychains -d user | tr -d '"')
  security find-identity -v -p codesigning "$kc"
  identity=$(security find-identity -v -p codesigning "$kc" |
    sed -n 's/.*"\(Developer ID Application: .*\)"$/\1/p' | head -1)
  if [ -z "$identity" ]; then
    echo "the .p12 holds no Developer ID Application identity with its private key"
    exit 1
  fi
  echo "CHIMERA_SIGN_IDENTITY=$identity" >> "${GITHUB_ENV:-/dev/null}"
  echo "signing as: $identity"
  ;;

notarize)
  if [ -z "${CHIMERA_SIGN_IDENTITY:-}" ] || [ -z "${NOTARY_KEY_P8:-}" ]; then
    echo "not signed with a Developer ID, or no NOTARY_KEY_P8: not notarised"
    exit 0
  fi
  bundles=(target/bundled/CHIMERA.clap target/bundled/CHIMERA.vst3 target/bundled/CHIMERA.component)
  key="$tmp/notary.p8"
  trap 'rm -f "$key"' EXIT
  printf '%s\n' "$NOTARY_KEY_P8" > "$key"
  auth=(--key "$key" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER_ID")

  rm -rf "$tmp/notarize" && mkdir -p "$tmp/notarize/CHIMERA"
  cp -R "${bundles[@]}" "$tmp/notarize/CHIMERA/"
  ditto -c -k --keepParent "$tmp/notarize/CHIMERA" "$tmp/notarize/CHIMERA.zip"

  json="$tmp/notarize/result.json"
  xcrun notarytool submit "$tmp/notarize/CHIMERA.zip" "${auth[@]}" \
    --wait --timeout 30m --output-format json > "$json" || true
  cat "$json"; echo
  status=$(plutil -extract status raw -o - "$json" 2>/dev/null || echo "no result")
  if [ "$status" != Accepted ]; then
    # Apple's log names each file it rejected and why
    xcrun notarytool log "$(plutil -extract id raw -o - "$json")" "${auth[@]}" || true
    echo "notarisation: $status"
    exit 1
  fi

  for b in "${bundles[@]}"; do
    xcrun stapler staple "$b"
    xcrun stapler validate "$b"
  done
  codesign -dvv "${bundles[0]}" 2>&1 | grep -E '^(Authority|TeamIdentifier|Timestamp|Runtime)'
  ;;

*)
  echo "usage: sign.sh keychain | notarize" >&2
  exit 2
  ;;
esac
