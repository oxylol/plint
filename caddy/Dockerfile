# Caddy with the Cloudflare DNS module, so it can obtain a wildcard
# certificate for *.${DOMAIN} without anything being reachable from the internet,
# and replace-response, which slips the Plint theme into each app's pages.
FROM caddy:2-builder AS builder
RUN xcaddy build \
    --with github.com/caddy-dns/cloudflare \
    --with github.com/caddyserver/replace-response

FROM caddy:2
COPY --from=builder /usr/bin/caddy /usr/bin/caddy
