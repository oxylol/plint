# Caddy with the Cloudflare DNS module, so it can obtain a wildcard
# certificate for *.${DOMAIN} without anything being reachable from the internet.
FROM caddy:2-builder AS builder
RUN xcaddy build --with github.com/caddy-dns/cloudflare

FROM caddy:2
COPY --from=builder /usr/bin/caddy /usr/bin/caddy
