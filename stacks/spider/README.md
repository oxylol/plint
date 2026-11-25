# SpiderFoot · spider.plint.net

OSINT scans: give it a domain, IP, email, name, username, phone number or Bitcoin
address and its 233 modules go looking. Ready-made (github.com/smicallef/spiderfoot),
built from the repo's own Dockerfile at the last commit (2023-11-05; v4.0 is from
2022). No official image, no login (LAN and tailnet only, like the rest), its own look.

- Scans, settings and API keys: its SQLite database in `data/spiderfoot` (backed up).
- Rebuild: `plint up spider` (it builds from the pinned commit). Logs: `plint logs spiderfoot`.
- DNS goes to 1.1.1.1 / 9.9.9.9, not AdGuard, so a scan sees real answers instead of
  AdGuard's 0.0.0.0 for blocklisted domains.

## Running a scan

New Scan → target. A name or username goes **in straight double quotes**, or it says
"invalid target": `"Filip Munkeby"` (a space makes it a name), `"filipm"` (a username).
Curly quotes from a phone keyboard don't count. A phone number starts with `+` and the
country code (`+46701234567`). Domains, IPs and emails go in as they are.

Then **By Use Case**:
- **Passive**: only third-party sources, so the target never sees you. The default for
  looking into someone else.
- **Footprint / Investigate / All**: also touch the target directly (crawl its site,
  grab banners, try ports). Noisy, and **All** takes hours.

Tested 2026-11-25 against spiderfoot.net: DNS, co-hosted sites (HackerTarget, Robtex),
certificates, open ports and urlscan.io all returned data. crt.sh often times out.

## API keys

81 modules want a key; the other 152 run without. Keys go in **Settings → the module →
API Key** and are kept in its database. Free tiers with small quotas, endpoints answering
on 2026-11-25:

| key | gives |
|---|---|
| VirusTotal | domain/IP reputation, subdomains, malware links |
| AlienVault OTX | threat intel, passive DNS, URLs |
| Shodan | open ports and services on IPs |
| SecurityTrails | subdomains, DNS history |
| Hunter.io | email addresses at a domain |
| AbuseIPDB | abuse reports on IPs |
| IntelligenceX | leaks, pastes, darknet mentions |
| LeakIX | exposed services and leaks |
| FullHunt | attack surface of a domain |
| IPInfo | IP owner and location |
| GreyNoise Community | whether an IP is a known internet scanner |
| Pulsedive · MetaDefender · WiGLE | reputation · file/IP scans · Wi-Fi networks by SSID |

Dead or not worth it: Bing (API retired), BitcoinAbuse (shut down), RiskIQ, ThreatJammer,
Clearbit, Trashpanda, SpyOnWeb (their APIs no longer resolve); Censys (the v2 API this
version speaks was shut down on 2026-09-30); HaveIBeenPwned, Dehashed, C99, Seon, Whoxy and
others are paid only. The four Tor modules (ahmia, onioncity, onionsearchengine, torch)
need a Tor proxy, which isn't set up: it would send all of SpiderFoot's traffic through Tor.
