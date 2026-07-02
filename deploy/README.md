# Deploy guide — Hetzner CX32 + Streamlit Cloud

End-to-end provisioning of the Ocean Quest Fish Economy backend. The dashboard
lives on Streamlit Community Cloud (free); the dumper daemon and Postgres
live on a single Hetzner box.

---

## 1. Provision the box

1. Sign up at hetzner.com.
2. Create a **CX32** server (~€7/mo, 8 GB RAM, 80 GB SSD), Ubuntu 24.04, EU
   datacenter (Falkenstein/Helsinki). Add your SSH key during creation.
3. SSH in: `ssh root@<IP>`.

## 2. Clone the repo

```bash
mkdir -p /opt/projects
cd /opt/projects
git clone https://github.com/MetisPrometheus/ocean-quest-fish-economy.git ocean-quest
cd ocean-quest
```

## 3. Run setup.sh

```bash
sudo bash deploy/setup.sh ocean-quest
```

This is idempotent and takes ~3 min. It:

- Installs Postgres 16, Caddy, Python venv
- Creates the `ocean-quest` Linux user, `ocean_quest` database, `ocean_quest_rw`
  and `ocean_quest_ro` roles (random passwords, written to `/etc/ocean-quest/env`)
- Configures Postgres to listen externally with SSL forced (self-signed cert)
- Installs Python deps, runs `db.py init`
- Drops in `ocean-quest-fetch.service` (auto-restart daemon) and
  `ocean-quest-discover.timer` (daily DataStore rescan)
- Opens UFW: 22 (SSH), 80, 443 (Caddy, future), 5432 (Postgres external)

The script prints a **read-only Postgres connection string** at the end —
**copy it now**. It looks like:

```
postgresql://ocean_quest_ro:HEX_PASSWORD@1.2.3.4:5432/ocean_quest?sslmode=require
```

## 4. Set the Roblox secrets

`setup.sh` left placeholders in the env file:

```bash
sudo nano /etc/ocean-quest/env
# Set ROBLOX_API_KEY and ROBLOX_UNIVERSE_ID
sudo systemctl restart ocean-quest-fetch
```

## 5. Verify the box

```bash
# Daemon running and fetching?
sudo journalctl -u ocean-quest-fetch -f

# Discovery timer scheduled?
systemctl list-timers ocean-quest-discover

# DB has rows?
sudo -u postgres psql -d ocean_quest -c "SELECT COALESCE(status,'unfetched'), COUNT(*) FROM players GROUP BY 1;"

# External connection works?  (from your laptop, requires `psql` client)
psql 'postgresql://ocean_quest_ro:...@HETZNER_IP:5432/ocean_quest?sslmode=require' -c 'SELECT COUNT(*) FROM players;'
```

## 6. Deploy the dashboard to Streamlit Cloud

1. Push the repo to GitHub (the `main` branch).
2. Go to **share.streamlit.io** → **Create app**.
3. Pick the repo, branch `main`, main file `dashboard.py`.
4. Click **Advanced settings → Secrets** and paste:

   ```toml
   DATABASE_URL = "postgresql://ocean_quest_ro:HEX_PASSWORD@1.2.3.4:5432/ocean_quest?sslmode=require"
   ```

5. Click **Deploy**. The dashboard URL will be `https://<repo-name>.streamlit.app`.

Future pushes to `main` auto-redeploy.

---

## Troubleshooting

| Problem | Check |
|---|---|
| Daemon not making requests | `sudo journalctl -u ocean-quest-fetch -f` — look for "rate limiter 429" or "DATABASE_URL" errors |
| 429-stuck loop | Should auto-recover: look for `[cooldown] sleeping 10m` log line. If not, drop `--rate` in the systemd unit |
| Dashboard can't connect | From the box, run `sudo -u postgres psql -d ocean_quest -c '\du'` to verify the `ocean_quest_ro` role exists. Check `/etc/postgresql/16/main/pg_hba.conf` has the `hostssl` line |
| Streamlit Cloud TLS error | Ensure `sslmode=require` is in the connection string. Postgres is using a self-signed cert; libpq accepts it with `sslmode=require` (no cert verification) |
| Dashboard cached stale data | Cache TTL is 5 min. Force-refresh the Streamlit page or wait |

---

## Adding a second project to this same box

Clone it under `/opt/projects/<name>` and run `setup.sh <name>`. You get:

- Linux user `<name>`
- DB `<name_underscored>`, roles `<name>_rw` and `<name>_ro`
- systemd units namespaced by project name
- Same Postgres instance, same Caddy

Costs zero extra — you're already paying for the box.

---

## Hardening to-dos (optional, later)

- **Tailscale** instead of public Postgres: install Tailscale on the box and on
  whatever machine connects to it. Drop the `ufw allow 5432/tcp` rule.
  Streamlit Cloud sadly can't be on a Tailnet, so this only works if you migrate
  the dashboard onto the box too (`oqfe-streamlit.service` + Caddy reverse-proxy).
- **Backups**: `pg_dump` to Backblaze B2 every night. ~5 lines of cron.
- **Monitoring**: install Netdata (`bash <(curl -Ss …)`) for a free per-host dashboard.
- **Auto-deploy on git push**: a tiny webhook listener that runs
  `cd /opt/projects/ocean-quest && git pull && sudo systemctl restart ocean-quest-fetch`.
