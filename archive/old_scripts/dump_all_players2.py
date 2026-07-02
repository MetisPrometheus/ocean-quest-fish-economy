"""
Dump ALL players' DataStore2 combined DATA from Roblox Open Cloud v1 API.

Sequential flow: discover all player IDs → fetch usernames → dump in parallel.
The counter always shows a stable total: [done/total].

Saves to: player_dumps/{userId}_{username}/

Usage:
    python dump_all_players2.py
    python dump_all_players2.py --max 50
    python dump_all_players2.py --skip-existing
    python dump_all_players2.py --workers 20      # parallel threads (default 20)
    python dump_all_players2.py --rate 120        # total req/s across all threads (default 120)
    python dump_all_players2.py --delay 0.05      # legacy: sets rate = 1/delay
"""

import os, sys, json, time, re, argparse, threading, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── .env loading ──────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

API_KEY = os.environ.get("ROBLOX_API_KEY", "YOUR_API_KEY_HERE")
UNIVERSE_ID = os.environ.get("ROBLOX_UNIVERSE_ID", "YOUR_UNIVERSE_ID_HERE")

BASE = f"https://apis.roblox.com/datastores/v1/universes/{UNIVERSE_ID}/standard-datastores"
ORDERED_BASE = f"https://apis.roblox.com/ordered-datastores/v1/universes/{UNIVERSE_ID}/orderedDataStores"

MAX_RETRIES = 3


# ── Shared adaptive token-bucket rate limiter ─────────────────
class _TokenBucket:
    """All threads share one bucket. Rate backs off on 429, slowly recovers."""

    def __init__(self, rate: float):
        self._rate = float(rate)
        self._tokens = float(rate)
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self._consecutive_ok = 0
        self._last_throttle = 0.0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)

    def on_success(self) -> None:
        with self._lock:
            self._consecutive_ok += 1
            if self._consecutive_ok % 50 == 0:
                self._rate = min(self._rate * 1.1, _INITIAL_RATE)

    def on_throttle(self) -> None:
        now = time.monotonic()
        with self._lock:
            # One adjustment per 2-second window — prevents 20 threads all printing at once
            if now - self._last_throttle < 2.0:
                return
            self._last_throttle = now
            new_rate = max(5.0, self._rate * 0.6)
            if new_rate < self._rate:
                self._rate = new_rate
                self._consecutive_ok = 0
                print(f"\n  [rate limiter] 429 — throttled to {self._rate:.0f} req/s", flush=True)


_INITIAL_RATE = 120.0
_bucket: _TokenBucket | None = None

# ── Thread-safe counters ──────────────────────────────────────
_lock = threading.Lock()
_request_count = 0
_success = 0
_empty = 0
_failed = 0
_done = 0
_empty_ids = []
_start_time = 0.0


def inc_requests():
    global _request_count
    with _lock:
        _request_count += 1


def inc_result(kind, uid=None):
    global _success, _empty, _failed, _done
    with _lock:
        if kind == "ok":
            _success += 1
        elif kind == "empty":
            _empty += 1
            if uid is not None:
                _empty_ids.append(uid)
        else:
            _failed += 1
        _done += 1


# ── API helpers ───────────────────────────────────────────────

def api_get(url):
    """GET with shared token-bucket rate limiting, adaptive 429 handling, and retries."""
    for attempt in range(MAX_RETRIES):
        _bucket.acquire()
        inc_requests()
        req = urllib.request.Request(url, headers={"x-api-key": API_KEY})
        try:
            with urllib.request.urlopen(req) as resp:
                _bucket.on_success()
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                _bucket.on_throttle()
                backoff = (2 ** attempt) * (0.5 + (time.monotonic() % 1.0))
                time.sleep(backoff)
                continue
            elif e.code == 404:
                _bucket.on_success()
                return None
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
    return None


def get_usernames_bulk(user_ids):
    """Batch-fetch usernames for up to 100 IDs at a time."""
    results = {}
    for i in range(0, len(user_ids), 100):
        batch = user_ids[i:i + 100]
        time.sleep(0.1)
        body = json.dumps({"userIds": batch, "excludeBannedUsers": False}).encode()
        req = urllib.request.Request(
            "https://users.roblox.com/v1/users",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as r:
                data = json.loads(r.read().decode())
                for u in data.get("data", []):
                    results[u["id"]] = (u.get("name"), u.get("displayName"))
        except Exception:
            pass
    return results


# ── DataStore helpers ─────────────────────────────────────────

def list_all_keys(ds_name):
    encoded = urllib.request.quote(ds_name, safe="")
    keys, cursor = [], ""
    while True:
        url = f"{BASE}/datastore/entries?datastoreName={encoded}&limit=100"
        if cursor:
            url += f"&cursor={cursor}"
        data = api_get(url)
        if not data:
            break
        keys.extend([k["key"] for k in data.get("keys", [])])
        cursor = data.get("nextPageCursor", "")
        if not cursor:
            break
    return keys


def get_entry(ds_name, key):
    encoded = urllib.request.quote(ds_name, safe="")
    return api_get(f"{BASE}/datastore/entries/entry?datastoreName={encoded}&entryKey={key}")


def get_latest_key_ordered(ds_name):
    encoded = urllib.request.quote(ds_name, safe="")
    data = api_get(f"{ORDERED_BASE}/{encoded}/scopes/global/entries?max_page_size=1&order_by=desc")
    if data and "entries" in data and data["entries"]:
        return int(data["entries"][0].get("value", 0))
    return None


def fetch_latest(ds_name):
    latest = get_latest_key_ordered(ds_name)
    if latest:
        return get_entry(ds_name, str(latest)), latest
    keys = list_all_keys(ds_name)
    int_keys = [int(k) for k in keys if k.isdigit()]
    if int_keys:
        latest = max(int_keys)
        return get_entry(ds_name, str(latest)), latest
    return None, None


# ── Discovery ─────────────────────────────────────────────────

def discover_player_ids(existing: set[int], max_players: int) -> list[int]:
    """Page through all DataStores, return sorted list of matching DATA/{userId} IDs."""
    player_ids = []
    cursor = ""
    while True:
        url = f"{BASE}?limit=100"
        if cursor:
            url += f"&cursor={cursor}"
        data = api_get(url)
        if not data:
            break
        for ds in data.get("datastores", []):
            m = re.match(r"^DATA/(\d{5,})$", ds["name"])
            if not m:
                continue
            uid = int(m.group(1))
            if uid in existing:
                continue
            player_ids.append(uid)
            if max_players > 0 and len(player_ids) >= max_players:
                return player_ids
        cursor = data.get("nextPageCursor", "")
        if not cursor:
            break
    return player_ids


# ── Per-player dump ───────────────────────────────────────────

def dump_player(user_id, username=None, display_name=None):
    """Dump a single player's combined DATA. Returns True if data found."""
    folder_name = f"{user_id}_{username}" if username else str(user_id)
    folder = Path("player_dumps") / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    ds = f"DATA/{user_id}"
    combined, key = fetch_latest(ds)

    if not combined or not isinstance(combined, dict):
        for file in folder.iterdir():
            file.unlink()
        folder.rmdir()
        return False

    with open(folder / "DATA.json", "w") as f:
        json.dump(combined, f, indent=2)

    for store_name, store_data in combined.items():
        with open(folder / f"{store_name}.json", "w") as f:
            json.dump(store_data, f, indent=2)

    info = {
        "userId": int(user_id),
        "username": username,
        "displayName": display_name,
        "combinedSaveKey": key,
        "storesInCombined": sorted(combined.keys()),
        "dumpedAt": datetime.utcnow().isoformat() + "Z",
    }
    with open(folder / "_player.json", "w") as f:
        json.dump(info, f, indent=2)

    return True


def worker(uid, username, display_name, total):
    label = f"@{username}" if username else f"id:{uid}"
    try:
        found = dump_player(uid, username, display_name)
        if found:
            inc_result("ok")
            status = "OK"
        else:
            inc_result("empty", uid)
            status = "empty"
    except Exception as e:
        inc_result("fail", uid)
        status = f"ERROR: {e}"

    with _lock:
        elapsed = time.time() - _start_time
        rps = _request_count / elapsed if elapsed > 0 else 0
        print(f"  [{_done}/{total}] {label:30s} {status}  ({rps:.0f} req/s)", flush=True)


# ── Main ──────────────────────────────────────────────────────

def main():
    global _INITIAL_RATE, _bucket, _start_time

    parser = argparse.ArgumentParser(description="Dump all players' combined DATA")
    parser.add_argument("--max", type=int, default=0, help="Max players to dump (0 = all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already-dumped players")
    parser.add_argument("--workers", type=int, default=20, help="Parallel worker threads (default 20)")
    parser.add_argument("--rate", type=float, default=120.0, help="Total requests/sec across all threads (default 120)")
    parser.add_argument("--delay", type=float, default=None, help="Legacy: per-request delay — sets rate=1/delay")
    args = parser.parse_args()

    if args.delay is not None:
        args.rate = 1.0 / args.delay

    _INITIAL_RATE = args.rate
    _bucket = _TokenBucket(args.rate)

    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Set ROBLOX_API_KEY in .env"); sys.exit(1)
    if UNIVERSE_ID == "YOUR_UNIVERSE_ID_HERE":
        print("ERROR: Set ROBLOX_UNIVERSE_ID in .env"); sys.exit(1)

    print("=" * 60)
    print("DataStore2 Player Dump (combined DATA only)")
    print(f"Universe: {UNIVERSE_ID}")
    print(f"Workers:  {args.workers}")
    print(f"Rate:     {args.rate:.0f} req/s (shared across all threads)")
    print(f"Started:  {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    # ── Step 1: Build skip-set ────────────────────────────────
    existing: set[int] = set()
    if args.skip_existing:
        dumps_dir = Path("player_dumps")
        if dumps_dir.exists():
            for d in dumps_dir.iterdir():
                if d.is_dir():
                    uid_str = d.name.split("_")[0]
                    if uid_str.isdigit():
                        existing.add(int(uid_str))
        print(f"\n  Skipping {len(existing)} already-dumped players")

    # ── Step 2: Discover all player IDs ──────────────────────
    print(f"\n[1/3] Discovering player IDs...")
    player_ids = discover_player_ids(existing, args.max)
    total = len(player_ids)
    print(f"  >> Found {total} players")

    if not player_ids:
        print("\nNo players found. Check API key and universe ID.")
        sys.exit(0)

    # ── Step 3: Fetch usernames ───────────────────────────────
    print(f"\n[2/3] Fetching usernames ({total} players)...")
    usernames = get_usernames_bulk(player_ids)
    print(f"  >> Resolved {len(usernames)}/{total}")

    # ── Step 4: Parallel dump ─────────────────────────────────
    print(f"\n[3/3] Dumping {total} players with {args.workers} threads...\n")
    Path("player_dumps").mkdir(exist_ok=True)
    _start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(worker, uid, *usernames.get(uid, (None, None)), total)
            for uid in player_ids
        ]
        for f in as_completed(futures):
            f.result()

    elapsed = time.time() - _start_time

    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.0f}s ({_request_count} API requests)")
    print(f"  Dumped:  {_success}")
    print(f"  Empty:   {_empty}")
    print(f"  Failed:  {_failed}")
    print(f"  Total:   {total}")
    if elapsed > 0:
        print(f"  Speed:   {total / elapsed:.1f} players/s, {_request_count / elapsed:.1f} req/s")

    if _empty_ids:
        print(f"\nEmpty players ({len(_empty_ids)}):")
        for uid in sorted(_empty_ids):
            uinfo = usernames.get(uid, (None, None))
            label = f"@{uinfo[0]}" if uinfo[0] else f"id:{uid}"
            print(f"  {uid}  {label}")

    print(f"\nFiles saved to: player_dumps/")

    manifest = {
        "dumpedAt": datetime.utcnow().isoformat() + "Z",
        "universeId": UNIVERSE_ID,
        "totalPlayers": total,
        "success": _success,
        "empty": _empty,
        "failed": _failed,
        "apiRequests": _request_count,
        "elapsedSeconds": round(elapsed, 1),
        "workers": args.workers,
        "rate": args.rate,
        "playerIds": sorted(player_ids),
        "emptyPlayerIds": sorted(_empty_ids),
    }
    with open("player_dumps/_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: player_dumps/_manifest.json")


if __name__ == "__main__":
    main()
