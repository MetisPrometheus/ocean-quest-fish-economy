"""
Dump a player's DataStore2 data from Roblox Open Cloud API.
Checks both combined DATA store and separate per-store DataStores.
Saves to folder: player_dumps/{userId}_{username}/

Usage:
    python dump_player_data.py <userId>
    python dump_player_data.py 10384036675
"""

import os, sys, json, urllib.request, urllib.error
from pathlib import Path

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

API_KEY = os.environ.get("ROBLOX_API_KEY", "YOUR_API_KEY_HERE")
UNIVERSE_ID = os.environ.get("ROBLOX_UNIVERSE_ID", "YOUR_UNIVERSE_ID_HERE")

BASE = (
    f"https://apis.roblox.com/datastores/v1/universes/{UNIVERSE_ID}/standard-datastores"
)
ORDERED_BASE = f"https://apis.roblox.com/ordered-datastores/v1/universes/{UNIVERSE_ID}/orderedDataStores"

ALL_STORES = [
    "Inventory",
    "ShipsData",
    "FishDex",
    "OfflineRewards",
    "QuestData",
    "QuestProgress",
    "Purchases",
    "SafeStorage",
    "PlayerMetrics",
]


def api_get(url):
    req = urllib.request.Request(url, headers={"x-api-key": API_KEY})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError:
        return None


def get_username(user_id):
    try:
        with urllib.request.urlopen(
            f"https://users.roblox.com/v1/users/{user_id}"
        ) as r:
            d = json.loads(r.read().decode())
            return d.get("name"), d.get("displayName")
    except:
        return None, None


def get_latest_key_ordered(ds_name):
    encoded = urllib.request.quote(ds_name, safe="")
    data = api_get(
        f"{ORDERED_BASE}/{encoded}/scopes/global/entries?max_page_size=1&order_by=desc"
    )
    if data and "entries" in data and data["entries"]:
        return int(data["entries"][0].get("value", 0))
    return None


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
    return api_get(
        f"{BASE}/datastore/entries/entry?datastoreName={encoded}&entryKey={key}"
    )


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


def dump_player(user_id):
    username, display_name = get_username(user_id)
    label = f"{display_name} (@{username})" if username else f"userId:{user_id}"
    print(f"Player: {label}")
    print(f"UserId: {user_id}")
    print("=" * 60)

    # Create output folder
    folder_name = f"{user_id}_{username}" if username else str(user_id)
    folder = Path("player_dumps") / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    # 1. Combined DATA store
    ds = f"DATA/{user_id}"
    print(f"\n1. Combined store: {ds}")
    combined, key = fetch_latest(ds)
    if combined and isinstance(combined, dict):
        print(f"   OK (key={key}) — stores: {sorted(combined.keys())}")
        # Save full combined blob
        with open(folder / "DATA.json", "w") as f:
            json.dump(combined, f, indent=2)
        print(f"   -> {folder}/DATA.json")
        # Save each store as individual file
        for store_name, store_data in combined.items():
            with open(folder / f"{store_name}.json", "w") as f:
                json.dump(store_data, f, indent=2)
            print(f"   -> {folder}/{store_name}.json")
    else:
        print(f"   No combined data")

    # 2. Check separate DataStores for stores NOT in combined blob
    combined_keys = set(combined.keys()) if combined else set()
    missing = [s for s in ALL_STORES if s not in combined_keys]
    if missing:
        print(f"\n2. Checking separate DataStores: {missing}")
        for store in missing:
            data, k = fetch_latest(f"{store}/{user_id}")
            if data:
                print(f"   [{store}] FOUND separately (key={k})")
                with open(folder / f"{store}_SEPARATE.json", "w") as f:
                    json.dump(data, f, indent=2)
                print(f"   -> {folder}/{store}_SEPARATE.json")
            else:
                print(f"   [{store}] not found")

    # 3. Save player info
    info = {
        "userId": int(user_id),
        "username": username,
        "displayName": display_name,
        "combinedSaveKey": key,
        "storesInCombined": sorted(list(combined_keys)),
    }
    with open(folder / "_player.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All files saved to: {folder}/")
    for f in sorted(folder.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:30s} {size:>8,} bytes")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dump_player_data.py <userId>")
        sys.exit(1)
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Set ROBLOX_API_KEY in .env")
        sys.exit(1)
    if UNIVERSE_ID == "YOUR_UNIVERSE_ID_HERE":
        print("ERROR: Set ROBLOX_UNIVERSE_ID in .env")
        sys.exit(1)
    dump_player(sys.argv[1])
