"""
Postgres layer for Ocean Quest Fish Economy.

One module owns all SQL: schema creation, the daemon's writes, and the
dashboard's reads. Latest-state-only — re-fetching a player overwrites
their row and the contents of their child tables.

Connection string comes from $DATABASE_URL.
  - Daemon (on the box):     postgresql://ocean_quest_rw:...@localhost/ocean_quest
  - Streamlit Cloud reader:  postgresql://ocean_quest_ro:...@<host>:5432/ocean_quest?sslmode=require
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, date
from typing import Any, Iterable

import psycopg
from psycopg.types.json import Jsonb


# ─── Connection ─────────────────────────────────────────────────────────

def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Example: "
            "postgresql://ocean_quest_rw:PASS@localhost/ocean_quest"
        )
    return url


def connect(autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(database_url(), autocommit=autocommit)


# ─── Schema ─────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS players (
    user_id              BIGINT PRIMARY KEY,
    username             TEXT,
    display_name         TEXT,
    discovered_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_fetched_at      TIMESTAMPTZ,
    status               TEXT,

    gold                 BIGINT,
    pearls               BIGINT,
    best_rod_level       INT,
    playtime_seconds     BIGINT,
    sessions_count       INT,
    days_played          INT,
    longest_session_sec  INT,
    first_join_at        TIMESTAMPTZ,
    last_join_at         TIMESTAMPTZ,
    join_date            DATE,
    fish_discovered      INT,
    fish_total_caught    INT,
    fish_mutations_count INT,
    ships_owned          INT,
    active_ship          TEXT,
    best_ship            TEXT,
    total_gold_earned    BIGINT,
    total_gold_spent     BIGINT,
    pvp_kills            INT,
    deaths               INT,
    current_platform     TEXT,

    data                 JSONB
);

CREATE INDEX IF NOT EXISTS idx_players_fetch_priority
    ON players (last_fetched_at NULLS FIRST);
CREATE INDEX IF NOT EXISTS idx_players_status
    ON players (status);
CREATE INDEX IF NOT EXISTS idx_players_gold_desc
    ON players (gold DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_players_last_join
    ON players (last_join_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_players_join_date
    ON players (join_date);
CREATE INDEX IF NOT EXISTS idx_players_data_gin
    ON players USING GIN (data jsonb_path_ops);


CREATE TABLE IF NOT EXISTS player_sessions (
    user_id          BIGINT NOT NULL REFERENCES players(user_id) ON DELETE CASCADE,
    session_index    INT NOT NULL,
    started_at       TIMESTAMPTZ,
    ended_at         TIMESTAMPTZ,
    duration_seconds INT,
    platform         TEXT,
    PRIMARY KEY (user_id, session_index)
);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at
    ON player_sessions (started_at);


CREATE TABLE IF NOT EXISTS player_fishdex (
    user_id         BIGINT NOT NULL REFERENCES players(user_id) ON DELETE CASCADE,
    species_id      TEXT NOT NULL,
    total_caught    INT,
    first_caught_at TIMESTAMPTZ,
    mutations       JSONB,
    PRIMARY KEY (user_id, species_id)
);
CREATE INDEX IF NOT EXISTS idx_fishdex_species
    ON player_fishdex (species_id);


CREATE TABLE IF NOT EXISTS player_ships (
    user_id BIGINT NOT NULL REFERENCES players(user_id) ON DELETE CASCADE,
    ship_id TEXT NOT NULL,
    owned   BOOLEAN,
    PRIMARY KEY (user_id, ship_id)
);
CREATE INDEX IF NOT EXISTS idx_ships_owned
    ON player_ships (ship_id) WHERE owned;


CREATE TABLE IF NOT EXISTS player_zones (
    user_id         BIGINT NOT NULL REFERENCES players(user_id) ON DELETE CASCADE,
    zone            TEXT NOT NULL,
    species_count   INT,
    mutations_count INT,
    time_seconds    INT,
    visited         BOOLEAN,
    PRIMARY KEY (user_id, zone)
);
CREATE INDEX IF NOT EXISTS idx_zones_zone
    ON player_zones (zone);
"""


def init_schema(conn: psycopg.Connection | None = None) -> None:
    own = conn is None
    if own:
        conn = connect(autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        if not own:
            conn.commit()
    finally:
        if own:
            conn.close()


# ─── Flatten: JSON blob → typed scalars + child rows ────────────────────

def _epoch_to_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    try:
        return datetime.fromtimestamp(v, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _epoch_to_date(value: Any) -> date | None:
    dt = _epoch_to_dt(value)
    return dt.date() if dt else None


def _count_truthy(d: Any) -> int:
    if isinstance(d, dict):
        return sum(1 for v in d.values() if v)
    if isinstance(d, list):
        return sum(1 for v in d if v)
    return 0


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None if not isinstance(value, bool) else int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def flatten(data: dict) -> dict[str, Any]:
    """Pull hot scalars out of a combined save dict.

    Returns a flat dict of column_name → value. Missing fields → None.
    Never raises on shape variation; the JSON has evolved over time and
    will keep evolving.
    """
    out: dict[str, Any] = {}

    inv = data.get("Inventory") or {}
    cur = inv.get("currencies") or {}
    out["gold"] = _safe_int(cur.get("gold"))
    out["pearls"] = _safe_int(cur.get("pearls"))
    out["best_rod_level"] = _safe_int(inv.get("bestRodLevel"))

    pm = data.get("PlayerMetrics") or {}
    sessions = pm.get("sessions") or {}
    if isinstance(sessions, dict):
        out["playtime_seconds"] = _safe_int(sessions.get("totalPlaytime"))
        out["sessions_count"] = _safe_int(sessions.get("totalCount"))
        out["days_played"] = _safe_int(sessions.get("daysPlayed"))
        out["longest_session_sec"] = _safe_int(sessions.get("longestSession"))
        out["first_join_at"] = _epoch_to_dt(sessions.get("firstJoinDate"))
        out["last_join_at"] = _epoch_to_dt(sessions.get("lastJoinDate"))
        out["join_date"] = _epoch_to_date(sessions.get("firstJoinDate"))
        out["current_platform"] = sessions.get("currentPlatform") or None
    else:
        out["playtime_seconds"] = None
        out["sessions_count"] = None
        out["days_played"] = None
        out["longest_session_sec"] = None
        out["first_join_at"] = None
        out["last_join_at"] = None
        out["join_date"] = None
        out["current_platform"] = None

    fishing = pm.get("fishing") or {}
    out["fish_discovered"] = _safe_int(fishing.get("fishDiscovered"))
    out["fish_total_caught"] = _safe_int(fishing.get("totalCaught"))

    fd = data.get("FishDex") or {}
    zones = fd.get("zones")
    if isinstance(zones, dict):
        out["fish_mutations_count"] = sum(
            _safe_int((z or {}).get("mutationsCount")) or 0 for z in zones.values()
        )
    else:
        out["fish_mutations_count"] = None

    sd = data.get("ShipsData") or {}
    owned = sd.get("OwnedShips") or {}
    out["ships_owned"] = _count_truthy(owned)
    out["active_ship"] = (
        str(sd.get("ActiveShip")) if sd.get("ActiveShip") is not None else None
    )
    out["best_ship"] = (
        str(sd.get("BestShip")) if sd.get("BestShip") is not None else None
    )

    economy = pm.get("economy") or {}
    out["total_gold_earned"] = _safe_int(economy.get("totalGoldEarned"))
    out["total_gold_spent"] = _safe_int(economy.get("totalGoldSpent"))

    combat = pm.get("combat") or {}
    out["pvp_kills"] = _safe_int(combat.get("pvpKills"))
    out["deaths"] = _safe_int(combat.get("deaths"))

    return out


def _session_rows(uid: int, data: dict) -> Iterable[tuple]:
    pm = data.get("PlayerMetrics") or {}
    sessions = pm.get("sessions") or {}
    history = sessions.get("history") if isinstance(sessions, dict) else None
    if not isinstance(history, list):
        return
    for idx, s in enumerate(history):
        if not isinstance(s, dict):
            continue
        yield (
            uid,
            idx,
            _epoch_to_dt(s.get("startTime")),
            _epoch_to_dt(s.get("endTime")),
            _safe_int(s.get("duration")),
            s.get("platform") or None,
        )


def _fishdex_rows(uid: int, data: dict) -> Iterable[tuple]:
    fd = data.get("FishDex") or {}
    species = fd.get("fishDex")
    if not isinstance(species, dict):
        return
    for species_id, entry in species.items():
        if not isinstance(entry, dict):
            continue
        yield (
            uid,
            str(species_id),
            _safe_int(entry.get("totalCaught")),
            _epoch_to_dt(entry.get("firstCaughtAt")),
            Jsonb(entry.get("mutations")) if entry.get("mutations") is not None else None,
        )


def _ship_rows(uid: int, data: dict) -> Iterable[tuple]:
    sd = data.get("ShipsData") or {}
    owned = sd.get("OwnedShips")
    if not isinstance(owned, dict):
        return
    for ship_id, owned_flag in owned.items():
        yield (uid, str(ship_id), bool(owned_flag))


def _zone_rows(uid: int, data: dict) -> Iterable[tuple]:
    fd = data.get("FishDex") or {}
    zones = fd.get("zones") or {}
    pm = data.get("PlayerMetrics") or {}
    explor = pm.get("exploration") or {}
    time_per_zone = explor.get("timePerZone") or {}
    visited_map = explor.get("zonesVisited") or {}

    zone_names = set()
    if isinstance(zones, dict):
        zone_names |= set(zones.keys())
    if isinstance(time_per_zone, dict):
        zone_names |= set(time_per_zone.keys())
    if isinstance(visited_map, dict):
        zone_names |= set(visited_map.keys())

    for zone in zone_names:
        z = zones.get(zone) if isinstance(zones, dict) else None
        species_count = _safe_int((z or {}).get("speciesCount"))
        mutations_count = _safe_int((z or {}).get("mutationsCount"))
        time_sec = _safe_int(time_per_zone.get(zone)) if isinstance(time_per_zone, dict) else None
        visited = bool(visited_map.get(zone)) if isinstance(visited_map, dict) else None
        yield (uid, zone, species_count, mutations_count, time_sec, visited)


# ─── Writes ─────────────────────────────────────────────────────────────

def record_discovered(conn: psycopg.Connection, uid: int) -> bool:
    """Mark a player ID as known. Returns True if newly inserted."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO players (user_id) VALUES (%s) "
            "ON CONFLICT (user_id) DO NOTHING",
            (uid,),
        )
        return cur.rowcount == 1


def record_fetch(
    conn: psycopg.Connection,
    uid: int,
    status: str,
    username: str | None,
    display_name: str | None,
    data: dict | None,
) -> None:
    """ETL one player's fetch result.

    For status='ok' with `data`, this:
      1) upserts players row (hot scalars + JSONB blob)
      2) replaces all child rows for this player

    For 'empty' or 'failed' (no data), only the players row is updated —
    child tables for this player are *left as-is* so a transient failure
    doesn't wipe history.
    """
    fetched_at = datetime.now(timezone.utc)

    if status == "ok" and isinstance(data, dict):
        scalars = flatten(data)
        cols = list(scalars.keys())
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * (5 + len(cols)))
            update_set = ", ".join(
                f"{c} = EXCLUDED.{c}" for c in [
                    "username", "display_name", "last_fetched_at", "status", *cols, "data"
                ]
            )
            sql = (
                f"INSERT INTO players "
                f"(user_id, username, display_name, last_fetched_at, status, "
                f"{', '.join(cols)}, data) "
                f"VALUES ({placeholders}, %s) "
                f"ON CONFLICT (user_id) DO UPDATE SET {update_set}"
            )
            params = [
                uid, username, display_name, fetched_at, status,
                *[scalars[c] for c in cols],
                Jsonb(data),
            ]
            cur.execute(sql, params)

            cur.execute("DELETE FROM player_sessions WHERE user_id = %s", (uid,))
            session_data = list(_session_rows(uid, data))
            if session_data:
                cur.executemany(
                    "INSERT INTO player_sessions "
                    "(user_id, session_index, started_at, ended_at, duration_seconds, platform) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    session_data,
                )

            cur.execute("DELETE FROM player_fishdex WHERE user_id = %s", (uid,))
            fishdex_data = list(_fishdex_rows(uid, data))
            if fishdex_data:
                cur.executemany(
                    "INSERT INTO player_fishdex "
                    "(user_id, species_id, total_caught, first_caught_at, mutations) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    fishdex_data,
                )

            cur.execute("DELETE FROM player_ships WHERE user_id = %s", (uid,))
            ship_data = list(_ship_rows(uid, data))
            if ship_data:
                cur.executemany(
                    "INSERT INTO player_ships (user_id, ship_id, owned) VALUES (%s, %s, %s)",
                    ship_data,
                )

            cur.execute("DELETE FROM player_zones WHERE user_id = %s", (uid,))
            zone_data = list(_zone_rows(uid, data))
            if zone_data:
                cur.executemany(
                    "INSERT INTO player_zones "
                    "(user_id, zone, species_count, mutations_count, time_seconds, visited) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    zone_data,
                )
    else:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO players (user_id, username, display_name, "
                "last_fetched_at, status) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (user_id) DO UPDATE SET "
                "username = COALESCE(EXCLUDED.username, players.username), "
                "display_name = COALESCE(EXCLUDED.display_name, players.display_name), "
                "last_fetched_at = EXCLUDED.last_fetched_at, "
                "status = EXCLUDED.status",
                (uid, username, display_name, fetched_at, status),
            )

    conn.commit()


# ─── Reads ──────────────────────────────────────────────────────────────

def select_targets(
    conn: psycopg.Connection,
    refresh_after_hours: float | None,
    limit: int,
) -> list[int]:
    """UIDs to fetch this run, in priority order:
      1. Never fetched (status IS NULL or last_fetched_at IS NULL)
      2. Previously failed
      3. Stale (last_fetched_at older than cutoff, if cutoff given)
    """
    parts = []
    parts.append(
        "SELECT user_id FROM players "
        "WHERE last_fetched_at IS NULL OR status IS NULL "
        "ORDER BY discovered_at"
    )
    parts.append(
        "SELECT user_id FROM players "
        "WHERE status = 'failed' "
        "ORDER BY last_fetched_at NULLS FIRST"
    )
    if refresh_after_hours is not None and refresh_after_hours > 0:
        parts.append(
            f"SELECT user_id FROM players "
            f"WHERE status IN ('ok','empty') "
            f"AND last_fetched_at < NOW() - INTERVAL '{float(refresh_after_hours)} hours' "
            f"ORDER BY last_fetched_at NULLS FIRST"
        )

    union = " UNION ALL ".join(f"({p})" for p in parts)
    sql = f"SELECT user_id FROM ({union}) q LIMIT %s"
    with conn.cursor() as cur:
        cur.execute(sql, (limit,))
        return [r[0] for r in cur.fetchall()]


def count_by_status(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(status, 'unfetched'), COUNT(*) "
            "FROM players GROUP BY 1"
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def count_total(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM players")
        return cur.fetchone()[0]


# ─── CLI: `python db.py init` ───────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "init"
    if cmd == "init":
        init_schema()
        print("Schema initialised.")
    elif cmd == "status":
        with connect() as c:
            counts = count_by_status(c)
            total = count_total(c)
        print(f"Total players: {total}")
        for k, v in sorted(counts.items()):
            print(f"  {k:11s} {v}")
    else:
        print(f"Unknown command: {cmd!r}")
        print("Usage: python db.py [init|status]")
        sys.exit(1)
