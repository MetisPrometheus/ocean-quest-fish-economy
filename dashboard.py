"""
Pirate Fishing & Combat — Player Data Dashboard

Reads from Postgres (`players` + child tables, populated by
`dump_all_players.py` running on the Hetzner box). The full combined save
for each player still lives in the JSONB `data` column, so analytics code
that reaches into `Inventory`, `PlayerMetrics`, etc. keeps working.

Connection string is read from `st.secrets["DATABASE_URL"]` on Streamlit
Cloud, falling back to `os.environ["DATABASE_URL"]` for local dev.
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

import psycopg
from psycopg.rows import dict_row

# ── Config ──────────────────────────────────────────────────────────────
DATA_FILES = [
    "FishDex",
    "Inventory",
    "OfflineRewards",
    "PlayerMetrics",
    "Purchases",
    "SafeStorage",
    "ShipsData",
]

ZONE_COLORS = {
    "Pirate": "#D4763B",
    "Greek": "#4A90D9",
    "Japanese": "#D94A6B",
    "Viking": "#6B4AD9",
}
RARITY_ORDER = ["Poor", "Common", "Uncommon", "Rare", "Epic", "Legendary", "Mythic"]
RARITY_COLORS = {
    "Poor": "#9E9E9E",
    "Common": "#4CAF50",
    "Uncommon": "#2196F3",
    "Rare": "#9C27B0",
    "Epic": "#FF9800",
    "Legendary": "#F44336",
    "Mythic": "#FFD700",
}
SHIP_NAMES = {
    100: "Dinghy",
    200: "Sloop",
    300: "Caravel",
    400: "Ketch",
    420: "Destroyer",
    500: "Frigate",
}
SHIP_ORDER = [100, 200, 300, 400, 420, 500]

ZONE_RARITY_RNG = {
    "Pirate": {
        "Poor": 5,
        "Common": 25,
        "Uncommon": 50,
        "Rare": 120,
        "Epic": 250,
        "Legendary": 400,
        "Mythic": 800,
    },
    "Greek": {
        "Poor": 5,
        "Common": 300,
        "Uncommon": 550,
        "Rare": 850,
        "Epic": 1200,
        "Legendary": 1600,
        "Mythic": 2200,
    },
    "Japanese": {
        "Poor": 5,
        "Common": 1400,
        "Uncommon": 2800,
        "Rare": 3400,
        "Epic": 4300,
        "Legendary": 6200,
        "Mythic": 8000,
    },
    "Viking": {
        "Poor": 5,
        "Common": 6400,
        "Uncommon": 9400,
        "Rare": 10500,
        "Epic": 14200,
        "Legendary": 18500,
        "Mythic": 25000,
    },
}


# ── Postgres data layer ─────────────────────────────────────────────────
def _database_url() -> str:
    if hasattr(st, "secrets"):
        try:
            url = st.secrets.get("DATABASE_URL")
            if url:
                return url
        except (FileNotFoundError, KeyError, AttributeError):
            pass
    url = os.environ.get("DATABASE_URL")
    if not url:
        st.error(
            "DATABASE_URL is not configured. "
            "Set it in `.streamlit/secrets.toml` (Streamlit Cloud secrets panel) "
            "or as an environment variable for local development."
        )
        st.stop()
    return url


@st.cache_resource(show_spinner=False)
def get_conn() -> psycopg.Connection:
    """One long-lived connection, reused across reruns."""
    return psycopg.connect(_database_url(), row_factory=dict_row)


@st.cache_data(ttl=300, show_spinner=False)
def list_players() -> list[dict]:
    """All players with `status='ok'`, latest first. Used for the sidebar picker."""
    sql = (
        "SELECT user_id, username, display_name, last_fetched_at "
        "FROM players WHERE status = 'ok' "
        "ORDER BY username NULLS LAST, user_id"
    )
    with get_conn().cursor() as cur:
        cur.execute(sql)
        return list(cur.fetchall())


@st.cache_data(ttl=300, show_spinner=False)
def load_player_data(user_id: int) -> dict:
    """Returns the same dict shape the old folder loader did:
       {"FishDex": ..., "Inventory": ..., "PlayerMetrics": ..., ...}.
    """
    with get_conn().cursor() as cur:
        cur.execute("SELECT data FROM players WHERE user_id = %s", (int(user_id),))
        row = cur.fetchone()
    if not row or not row["data"]:
        return {name: {} for name in DATA_FILES}
    blob = row["data"]
    out = {name: blob.get(name, {}) for name in DATA_FILES}
    return out


@st.cache_data(ttl=300, show_spinner=False)
def players_summary() -> dict:
    """Cheap aggregate counts for the sidebar footer."""
    with get_conn().cursor() as cur:
        cur.execute(
            "SELECT COALESCE(status, 'unfetched') AS s, COUNT(*) AS c "
            "FROM players GROUP BY 1"
        )
        rows = cur.fetchall()
    return {row["s"]: row["c"] for row in rows}


def player_label(p: dict) -> str:
    name = p.get("username") or p.get("display_name") or str(p["user_id"])
    return f"{name} ({p['user_id']})"


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def fmt_timestamp(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError):
        return "—"


def get_first_join_date(metrics: dict) -> date | None:
    """Extract the date of the player's first session."""
    sessions = metrics.get("sessions", {})
    history = sessions.get("history", [])
    if history and history[0].get("startTime"):
        try:
            return datetime.fromtimestamp(history[0]["startTime"]).date()
        except (OSError, ValueError):
            pass
    return None


def get_zone_for_fish_id(fish_id: int) -> str:
    if 1000 <= fish_id <= 1099:
        return "Pirate"
    elif 2000 <= fish_id <= 2099:
        return "Greek"
    elif 3000 <= fish_id <= 3099:
        return "Japanese"
    elif 4000 <= fish_id <= 4099:
        return "Viking"
    # Subzone fish
    elif 10000 <= fish_id <= 11999:
        return "Pirate"
    elif 20000 <= fish_id <= 22999:
        return "Greek"
    elif 30000 <= fish_id <= 32999:
        return "Japanese"
    elif 40000 <= fish_id <= 41999:
        return "Viking"
    return "Unknown"


# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pirate Game — Player Dashboard", layout="wide", page_icon="🏴‍☠️"
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 { margin: 0; color: #8892b0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card p { margin: 0.3rem 0 0; color: #e6f1ff; font-size: 1.8rem; font-weight: 700; }
    .metric-card .sub { color: #5a6785; font-size: 0.75rem; margin-top: 0.2rem; }
    .section-header { border-left: 4px solid #D4763B; padding-left: 12px; margin: 1.5rem 0 1rem; }
    div[data-testid="stMetric"] label { color: #8892b0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar: Player Selection ───────────────────────────────────────────
st.sidebar.title("🏴‍☠️ Player Dashboard")

# Snapshot of all `status='ok'` players from Postgres. List is cached for 5min.
players_index = list_players()

if not players_index:
    st.warning("No player data in Postgres yet — has the dumper run?")
    st.stop()

# uid → row dict, plus a label → uid mapping for the picker
players_by_uid = {str(p["user_id"]): p for p in players_index}
player_options = {player_label(p): str(p["user_id"]) for p in players_index}

view_mode = st.sidebar.radio(
    "View", ["📊 Global Data", "👤 Individual Player"], index=0
)

if view_mode == "👤 Individual Player":
    st.sidebar.markdown("---")
    selected_label = st.sidebar.selectbox("Select Player", list(player_options.keys()))
    uid = player_options[selected_label]
    _player_row = players_by_uid[uid]
    player_name = (
        _player_row.get("username")
        or _player_row.get("display_name")
        or str(uid)
    )

    data = load_player_data(int(uid))
    inv = data.get("Inventory", {})
    metrics = data.get("PlayerMetrics", {})
    fishdex = data.get("FishDex", {})
    ships_data = data.get("ShipsData", {})
    purchases = data.get("Purchases", {})
    offline = data.get("OfflineRewards", {})
    safe_storage = data.get("SafeStorage", {})

    # Sidebar: quick stats
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**User ID:** `{uid}`")
    st.sidebar.markdown(f"**Best Rod Level:** `{inv.get('bestRodLevel', '?')}`")
    st.sidebar.markdown(f"**Gold:** `{inv.get('currencies', {}).get('gold', 0):,}`")
    st.sidebar.markdown(f"**Pearls:** `{inv.get('currencies', {}).get('pearls', 0):,}`")
    sessions = metrics.get("sessions", {})
    st.sidebar.markdown(
        f"**Total Playtime:** `{fmt_time(sessions.get('totalPlaytime', 0))}`"
    )
    st.sidebar.markdown(f"**Sessions:** `{sessions.get('totalCount', 0)}`")
    _join_period = sessions.get("joinedDuringPeriod", "")
    if _join_period:
        st.sidebar.markdown(f"**Last Join During:** `{_join_period}`")
    tutorial = metrics.get("tutorial", {})
    tut_status = (
        "✅ Complete"
        if tutorial.get("completed")
        else f"❌ Step {tutorial.get('highestStepReached', '?')} ({tutorial.get('stepIdReached', '?')})"
    )
    st.sidebar.markdown(f"**Tutorial:** {tut_status}")

    # Sidebar: dump freshness + sub-store presence
    st.sidebar.markdown("---")
    _last = _player_row.get("last_fetched_at")
    if _last:
        st.sidebar.markdown(f"**Last fetched:** `{_last:%Y-%m-%d %H:%M UTC}`")
    st.sidebar.markdown("**Sub-stores in JSON:**")
    for name in DATA_FILES:
        present = isinstance(data.get(name), dict) and bool(data.get(name))
        st.sidebar.markdown(
            f"{'✅' if present else '⬜'} {name}{'' if present else ' — empty'}"
        )
else:
    st.sidebar.markdown("---")

    # ── Date filter for Global Data ─────────────────────────────────────
    st.sidebar.markdown("**Filter by First Join Date**")
    date_filter_mode = st.sidebar.radio(
        "Date filter",
        ["All players", "Joined in date range"],
        index=0,
        key="date_filter_mode",
    )

    filter_date_start = None
    filter_date_end = None

    if date_filter_mode == "Joined in date range":
        filter_date_start = st.sidebar.date_input(
            "From", value=date(2026, 4, 1), key="filter_date_from"
        )
        filter_date_end = st.sidebar.date_input(
            "To", value=date.today(), key="filter_date_to"
        )


# ── Main Content ────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════
# PLAYER DATA VIEW
# ════════════════════════════════════════════════════════════════════════
if view_mode == "👤 Individual Player":
    st.title(f"🏴‍☠️ {player_name}")

    (
        tab_overview,
        tab_fishing,
        tab_economy,
        tab_ships,
        tab_sessions,
        tab_map,
        tab_raw,
    ) = st.tabs(
        [
            "📊 Overview",
            "🐟 Fishing & FishDex",
            "💰 Economy",
            "⛵ Ships & Weapons",
            "📅 Sessions",
            "🗺️ Location Map",
            "🔍 Raw Data",
        ]
    )

    # ════════════════════════════════════════════════════════════════════════
    # OVERVIEW TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_overview:
        fishing = metrics.get("fishing", {})
        economy = metrics.get("economy", {})
        combat = metrics.get("combat", {})

        # Top metrics row
        cols = st.columns(6)
        metric_data = [
            ("Total Caught", fishing.get("totalCaught", 0), "fish"),
            ("Species Found", fishing.get("fishDiscovered", 0), "unique"),
            ("Gold Earned", f"{economy.get('totalGoldEarned', 0):,}", "lifetime"),
            ("Gold Spent", f"{economy.get('totalGoldSpent', 0):,}", "lifetime"),
            ("PvP Kills", combat.get("pvpKills", 0), "players"),
            ("Ships Sunk", combat.get("shipsSunk", 0), "total"),
        ]
        for i, (label, value, sub) in enumerate(metric_data):
            with cols[i]:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h3>{label}</h3>
                    <p>{value}</p>
                    <div class="sub">{sub}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Second row
        cols2 = st.columns(7)
        metric_data2 = [
            ("Best Rod", inv.get("bestRodLevel", 0), f"of 30"),
            (
                "Playtime",
                fmt_time(sessions.get("totalPlaytime", 0)),
                f"{sessions.get('totalCount', 0)} sessions",
            ),
            (
                "Avg Session",
                fmt_time(
                    sessions.get("totalPlaytime", 0)
                    / max(1, sessions.get("totalCount", 1))
                ),
                "",
            ),
            (
                "Fish Sold",
                fishing.get("totalSold", 0),
                f"{fishing.get('totalGoldFromFish', 0):,}g",
            ),
            ("Deaths", combat.get("deaths", 0), ""),
            ("Damage Dealt", f"{combat.get('totalDamageDealt', 0):,}", "HP"),
            (
                "Floatables Shot",
                f"{combat.get('floatablesShot', 0):,}",
                "harpoon hits",
            ),
        ]
        for i, (label, value, sub) in enumerate(metric_data2):
            with cols2[i]:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h3>{label}</h3>
                    <p>{value}</p>
                    <div class="sub">{sub}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Fishing breakdown & Zone discovery side by side
        col_fish, col_zone = st.columns(2)

        with col_fish:
            st.markdown(
                '<div class="section-header"><h4>Fishing Breakdown</h4></div>',
                unsafe_allow_html=True,
            )
            fish_breakdown = {
                "Safe Zone": fishing.get("caughtInSafeZone", 0),
                "PvP Zone": fishing.get("caughtInPvPZone", 0),
                "Border": fishing.get("caughtBorderFishing", 0),
                "Mutation Event": fishing.get("caughtInMutationEvent", 0),
            }
            fb_df = pd.DataFrame(
                list(fish_breakdown.items()), columns=["Location", "Count"]
            )
            fb_df = fb_df[fb_df["Count"] > 0]
            if not fb_df.empty:
                fig = px.pie(
                    fb_df,
                    values="Count",
                    names="Location",
                    color_discrete_sequence=[
                        "#4CAF50",
                        "#F44336",
                        "#FF9800",
                        "#9C27B0",
                    ],
                )
                fig.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccd6f6",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fishing data yet")

        with col_zone:
            st.markdown(
                '<div class="section-header"><h4>Zone Discovery</h4></div>',
                unsafe_allow_html=True,
            )
            zones = fishdex.get("zones", {})
            zone_rows = []
            for zname in ["Pirate", "Greek", "Japanese", "Viking"]:
                zd = zones.get(zname, {})
                zone_rows.append(
                    {
                        "Zone": zname,
                        "Species": zd.get("speciesCount", 0),
                        "Mutations": zd.get("mutationsCount", 0),
                    }
                )
            zone_df = pd.DataFrame(zone_rows)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="Species",
                    x=zone_df["Zone"],
                    y=zone_df["Species"],
                    marker_color=[ZONE_COLORS.get(z, "#888") for z in zone_df["Zone"]],
                )
            )
            fig.add_trace(
                go.Bar(
                    name="Mutations",
                    x=zone_df["Zone"],
                    y=zone_df["Mutations"],
                    marker_color="#FFD700",
                    opacity=0.7,
                )
            )
            fig.update_layout(
                barmode="group",
                margin=dict(t=20, b=20, l=20, r=20),
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Exploration progress
        st.markdown(
            '<div class="section-header"><h4>Exploration</h4></div>',
            unsafe_allow_html=True,
        )
        exploration = metrics.get("exploration", {})
        visited = exploration.get("zonesVisited", {})
        zone_cols = st.columns(4)
        for i, zone in enumerate(["Pirate", "Greek", "Japanese", "Viking"]):
            with zone_cols[i]:
                v = visited.get(zone, False)
                emoji = "✅" if v else "⬜"
                st.markdown(f"**{emoji} {zone}**")

        # Purchases
        st.markdown(
            '<div class="section-header"><h4>Purchases</h4></div>',
            unsafe_allow_html=True,
        )
        perm = purchases.get("permanent", {})
        purch_cols = st.columns(4)
        purch_items = [
            ("Bag Tier", perm.get("bagTier", 0)),
            ("Sell Anywhere", "✅" if perm.get("sellAnywhere") else "❌"),
            ("Offline Extend", "✅" if perm.get("offlineExtend") else "❌"),
            ("Safe Storage Slots", safe_storage.get("purchaseCount", 0)),
        ]
        for i, (label, val) in enumerate(purch_items):
            with purch_cols[i]:
                st.metric(label, val)

    # ════════════════════════════════════════════════════════════════════════
    # FISHING & FISHDEX TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_fishing:
        st.markdown(
            '<div class="section-header"><h4>FishDex Entries</h4></div>',
            unsafe_allow_html=True,
        )

        fish_entries = fishdex.get("fishDex", {})
        if fish_entries:
            rows = []
            for fid_str, entry in fish_entries.items():
                fid = int(fid_str)
                zone = get_zone_for_fish_id(fid)
                is_subzone = fid >= 10000
                rows.append(
                    {
                        "Fish ID": fid,
                        "Zone": zone,
                        "Subzone": "Yes" if is_subzone else "No",
                        "Total Caught": entry.get("totalCaught", 0),
                        "Mutations": (
                            len(entry.get("mutations", []))
                            if isinstance(entry.get("mutations"), list)
                            else (
                                len(entry.get("mutations", {}).keys())
                                if isinstance(entry.get("mutations"), dict)
                                else 0
                            )
                        ),
                        "First Caught": fmt_timestamp(entry.get("firstCaughtAt", 0)),
                    }
                )
            fish_df = pd.DataFrame(rows).sort_values("Fish ID")

            # Summary metrics
            mcols = st.columns(4)
            with mcols[0]:
                st.metric("Total Species", len(fish_df))
            with mcols[1]:
                st.metric("Total Caught", fish_df["Total Caught"].sum())
            with mcols[2]:
                st.metric("With Mutations", (fish_df["Mutations"] > 0).sum())
            with mcols[3]:
                subzone_count = (fish_df["Subzone"] == "Yes").sum()
                st.metric("Subzone Fish", subzone_count)

            # Fish per zone chart
            zone_counts = fish_df.groupby("Zone")["Fish ID"].count().reset_index()
            zone_counts.columns = ["Zone", "Species Discovered"]
            fig = px.bar(
                zone_counts,
                x="Zone",
                y="Species Discovered",
                color="Zone",
                color_discrete_map=ZONE_COLORS,
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(t=20, b=20),
                height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Full table
            st.dataframe(fish_df, use_container_width=True, hide_index=True)
        else:
            st.info("No FishDex entries yet")

        # Rarity odds reference
        st.markdown(
            '<div class="section-header"><h4>Rarity Odds Reference (1 in X)</h4></div>',
            unsafe_allow_html=True,
        )
        rarity_rows = []
        for zone, rarities in ZONE_RARITY_RNG.items():
            row = {"Zone": zone}
            for rarity in RARITY_ORDER:
                row[rarity] = f"1 in {rarities.get(rarity, '?'):,}"
            rarity_rows.append(row)
        rarity_df = pd.DataFrame(rarity_rows)
        st.dataframe(rarity_df, use_container_width=True, hide_index=True)

        # ── Catch Timeline ───────────────────────────────────────────────────
        catch_log = metrics.get("fishing", {}).get("catchLog", [])
        if catch_log:
            st.markdown(
                '<div class="section-header"><h4>Catch Timeline</h4></div>',
                unsafe_allow_html=True,
            )
            session_start = sessions.get("history", [{}])[0].get("startTime", 0)

            cl_rows = []
            for entry in catch_log:
                ts = entry.get("ts", 0)
                elapsed = ts - session_start if session_start else 0
                cl_rows.append(
                    {
                        "Timestamp": fmt_timestamp(ts),
                        "Elapsed (s)": elapsed,
                        "Elapsed (min)": round(elapsed / 60, 2),
                        "Fish ID": entry.get("i", 0),
                        "Rarity": entry.get("ra", "Unknown"),
                        "Mutation": entry.get("mu", "Base"),
                        "Size": entry.get("sz", "?"),
                        "Rod Level": entry.get("r", 0),
                        "Zone": entry.get("z", "?"),
                        "Type": entry.get("t", "?"),
                    }
                )
            cl_df = pd.DataFrame(cl_rows)
            cl_df["Cumulative"] = range(1, len(cl_df) + 1)

            # Cumulative catches over time, coloured by rarity
            fig_cl = px.scatter(
                cl_df,
                x="Elapsed (min)",
                y="Cumulative",
                color="Rarity",
                symbol="Type",
                hover_data=[
                    "Timestamp",
                    "Fish ID",
                    "Mutation",
                    "Size",
                    "Rod Level",
                    "Zone",
                ],
                color_discrete_map=RARITY_COLORS,
                category_orders={"Rarity": RARITY_ORDER},
            )
            # Add connecting line beneath the dots
            fig_cl.add_trace(
                go.Scatter(
                    x=cl_df["Elapsed (min)"],
                    y=cl_df["Cumulative"],
                    mode="lines",
                    line=dict(color="#2a2a5e", width=1),
                    showlegend=False,
                )
            )
            fig_cl.update_traces(marker=dict(size=9), selector=dict(mode="markers"))
            fig_cl.update_layout(
                xaxis_title="Elapsed Time (min)",
                yaxis_title="Cumulative Fish Caught",
                margin=dict(t=20, b=20),
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                xaxis=dict(gridcolor="#1a1a3e"),
                yaxis=dict(gridcolor="#1a1a3e"),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_cl, use_container_width=True)

            # Rarity & size breakdown from catch log
            col_rar, col_sz = st.columns(2)
            with col_rar:
                rar_counts = (
                    cl_df["Rarity"]
                    .value_counts()
                    .reindex(RARITY_ORDER, fill_value=0)
                    .reset_index()
                )
                rar_counts.columns = ["Rarity", "Count"]
                fig_rar = px.bar(
                    rar_counts,
                    x="Rarity",
                    y="Count",
                    color="Rarity",
                    color_discrete_map=RARITY_COLORS,
                )
                fig_rar.update_layout(
                    showlegend=False,
                    margin=dict(t=20, b=20),
                    height=260,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccd6f6",
                    xaxis=dict(gridcolor="#1a1a3e"),
                    yaxis=dict(gridcolor="#1a1a3e"),
                )
                st.plotly_chart(fig_rar, use_container_width=True)
            with col_sz:
                sz_counts = cl_df["Size"].value_counts().reset_index()
                sz_counts.columns = ["Size", "Count"]
                fig_sz = px.bar(
                    sz_counts,
                    x="Size",
                    y="Count",
                    color_discrete_sequence=["#4A90D9"],
                )
                fig_sz.update_layout(
                    showlegend=False,
                    margin=dict(t=20, b=20),
                    height=260,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccd6f6",
                    xaxis=dict(gridcolor="#1a1a3e"),
                    yaxis=dict(gridcolor="#1a1a3e"),
                )
                st.plotly_chart(fig_sz, use_container_width=True)

            # Catch rate over time (rolling window)
            st.markdown("**Catches per minute (rolling 2-min window):**")
            if len(cl_df) > 1:
                _bin_size = 2.0  # minutes
                _max_min = cl_df["Elapsed (min)"].max()
                _bins = [i * _bin_size for i in range(int(_max_min / _bin_size) + 2)]
                _counts = []
                for _b in _bins[:-1]:
                    _c = (
                        (cl_df["Elapsed (min)"] >= _b)
                        & (cl_df["Elapsed (min)"] < _b + _bin_size)
                    ).sum()
                    _counts.append(
                        {"Minute": round(_b + _bin_size / 2, 1), "Catches": int(_c)}
                    )
                _rate_df = pd.DataFrame(_counts)
                fig_rate = px.bar(
                    _rate_df,
                    x="Minute",
                    y="Catches",
                    color="Catches",
                    color_continuous_scale=["#16213e", "#D4763B", "#FFD700"],
                )
                fig_rate.update_layout(
                    xaxis_title="Session Minute",
                    yaxis_title="Catches in 2-min window",
                    margin=dict(t=20, b=20),
                    height=220,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccd6f6",
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="#1a1a3e"),
                    yaxis=dict(gridcolor="#1a1a3e"),
                )
                st.plotly_chart(fig_rate, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # ECONOMY TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_economy:
        economy = metrics.get("economy", {})

        st.markdown(
            '<div class="section-header"><h4>Gold Flow</h4></div>',
            unsafe_allow_html=True,
        )

        earned = economy.get("totalGoldEarned", 0)
        spent = economy.get("totalGoldSpent", 0)
        from_fish = metrics.get("fishing", {}).get("totalGoldFromFish", 0)
        from_quests = economy.get("totalGoldFromQuests", 0)
        on_ships = economy.get("goldSpentOnShipUpgrades", 0)
        current = inv.get("currencies", {}).get("gold", 0)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Income Sources**")
            income_data = {
                "Fish Sales": from_fish,
                "Quests": from_quests,
                "Other": max(0, earned - from_fish - from_quests),
            }
            inc_df = pd.DataFrame(list(income_data.items()), columns=["Source", "Gold"])
            inc_df = inc_df[inc_df["Gold"] > 0]
            if not inc_df.empty:
                fig = px.pie(
                    inc_df,
                    values="Gold",
                    names="Source",
                    color_discrete_sequence=["#FFD700", "#4CAF50", "#2196F3"],
                )
                fig.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccd6f6",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No income data")

        with col2:
            st.markdown("**Spending**")
            spend_data = {
                "Ship Upgrades": on_ships,
                "Other": max(0, spent - on_ships),
            }
            sp_df = pd.DataFrame(list(spend_data.items()), columns=["Category", "Gold"])
            sp_df = sp_df[sp_df["Gold"] > 0]
            if not sp_df.empty:
                fig = px.pie(
                    sp_df,
                    values="Gold",
                    names="Category",
                    color_discrete_sequence=["#F44336", "#FF9800"],
                )
                fig.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccd6f6",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No spending data yet")

        # Gold balance waterfall
        st.markdown(
            '<div class="section-header"><h4>Gold Balance</h4></div>',
            unsafe_allow_html=True,
        )
        waterfall_data = pd.DataFrame(
            {
                "Category": ["Earned", "Spent", "Current"],
                "Amount": [earned, -spent, current],
                "Measure": ["relative", "relative", "total"],
            }
        )
        fig = go.Figure(
            go.Waterfall(
                x=waterfall_data["Category"],
                y=waterfall_data["Amount"],
                measure=waterfall_data["Measure"],
                connector={"line": {"color": "#5a6785"}},
                increasing={"marker": {"color": "#4CAF50"}},
                decreasing={"marker": {"color": "#F44336"}},
                totals={"marker": {"color": "#FFD700"}},
            )
        )
        fig.update_layout(
            margin=dict(t=20, b=20),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Economy efficiency
        st.markdown(
            '<div class="section-header"><h4>Efficiency</h4></div>',
            unsafe_allow_html=True,
        )
        total_caught = metrics.get("fishing", {}).get("totalCaught", 0)
        playtime_hrs = sessions.get("totalPlaytime", 0) / 3600

        eff_cols = st.columns(3)
        with eff_cols[0]:
            avg_fish_val = from_fish / max(
                1, metrics.get("fishing", {}).get("totalSold", 1)
            )
            st.metric("Avg Gold per Fish", f"{avg_fish_val:.0f}g")
        with eff_cols[1]:
            gph = earned / max(0.01, playtime_hrs)
            st.metric("Gold per Hour", f"{gph:.0f}g/hr")
        with eff_cols[2]:
            fph = total_caught / max(0.01, playtime_hrs)
            st.metric("Fish per Hour", f"{fph:.1f}")

        # ── Sales Timeline ───────────────────────────────────────────────────
        sales_log = economy.get("salesLog", [])
        purchase_log = economy.get("purchaseLog", [])

        if sales_log:
            st.markdown(
                '<div class="section-header"><h4>Sales Timeline</h4></div>',
                unsafe_allow_html=True,
            )
            _sess_start = sessions.get("history", [{}])[0].get("startTime", 0)
            sl_rows = []
            _running_gold = 0
            for sale in sales_log:
                ts = sale.get("ts", 0)
                ge = sale.get("ge", 0)  # gold earned this sale
                fc = sale.get("fc", 0)  # fish count
                gb = sale.get("gb", 0)  # gold before sale
                _running_gold += ge
                sl_rows.append(
                    {
                        "Elapsed (min)": (
                            round((ts - _sess_start) / 60, 2) if _sess_start else 0
                        ),
                        "Time": fmt_timestamp(ts),
                        "Fish Sold": fc,
                        "Gold Earned": ge,
                        "Gold Before": gb,
                        "Gold After": gb + ge,
                        "Cumulative Gold": _running_gold,
                        "Avg per Fish": round(ge / max(1, fc), 1),
                    }
                )
            sl_df = pd.DataFrame(sl_rows)

            col_sales1, col_sales2 = st.columns(2)
            with col_sales1:
                # Bar per sale — height = gold earned, width = fish count (via size)
                fig_sales = px.bar(
                    sl_df,
                    x="Elapsed (min)",
                    y="Gold Earned",
                    color="Avg per Fish",
                    color_continuous_scale=["#4A90D9", "#FFD700"],
                    hover_data=["Time", "Fish Sold", "Gold Before", "Gold After"],
                    text="Fish Sold",
                )
                fig_sales.update_traces(textposition="outside")
                fig_sales.update_layout(
                    xaxis_title="Session Time (min)",
                    yaxis_title="Gold Earned per Sale",
                    margin=dict(t=20, b=20),
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccd6f6",
                    xaxis=dict(gridcolor="#1a1a3e"),
                    yaxis=dict(gridcolor="#1a1a3e"),
                    coloraxis_colorbar=dict(title="Gold/fish"),
                )
                st.plotly_chart(fig_sales, use_container_width=True)
            with col_sales2:
                # Cumulative gold earned from fish over time
                fig_cum = go.Figure()
                fig_cum.add_trace(
                    go.Scatter(
                        x=sl_df["Elapsed (min)"],
                        y=sl_df["Cumulative Gold"],
                        mode="lines+markers",
                        line=dict(color="#FFD700", width=2.5),
                        marker=dict(size=8, color="#FFD700"),
                        fill="tozeroy",
                        fillcolor="rgba(255,215,0,0.1)",
                        text=sl_df["Time"],
                        hovertemplate="%{text}<br>Cumulative: %{y:,}g<extra></extra>",
                    )
                )
                fig_cum.update_layout(
                    xaxis_title="Session Time (min)",
                    yaxis_title="Cumulative Gold from Fish",
                    margin=dict(t=20, b=20),
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccd6f6",
                    xaxis=dict(gridcolor="#1a1a3e"),
                    yaxis=dict(gridcolor="#1a1a3e"),
                )
                st.plotly_chart(fig_cum, use_container_width=True)

        # ── Purchase Timeline ────────────────────────────────────────────────
        if purchase_log:
            st.markdown(
                '<div class="section-header"><h4>Purchase Timeline</h4></div>',
                unsafe_allow_html=True,
            )
            _sess_start = sessions.get("history", [{}])[0].get("startTime", 0)
            pl_rows = []
            for p in purchase_log:
                ts = p.get("ts", 0)
                pl_rows.append(
                    {
                        "Elapsed (min)": (
                            round((ts - _sess_start) / 60, 2) if _sess_start else 0
                        ),
                        "Time": fmt_timestamp(ts),
                        "Item": p.get("nm", "?"),
                        "Type": p.get("it", "?"),
                        "Level": p.get("lv", 0),
                        "Source": p.get("src", "?"),
                        "Gold Spent": p.get("gs", 0),
                        "Gold Before": p.get("gb", 0),
                        "Gold After": p.get("gb", 0) - p.get("gs", 0),
                        "Label": f"{p.get('nm','?')} Lv{p.get('lv',0)}",
                    }
                )
            pl_df = pd.DataFrame(pl_rows)

            TYPE_COLORS = {
                "Fishing Rod": "#4CAF50",
                "Ship": "#D4763B",
                "Upgrade": "#9C27B0",
            }
            fig_purch = px.scatter(
                pl_df,
                x="Elapsed (min)",
                y="Gold Spent",
                color="Type",
                color_discrete_map=TYPE_COLORS,
                size="Gold Spent",
                size_max=30,
                text="Label",
                hover_data=["Time", "Source", "Gold Before", "Gold After"],
            )
            fig_purch.update_traces(textposition="top center")
            fig_purch.update_layout(
                xaxis_title="Session Time (min)",
                yaxis_title="Gold Spent",
                margin=dict(t=40, b=20),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                xaxis=dict(gridcolor="#1a1a3e"),
                yaxis=dict(gridcolor="#1a1a3e"),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_purch, use_container_width=True)
            st.dataframe(
                pl_df[
                    [
                        "Time",
                        "Item",
                        "Type",
                        "Level",
                        "Source",
                        "Gold Spent",
                        "Gold Before",
                        "Gold After",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # ════════════════════════════════════════════════════════════════════════
    # SHIPS & WEAPONS TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_ships:
        st.markdown(
            '<div class="section-header"><h4>Ship Ownership</h4></div>',
            unsafe_allow_html=True,
        )

        owned = ships_data.get("OwnedShips", {})
        active_id = ships_data.get("ActiveShip", 0)
        best_id = ships_data.get("BestShip", 0)

        ship_cols = st.columns(6)
        for i, sid in enumerate(SHIP_ORDER):
            sname = SHIP_NAMES.get(sid, f"Ship {sid}")
            is_owned = owned.get(str(sid), False)
            is_active = sid == active_id
            with ship_cols[i]:
                status = ""
                if is_active:
                    status = "🚢 Active"
                elif is_owned:
                    status = "✅ Owned"
                else:
                    status = "🔒 Locked"
                st.markdown(f"**{sname}**\n\n{status}")

        # Ship storage contents
        st.markdown(
            '<div class="section-header"><h4>Ship Storage</h4></div>',
            unsafe_allow_html=True,
        )
        ship_storage = inv.get("ships", {})
        storage_rows = []
        for sname in ["Dinghy", "Sloop", "Caravel", "Ketch", "Destroyer", "Frigate"]:
            chests = ship_storage.get(sname, {})
            for cname, contents in chests.items():
                count = len(contents) if isinstance(contents, (list, dict)) else 0
                storage_rows.append({"Ship": sname, "Chest": cname, "Items": count})
        storage_df = pd.DataFrame(storage_rows)
        total_stored = storage_df["Items"].sum()
        st.metric("Total Items in Ship Storage", total_stored)
        if total_stored > 0:
            st.dataframe(
                storage_df[storage_df["Items"] > 0],
                use_container_width=True,
                hide_index=True,
            )

        # Weapon upgrade progress
        st.markdown(
            '<div class="section-header"><h4>Weapon Upgrade Progress</h4></div>',
            unsafe_allow_html=True,
        )
        slot_progress = ships_data.get("SlotProgress", {})
        actives = ships_data.get("SlotActives", {})

        weapon_rows = []
        for weapon_type, slots in slot_progress.items():
            if weapon_type in ("Chest", "Helm"):
                continue
            for slot_name, elements in slots.items():
                for element, level in elements.items():
                    if level > 0:
                        weapon_rows.append(
                            {
                                "Weapon": weapon_type,
                                "Slot": slot_name,
                                "Element": element,
                                "Level": level,
                            }
                        )

        if weapon_rows:
            wp_df = pd.DataFrame(weapon_rows)
            fig = px.bar(
                wp_df,
                x="Slot",
                y="Level",
                color="Weapon",
                facet_col="Element",
                barmode="group",
            )
            fig.update_layout(
                margin=dict(t=40, b=20),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No weapon upgrades yet — all slots at level 0")

        # Active weapons
        st.markdown("**Active Weapons:**")
        for wtype, slots in actives.items():
            if isinstance(slots, dict) and slots:
                for slot, wname in slots.items():
                    st.markdown(f"- **{wtype}** {slot}: `{wname}`")
            elif isinstance(slots, list) and slots:
                st.markdown(f"- **{wtype}**: {', '.join(str(s) for s in slots)}")

    # ════════════════════════════════════════════════════════════════════════
    # SESSIONS TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_sessions:
        st.markdown(
            '<div class="section-header"><h4>Session History</h4></div>',
            unsafe_allow_html=True,
        )

        history = sessions.get("history", [])
        if history:
            sess_rows = []
            for i, s in enumerate(history):
                sess_rows.append(
                    {
                        "#": i + 1,
                        "Start": fmt_timestamp(s.get("startTime", 0)),
                        "End": fmt_timestamp(s.get("endTime", 0)),
                        "Duration": fmt_time(s.get("duration", 0)),
                        "Duration (s)": s.get("duration", 0),
                    }
                )
            sess_df = pd.DataFrame(sess_rows)

            # Session duration chart
            fig = px.bar(
                sess_df,
                x="#",
                y="Duration (s)",
                labels={"Duration (s)": "Duration (seconds)", "#": "Session"},
                color="Duration (s)",
                color_continuous_scale=["#16213e", "#D4763B", "#FFD700"],
            )
            fig.update_layout(
                margin=dict(t=20, b=20),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            scols = st.columns(4)
            with scols[0]:
                st.metric(
                    "Longest Session", fmt_time(sessions.get("longestSession", 0))
                )
            with scols[1]:
                st.metric(
                    "Shortest Session", fmt_time(sessions.get("shortestSession", 0))
                )
            with scols[2]:
                avg = sessions.get("totalPlaytime", 0) / max(
                    1, sessions.get("totalCount", 1)
                )
                st.metric("Average Session", fmt_time(avg))
            with scols[3]:
                st.metric("Total Sessions", sessions.get("totalCount", 0))

            st.dataframe(
                sess_df.drop(columns=["Duration (s)"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No session history")

    # ════════════════════════════════════════════════════════════════════════
    # LOCATION MAP TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_map:
        st.markdown(
            '<div class="section-header"><h4>Player Movement Path</h4></div>',
            unsafe_allow_html=True,
        )

        location_path = metrics.get("locationPath", [])
        if location_path:
            # Island locations for reference
            islands = [
                {"name": "Pirate", "x": -420, "z": -200},
                {"name": "Greek", "x": 2555, "z": 5110},
                {"name": "Japanese", "x": -160, "z": 10200},
                {"name": "Viking", "x": 7292, "z": 15790},
            ]

            fig = go.Figure()

            # Player path
            xs = [p["x"] for p in location_path]
            zs = [p["z"] for p in location_path]
            times = [fmt_timestamp(p.get("t", 0)) for p in location_path]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=zs,
                    mode="lines+markers",
                    name="Player Path",
                    line=dict(color="#4CAF50", width=2),
                    marker=dict(
                        size=6, color=list(range(len(xs))), colorscale="Viridis"
                    ),
                    text=times,
                    hovertemplate="x=%{x}, z=%{y}<br>%{text}",
                )
            )

            # Start & end markers
            fig.add_trace(
                go.Scatter(
                    x=[xs[0]],
                    y=[zs[0]],
                    mode="markers+text",
                    name="Start",
                    marker=dict(size=14, color="#4CAF50", symbol="star"),
                    text=["START"],
                    textposition="top center",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[xs[-1]],
                    y=[zs[-1]],
                    mode="markers+text",
                    name="End",
                    marker=dict(size=14, color="#F44336", symbol="x"),
                    text=["END"],
                    textposition="top center",
                )
            )

            # Island markers
            for island in islands:
                fig.add_trace(
                    go.Scatter(
                        x=[island["x"]],
                        y=[island["z"]],
                        mode="markers+text",
                        name=island["name"],
                        marker=dict(
                            size=20,
                            color=ZONE_COLORS.get(island["name"], "#888"),
                            symbol="diamond",
                            line=dict(width=2, color="white"),
                        ),
                        text=[island["name"]],
                        textposition="top center",
                        textfont=dict(
                            size=12, color=ZONE_COLORS.get(island["name"], "#888")
                        ),
                    )
                )

            fig.update_layout(
                xaxis_title="X",
                yaxis_title="Z",
                margin=dict(t=20, b=20),
                height=600,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,10,30,0.8)",
                font_color="#ccd6f6",
                xaxis=dict(
                    gridcolor="#1a1a3e", zerolinecolor="#2a2a5e", autorange="reversed"
                ),
                yaxis=dict(
                    gridcolor="#1a1a3e", zerolinecolor="#2a2a5e", scaleanchor="x"
                ),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**{len(location_path)} position samples** recorded")
        else:
            st.info("No location path data")

    # ════════════════════════════════════════════════════════════════════════
    # RAW DATA TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_raw:
        st.markdown(
            '<div class="section-header"><h4>Raw JSON Data</h4></div>',
            unsafe_allow_html=True,
        )

        raw_selection = st.selectbox("Select data store", DATA_FILES)
        raw_data = data.get(raw_selection, {})
        st.json(raw_data)


# ════════════════════════════════════════════════════════════════════════
# AGGREGATE STATISTICS VIEW
# ════════════════════════════════════════════════════════════════════════
else:  # view_mode == "📊 Global Data"
    st.title("📊 Global Data — All Players")

    # Staff/free-purchase UserIds to exclude from aggregate stats
    STAFF_USER_IDS = {
        "7796282",  # Deathlys (@XxIvIxX)
        "56651650",  # Emberos (@Rolfien)
        "1068573686",  # Eric (@Eriiicx)
        "10782708602",  # Jone (@Jone_than1)
        "10784724184",  # BartolomeusTheKnight
        "10811787773",  # NixieBlush
    }

    # ── Daily new-joins chart ────────────────────────────────────────────────
    @st.cache_data(ttl=300, show_spinner=False)
    def _build_joins_by_date(staff_ids: frozenset) -> dict:
        """Reads `join_date` directly from `players` — no JSON parsing needed."""
        with get_conn().cursor() as cur:
            cur.execute(
                "SELECT join_date, COUNT(*) AS c "
                "FROM players "
                "WHERE status = 'ok' AND join_date IS NOT NULL "
                "AND user_id NOT IN ("
                + ",".join(["%s"] * len(staff_ids))
                + ") "
                "GROUP BY join_date",
                tuple(int(s) for s in staff_ids),
            )
            return {row["join_date"]: row["c"] for row in cur.fetchall()}

    _joins_by_date: dict = _build_joins_by_date(frozenset(STAFF_USER_IDS))

    if _joins_by_date:
        _dates_sorted = sorted(_joins_by_date.keys())
        _all_dates = pd.date_range(_dates_sorted[0], _dates_sorted[-1]).date
        _joins_df = pd.DataFrame(
            {
                "Date": [str(d) for d in _all_dates],
                "New Players": [_joins_by_date.get(d, 0) for d in _all_dates],
            }
        )

        _fig_joins = go.Figure()
        _fig_joins.add_trace(
            go.Bar(
                x=_joins_df["Date"],
                y=_joins_df["New Players"],
                marker_color=[
                    (
                        "#D4763B"
                        if (
                            filter_date_start
                            and filter_date_end
                            and filter_date_start
                            <= date.fromisoformat(d)
                            <= filter_date_end
                        )
                        else "#4A90D9"
                    )
                    for d in _joins_df["Date"]
                ],
                name="New Players",
            )
        )
        _fig_joins.update_layout(
            xaxis_title="Date",
            yaxis_title="New Players",
            margin=dict(t=10, b=30),
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
            bargap=0.1,
            xaxis=dict(gridcolor="#1a1a3e"),
            yaxis=dict(gridcolor="#1a1a3e"),
        )

        st.markdown(
            '<div class="section-header"><h4>Daily New Player Joins</h4></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_fig_joins, use_container_width=True)

    # ── Load all player data (cached for 5 minutes) ──────────────────────────
    @st.cache_data(ttl=300, show_spinner="Loading player data…")
    def _load_all_players(
        staff_ids: frozenset,
    ) -> tuple[list, list, list, list]:
        """Returns (all_players_rows, session_durations, location_paths, skipped_staff).

        Pulls every `status='ok'` player's full JSON blob from Postgres in one
        query, then runs the same per-player aggregation logic as before.
        """
        _players = []
        _sessions = []
        _paths = []
        _skipped = []

        with get_conn().cursor() as cur:
            cur.execute(
                "SELECT user_id, username, display_name, data "
                "FROM players WHERE status = 'ok'"
            )
            rows = cur.fetchall()

        for row in rows:
            puid = str(row["user_id"])
            pname = (
                row.get("username") or row.get("display_name") or puid
            )
            if puid in staff_ids:
                _skipped.append(pname)
                continue

            pd_data = row["data"] or {}
            pm = pd_data.get("PlayerMetrics", {})
            pi = pd_data.get("Inventory", {})
            ps = pm.get("sessions", {})
            pf_data = pm.get("fishing", {})
            pe = pm.get("economy", {})
            pc = pm.get("combat", {})
            pt = pm.get("tutorial", {})
            pp = pd_data.get("Purchases", {}).get("permanent", {})
            pfd = pd_data.get("FishDex", {})

            first_join = get_first_join_date(pm)

            for sess in ps.get("history", []):
                _sessions.append(sess.get("duration", 0))

            lp = pm.get("locationPath", [])
            if lp:
                _paths.append({"player": pname, "path": lp})

            visited = pm.get("exploration", {}).get("zonesVisited", {})
            zones_visited = sum(1 for v in visited.values() if v)

            zone_species = pfd.get("zones", {})
            total_mutations = sum(
                z.get("mutationsCount", 0) for z in zone_species.values()
            )

            total_playtime = ps.get("totalPlaytime", 0)
            total_sessions = ps.get("totalCount", 0)
            total_caught = pf_data.get("totalCaught", 0)
            total_sold = pf_data.get("totalSold", 0)
            gold_earned = pe.get("totalGoldEarned", 0)

            _players.append(
                {
                    "Player": pname,
                    "User ID": puid,
                    "First Join": first_join.isoformat() if first_join else "Unknown",
                    "Rod Level": pi.get("bestRodLevel", 0),
                    "Current Gold": pi.get("currencies", {}).get("gold", 0),
                    "Pearls": pi.get("currencies", {}).get("pearls", 0),
                    "Fish Caught": total_caught,
                    "Fish Sold": total_sold,
                    "Species Discovered": pf_data.get("fishDiscovered", 0),
                    "Mutations Found": total_mutations,
                    "Gold Earned": gold_earned,
                    "Gold Spent": pe.get("totalGoldSpent", 0),
                    "Gold from Fish": pf_data.get("totalGoldFromFish", 0),
                    "Gold from Quests": pe.get("totalGoldFromQuests", 0),
                    "Gold on Ship Upgrades": pe.get("goldSpentOnShipUpgrades", 0),
                    "Playtime (s)": total_playtime,
                    "Playtime (min)": round(total_playtime / 60, 1),
                    "Sessions": total_sessions,
                    "Avg Session (s)": round(
                        total_playtime / max(1, total_sessions), 1
                    ),
                    "Longest Session (s)": ps.get("longestSession", 0),
                    "Shortest Session (s)": ps.get("shortestSession", 0),
                    "PvP Kills": pc.get("pvpKills", 0),
                    "Deaths": pc.get("deaths", 0),
                    "Ships Sunk": pc.get("shipsSunk", 0),
                    "Damage Dealt": pc.get("totalDamageDealt", 0),
                    "Floatables Shot": pc.get("floatablesShot", 0),
                    "Last Join Period": ps.get("joinedDuringPeriod", "") or "Unknown",
                    "Zones Visited": zones_visited,
                    "Tutorial Complete": pt.get("completed", False),
                    "Tutorial Step": pt.get("highestStepReached", 0),
                    "Ships Spawned": len(
                        pm.get("exploration", {}).get("shipsSpawned", [])
                    ),
                    "Bag Tier": pp.get("bagTier", 0),
                    "Safe Zone Catches": pf_data.get("caughtInSafeZone", 0),
                    "PvP Zone Catches": pf_data.get("caughtInPvPZone", 0),
                    "Border Catches": pf_data.get("caughtBorderFishing", 0),
                    "Mutation Event Catches": pf_data.get("caughtInMutationEvent", 0),
                    "Gold/hr": round(gold_earned / max(0.01, total_playtime / 3600), 1),
                    "Fish/hr": round(
                        total_caught / max(0.01, total_playtime / 3600), 1
                    ),
                    "Avg Fish Value": round(
                        pf_data.get("totalGoldFromFish", 0) / max(1, total_sold), 1
                    ),
                    "First Fish Time (s)": pf_data.get("firstFishTime", None),
                    "Tutorial Complete Time (s)": pt.get("timeToComplete", None),
                }
            )

        return _players, _sessions, _paths, _skipped

    (
        _all_players_raw,
        all_session_durations_raw,
        all_location_paths_raw,
        skipped_staff,
    ) = _load_all_players(frozenset(STAFF_USER_IDS))

    # Apply date filter (fast — operates on already-loaded data)
    if filter_date_start is not None and filter_date_end is not None:

        def _in_range(row):
            fj = row["First Join"]
            if fj == "Unknown":
                return False
            return filter_date_start <= date.fromisoformat(fj) <= filter_date_end

        _all_players_raw = [r for r in _all_players_raw if _in_range(r)]
        # session durations: we need per-player mapping — rebuild from filtered rows
        # (session durations are already aggregated per player row, so filter paths too)
        _filtered_names = {r["Player"] for r in _all_players_raw}
        all_session_durations = all_session_durations_raw  # kept full for sidebar stats
        all_location_paths = [
            p for p in all_location_paths_raw if p["player"] in _filtered_names
        ]
    else:
        all_session_durations = all_session_durations_raw
        all_location_paths = all_location_paths_raw

    all_players = _all_players_raw
    comp_df = pd.DataFrame(all_players)

    # Show active filter info
    if filter_date_start is not None and filter_date_end is not None:
        if filter_date_start == filter_date_end:
            st.info(
                f"🔍 Showing players who first joined on **{filter_date_start}** — {len(comp_df)} players matched"
            )
        else:
            st.info(
                f"🔍 Showing players who first joined between **{filter_date_start}** and **{filter_date_end}** — {len(comp_df)} players matched"
            )

    if comp_df.empty:
        st.warning(
            "No player data available after filtering. All players may be developers, filtered out by date, or the dumps folder is empty."
        )
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Players:** `0`")
        if skipped_staff:
            st.sidebar.markdown(f"**Staff excluded:** `{', '.join(skipped_staff)}`")
        st.stop()

    # Filter out bugged entries with >1M playtime minutes
    PLAYTIME_MAX_MINUTES = 1_000_000
    bugged = comp_df[comp_df["Playtime (min)"] > PLAYTIME_MAX_MINUTES]
    if len(bugged) > 0:
        comp_df = comp_df[
            comp_df["Playtime (min)"] <= PLAYTIME_MAX_MINUTES
        ].reset_index(drop=True)
        all_session_durations = [
            d for d in all_session_durations if d / 60 <= PLAYTIME_MAX_MINUTES
        ]
        st.warning(
            f"Filtered out {len(bugged)} bugged player(s) with >1M playtime minutes: {', '.join(bugged['Player'].tolist())}"
        )

    n_players = len(comp_df)

    # Tutorial classification (needed for sidebar)
    TOTAL_TUTORIAL_STEPS = 8
    completed_all = comp_df[
        (comp_df["Tutorial Complete"] == True)
        & (comp_df["Tutorial Step"] >= TOTAL_TUTORIAL_STEPS)
    ]
    skipped = comp_df[
        (comp_df["Tutorial Complete"] == True)
        & (comp_df["Tutorial Step"] < TOTAL_TUTORIAL_STEPS)
    ]
    abandoned = comp_df[comp_df["Tutorial Complete"] == False]
    # Players who reached the final step (sail_to_ocean) but never completed
    almost_completed = abandoned[abandoned["Tutorial Step"] >= TOTAL_TUTORIAL_STEPS]
    truly_abandoned = abandoned[abandoned["Tutorial Step"] < TOTAL_TUTORIAL_STEPS]
    # Players who spawned their ship but never completed the tutorial
    spawned_no_tut = comp_df[
        (comp_df["Ships Spawned"] > 0) & (comp_df["Tutorial Complete"] == False)
    ]

    import numpy as np

    # Sidebar: global stats
    st.sidebar.markdown(f"**{len(players_index)} players** loaded")
    if skipped_staff:
        st.sidebar.caption(f"Staff excluded: {', '.join(skipped_staff)}")
    if len(bugged) > 0:
        st.sidebar.caption(f"Bugged filtered: {len(bugged)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Total Players:** `{n_players}`")

    st.sidebar.markdown(
        f"**Tutorial Completed:** `{len(completed_all)}` ({len(completed_all)/max(1,n_players)*100:.0f}%)"
    )
    st.sidebar.markdown(
        f"**Almost Completed:** `{len(almost_completed)}` ({len(almost_completed)/max(1,n_players)*100:.0f}%)"
    )
    st.sidebar.markdown(
        f"**Tutorial Skipped:** `{len(skipped)}` ({len(skipped)/max(1,n_players)*100:.0f}%)"
    )
    st.sidebar.markdown(
        f"**Left Game:** `{len(truly_abandoned)}` ({len(truly_abandoned)/max(1,n_players)*100:.0f}%)"
    )
    st.sidebar.markdown(
        f"**Spawned Ship (No Tutorial):** `{len(spawned_no_tut)}` ({len(spawned_no_tut)/max(1,n_players)*100:.0f}%)"
    )

    st.sidebar.markdown("---")
    total_sessions = int(comp_df["Sessions"].sum())
    st.sidebar.markdown(f"**Total Sessions:** `{total_sessions:,}`")
    if all_session_durations:
        sess_arr = np.array(all_session_durations)
        st.sidebar.markdown(f"**Avg Session:** `{fmt_time(np.mean(sess_arr))}`")
        st.sidebar.markdown(f"**Median Session:** `{fmt_time(np.median(sess_arr))}`")

    st.sidebar.markdown("---")
    total_fish = int(comp_df["Fish Caught"].sum())
    st.sidebar.markdown(f"**Total Fish Caught:** `{total_fish:,}`")
    st.sidebar.markdown(f"**Avg Fish Caught:** `{comp_df['Fish Caught'].mean():.1f}`")
    st.sidebar.markdown(
        f"**Median Fish Caught:** `{comp_df['Fish Caught'].median():.0f}`"
    )

    st.sidebar.markdown("---")
    total_floatables = int(comp_df["Floatables Shot"].sum())
    st.sidebar.markdown(f"**Total Floatables Shot:** `{total_floatables:,}`")
    st.sidebar.markdown(
        f"**Avg Floatables / Player:** `{comp_df['Floatables Shot'].mean():.1f}`"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Last Join Period**")
    _period_order = ["Morning", "Afternoon", "Dusk", "Dark Night", "Unknown"]
    _period_counts = comp_df["Last Join Period"].value_counts()
    for _p in _period_order:
        _c = int(_period_counts.get(_p, 0))
        if _c == 0:
            continue
        _pct = _c / max(1, n_players) * 100
        st.sidebar.markdown(f"- {_p}: `{_c}` ({_pct:.0f}%)")

    st.markdown("")

    # ── Scatter Explorer ────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><h4>Scatter Explorer</h4></div>',
        unsafe_allow_html=True,
    )

    _NUMERIC_COLS = [
        "Rod Level",
        "Playtime (min)",
        "Sessions",
        "Avg Session (s)",
        "Longest Session (s)",
        "Shortest Session (s)",
        "Fish Caught",
        "Fish Sold",
        "Species Discovered",
        "Mutations Found",
        "Gold Earned",
        "Gold Spent",
        "Current Gold",
        "Gold from Fish",
        "Gold from Quests",
        "Gold on Ship Upgrades",
        "Gold/hr",
        "Fish/hr",
        "Avg Fish Value",
        "PvP Kills",
        "Deaths",
        "Ships Sunk",
        "Damage Dealt",
        "Zones Visited",
        "Tutorial Step",
        "Ships Spawned",
        "Bag Tier",
        "Safe Zone Catches",
        "PvP Zone Catches",
        "Border Catches",
        "Mutation Event Catches",
        "Floatables Shot",
    ]
    _COLOR_COLS = [
        "None",
        "Tutorial Complete",
        "Zones Visited",
        "Bag Tier",
        "Ships Spawned",
    ]
    _DISCRETE_COLS = {
        "Rod Level",
        "Tutorial Step",
        "Zones Visited",
        "Bag Tier",
        "Ships Spawned",
        "Sessions",
        "PvP Kills",
        "Deaths",
        "Ships Sunk",
    }

    _avail_num = [c for c in _NUMERIC_COLS if c in comp_df.columns]
    _avail_color = [c for c in _COLOR_COLS if c == "None" or c in comp_df.columns]

    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    with _sc1:
        _x_col = st.selectbox(
            "X axis",
            _avail_num,
            index=(
                _avail_num.index("Playtime (min)")
                if "Playtime (min)" in _avail_num
                else 0
            ),
            key="scatter_x",
        )
    with _sc2:
        _y_col = st.selectbox(
            "Y axis",
            _avail_num,
            index=_avail_num.index("Fish Caught") if "Fish Caught" in _avail_num else 1,
            key="scatter_y",
        )
    with _sc3:
        _color_col = st.selectbox(
            "Color by", _avail_color, index=0, key="scatter_color"
        )
    with _sc4:
        _show_trendline = st.checkbox("Trendline", value=False, key="scatter_trend")

    _x_is_discrete = _x_col in _DISCRETE_COLS or (
        _x_col in comp_df.columns and comp_df[_x_col].nunique() <= 15
    )
    _use_box = False
    if _x_is_discrete:
        _use_box = st.toggle("Candlestick / box plot", value=False, key="scatter_box")

    _scatter_df = comp_df.copy()
    _color_arg = None
    _color_map = None

    if _color_col != "None":
        if _scatter_df[_color_col].dtype == bool:
            _scatter_df[_color_col] = _scatter_df[_color_col].map(
                {True: "Yes", False: "No"}
            )
            _color_map = {"Yes": "#4CAF50", "No": "#F44336"}
        _scatter_df[_color_col] = _scatter_df[_color_col].astype(str)
        _color_arg = _color_col

    _layout_common = dict(
        xaxis_title=_x_col,
        yaxis_title=_y_col,
        margin=dict(t=20, b=20),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccd6f6",
        xaxis=dict(gridcolor="#1a1a3e"),
        yaxis=dict(gridcolor="#1a1a3e"),
        legend=dict(orientation="h", y=1.05),
    )

    if _use_box:
        _x_vals = sorted(_scatter_df[_x_col].unique())
        _fig_scatter = go.Figure()
        for _xv in _x_vals:
            _group = _scatter_df[_scatter_df[_x_col] == _xv][_y_col].dropna()
            if len(_group) == 0:
                continue
            _q1, _med, _q3 = np.percentile(_group, [25, 50, 75])
            _mn, _mx = _group.min(), _group.max()
            _fig_scatter.add_trace(
                go.Candlestick(
                    x=[str(_xv)],
                    open=[_q1],
                    high=[_mx],
                    low=[_mn],
                    close=[_q3],
                    name=str(_xv),
                    increasing_line_color="#4CAF50",
                    decreasing_line_color="#4CAF50",
                    whiskerwidth=0.6,
                )
            )
            _fig_scatter.add_trace(
                go.Scatter(
                    x=[str(_xv)],
                    y=[_med],
                    mode="markers",
                    marker=dict(
                        color="#FFD700",
                        size=8,
                        symbol="line-ew",
                        line=dict(width=2, color="#FFD700"),
                    ),
                    showlegend=False,
                    hovertemplate=f"<b>{_x_col}={_xv}</b><br>median={_med:.1f}<br>Q1={_q1:.1f} Q3={_q3:.1f}<br>min={_mn:.1f} max={_mx:.1f}<extra></extra>",
                )
            )
        _layout_common["xaxis_type"] = "category"
        _layout_common["showlegend"] = False
        _layout_common["xaxis_rangeslider_visible"] = False
        _fig_scatter.update_layout(**_layout_common)
    else:
        _fig_scatter = px.scatter(
            _scatter_df,
            x=_x_col,
            y=_y_col,
            color=_color_arg,
            color_discrete_map=_color_map,
            hover_name="Player",
            hover_data={_x_col: True, _y_col: True},
            color_discrete_sequence=px.colors.qualitative.Set2,
            trendline="ols" if (_show_trendline and _color_arg is None) else None,
        )
        _fig_scatter.update_traces(marker=dict(size=10, opacity=0.85))
        _fig_scatter.update_layout(**_layout_common)

    st.plotly_chart(_fig_scatter, use_container_width=True)

    # ── Aggregate Stats Table (Mean / Median / Min / Max) ───────────────────
    st.markdown(
        '<div class="section-header"><h4>Key Metrics — Mean / Median / Min / Max</h4></div>',
        unsafe_allow_html=True,
    )

    agg_metrics = [
        ("Session Length (s)", all_session_durations if all_session_durations else [0]),
        ("Playtime (min)", comp_df["Playtime (min)"].tolist()),
        ("Sessions per Player", comp_df["Sessions"].tolist()),
        ("Fish Caught", comp_df["Fish Caught"].tolist()),
        ("Fish Sold", comp_df["Fish Sold"].tolist()),
        ("Species Discovered", comp_df["Species Discovered"].tolist()),
        ("Gold Earned", comp_df["Gold Earned"].tolist()),
        ("Gold Spent", comp_df["Gold Spent"].tolist()),
        ("Current Gold", comp_df["Current Gold"].tolist()),
        ("Avg Fish Value", comp_df["Avg Fish Value"].tolist()),
        ("Gold/hr", comp_df["Gold/hr"].tolist()),
        ("Fish/hr", comp_df["Fish/hr"].tolist()),
        ("Rod Level", comp_df["Rod Level"].tolist()),
        ("Zones Visited", comp_df["Zones Visited"].tolist()),
        ("PvP Kills", comp_df["PvP Kills"].tolist()),
        ("Deaths", comp_df["Deaths"].tolist()),
        ("Tutorial Step Reached", comp_df["Tutorial Step"].tolist()),
    ]

    import numpy as np

    agg_rows = []
    for label, values in agg_metrics:
        arr = np.array(values, dtype=float)
        agg_rows.append(
            {
                "Metric": label,
                "Mean": f"{np.mean(arr):.1f}",
                "Median": f"{np.median(arr):.1f}",
                "Min": f"{np.min(arr):.1f}",
                "Max": f"{np.max(arr):.1f}",
                "Std Dev": f"{np.std(arr):.1f}",
                "N": len(arr),
            }
        )

    agg_df = pd.DataFrame(agg_rows)
    st.dataframe(agg_df, use_container_width=True, hide_index=True)

    # ── World Map ────────────────────────────────────────────────────────────
    st.info(
        "🗺️ The full player paths map has moved to its own page — use the **🗺️ World Map** link in the sidebar."
    )

    # ── Tutorial Step Dropout ────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><h4>Tutorial Progression & Dropout</h4></div>',
        unsafe_allow_html=True,
    )

    TUTORIAL_STEPS = [
        (1, "catch_fish"),
        (2, "go_to_fish_seller"),
        (3, "confirm_sell"),
        (4, "go_to_rod_shop"),
        (5, "upgrade_rod"),
        (6, "go_to_ship_spawn"),
        (7, "spawn_ship"),
        (8, "sail_to_ocean"),
    ]

    st.markdown(
        """
    > **Note:** Currently both "Skip Tutorial" and "Complete Tutorial" call the same `CompleteTutorial()`.
    > Players who skipped are identified here as `completed = true` but `highestStepReached < 8`.
    > To get cleaner data, consider tracking a `tutorial.skippedAtStep` field separately.
    """
    )

    cat_cols = st.columns(4)
    with cat_cols[0]:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>👥 Total Players</h3>
            <p>{n_players}</p>
            <div class="sub">in dataset</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with cat_cols[1]:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>✅ Completed All Steps</h3>
            <p>{len(completed_all)}</p>
            <div class="sub">{len(completed_all)/max(1,n_players)*100:.0f}% of players</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with cat_cols[2]:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>⏭️ Skipped Tutorial</h3>
            <p>{len(skipped)}</p>
            <div class="sub">{len(skipped)/max(1,n_players)*100:.0f}% of players</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with cat_cols[3]:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>🚪 Left Game</h3>
            <p>{len(truly_abandoned)}</p>
            <div class="sub">{len(truly_abandoned)/max(1,n_players)*100:.0f}% of players</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Spawned Ship Without Completing Tutorial ─────────────────────────────
    st.markdown(
        '<div class="section-header"><h4>⚓ Spawned Ship Without Completing Tutorial</h4></div>',
        unsafe_allow_html=True,
    )

    spawn_cols = st.columns(4)
    with spawn_cols[0]:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>⚓ Spawned Ship, No Tutorial</h3>
            <p>{len(spawned_no_tut)}</p>
            <div class="sub">{len(spawned_no_tut)/max(1,n_players)*100:.0f}% of players</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with spawn_cols[1]:
        avg_step = (
            spawned_no_tut["Tutorial Step"].mean() if len(spawned_no_tut) > 0 else 0
        )
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>📍 Avg Step Reached</h3>
            <p>{avg_step:.1f}</p>
            <div class="sub">of {TOTAL_TUTORIAL_STEPS} steps</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with spawn_cols[2]:
        avg_ships = (
            spawned_no_tut["Ships Spawned"].mean() if len(spawned_no_tut) > 0 else 0
        )
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>🚢 Avg Ships Spawned</h3>
            <p>{avg_ships:.1f}</p>
            <div class="sub">per player in this group</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with spawn_cols[3]:
        avg_playtime = (
            spawned_no_tut["Playtime (min)"].mean() if len(spawned_no_tut) > 0 else 0
        )
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>⏱️ Avg Playtime</h3>
            <p>{avg_playtime:.1f}m</p>
            <div class="sub">for players in this group</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    if len(spawned_no_tut) > 0:
        display_cols = [
            "Player",
            "User ID",
            "Tutorial Step",
            "Ships Spawned",
            "Playtime (min)",
            "Fish Caught",
            "Gold Earned",
            "Sessions",
        ]
        st.dataframe(
            spawned_no_tut[display_cols]
            .sort_values("Tutorial Step", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No players spawned their ship without completing the tutorial.")

    st.markdown("")

    # ── Combined Stacked Bar: at each step, show how many are still going vs skipped vs abandoned
    step_labels = [
        f"{num}. {sid.replace('_', ' ').title()}" for num, sid in TUTORIAL_STEPS
    ]

    still_going = []  # reached this step and didn't skip here
    skipped_at = []  # skipped at exactly this step
    abandoned_at = []  # last step was this, didn't complete

    for step_num, step_id in TUTORIAL_STEPS:
        # Abandoned at this step: tutorial not complete, highest step = this step
        aband = int(((abandoned["Tutorial Step"] == step_num)).sum())
        abandoned_at.append(aband)
        # Skipped at this step: tutorial complete, highest step = this step, and step < 12
        skip = int(((skipped["Tutorial Step"] == step_num)).sum())
        skipped_at.append(skip)
        # Still progressing: reached this step or beyond (total minus cumulative dropouts)
        reached = int((comp_df["Tutorial Step"] >= step_num).sum())
        still_going.append(reached)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Reached Step",
            x=step_labels,
            y=still_going,
            marker_color="#4CAF50",
            opacity=0.85,
        )
    )

    st.markdown("**How many players reached each step:**")

    fig.update_layout(
        xaxis_title="Tutorial Step",
        yaxis_title="Players",
        margin=dict(t=30, b=130),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccd6f6",
        bargap=0.12,
        xaxis=dict(gridcolor="#1a1a3e", tickangle=-45),
        yaxis=dict(gridcolor="#1a1a3e"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Where Players Drop Off (skipped vs abandoned) ────────────────────────
    st.markdown("**Where players stopped — skipped vs left game:**")

    # Only show steps where someone actually dropped
    drop_rows = []
    for i, (step_num, step_id) in enumerate(TUTORIAL_STEPS):
        if skipped_at[i] > 0 or abandoned_at[i] > 0:
            drop_rows.append(
                {
                    "Step": f"{step_num}. {step_id.replace('_', ' ').title()}",
                    "Skipped Here": skipped_at[i],
                    "Left Game Here": abandoned_at[i],
                    "Total Lost": skipped_at[i] + abandoned_at[i],
                }
            )

    if drop_rows:
        drop_df = pd.DataFrame(drop_rows)

        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                name="⏭️ Skipped",
                x=drop_df["Step"],
                y=drop_df["Skipped Here"],
                marker_color="#FF9800",
            )
        )
        fig2.add_trace(
            go.Bar(
                name="🚪 Left Game",
                x=drop_df["Step"],
                y=drop_df["Left Game Here"],
                marker_color="#F44336",
            )
        )
        fig2.update_layout(
            barmode="stack",
            xaxis_title="Step Where Player Stopped",
            yaxis_title="Players",
            margin=dict(t=30, b=130),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
            bargap=0.15,
            xaxis=dict(gridcolor="#1a1a3e", tickangle=-45),
            yaxis=dict(gridcolor="#1a1a3e"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(drop_df, use_container_width=True, hide_index=True)
    else:
        st.info("No tutorial dropouts recorded")

    # ── Per-step dropout rate table ──────────────────────────────────────────
    st.markdown("**Step-by-step dropout rate:**")
    dropout_rows = []
    for i in range(len(TUTORIAL_STEPS) - 1):
        prev = still_going[i]
        curr = still_going[i + 1]
        lost = prev - curr
        pct = (lost / max(1, prev)) * 100
        dropout_rows.append(
            {
                "From": step_labels[i],
                "To": step_labels[i + 1],
                "Lost": lost,
                "Dropout %": f"{pct:.1f}%",
                "Remaining": curr,
            }
        )
    dropout_df = pd.DataFrame(dropout_rows)
    st.dataframe(dropout_df, use_container_width=True, hide_index=True)

    # ── First-Fish Time & Tutorial Completion Time distributions ─────────────
    st.markdown(
        '<div class="section-header"><h4>Time-to-First-Fish & Time-to-Complete-Tutorial</h4></div>',
        unsafe_allow_html=True,
    )

    _fft_col = "First Fish Time (s)"
    _tct_col = "Tutorial Complete Time (s)"

    _fft_valid = (
        comp_df[_fft_col].dropna()
        if _fft_col in comp_df.columns
        else pd.Series([], dtype=float)
    )
    _tct_valid = (
        comp_df[_tct_col].dropna()
        if _tct_col in comp_df.columns
        else pd.Series([], dtype=float)
    )
    _fft_valid = _fft_valid[_fft_valid > 0].values.astype(float)
    _tct_valid = _tct_valid[_tct_valid > 0].values.astype(float)

    # ── Summary stat cards ────────────────────────────────────────────────────
    _stat_cols = st.columns(8)

    def _pct_within(arr, secs):
        return 100.0 * (arr <= secs).sum() / len(arr) if len(arr) else 0.0

    if len(_fft_valid):
        _stat_cols[0].metric("First Fish — N", len(_fft_valid))
        _stat_cols[1].metric("Median", fmt_time(float(np.median(_fft_valid))))
        _stat_cols[2].metric("≤ 30s", f"{_pct_within(_fft_valid, 30):.0f}%")
        _stat_cols[3].metric("≤ 60s", f"{_pct_within(_fft_valid, 60):.0f}%")
    if len(_tct_valid):
        _stat_cols[4].metric("Tutorial — N", len(_tct_valid))
        _stat_cols[5].metric("Median", fmt_time(float(np.median(_tct_valid))))
        _stat_cols[6].metric("≤ 3 min", f"{_pct_within(_tct_valid, 180):.0f}%")
        _stat_cols[7].metric("≤ 5 min", f"{_pct_within(_tct_valid, 300):.0f}%")

    st.markdown("")

    # ── Violin + box plots side by side ──────────────────────────────────────
    _viol_c1, _viol_c2 = st.columns(2)

    with _viol_c1:
        if len(_fft_valid):
            # Cap display at 99th pct so outliers don't squash everything
            _fft_cap = min(float(np.percentile(_fft_valid, 99)), 300)
            _fft_plot = _fft_valid[_fft_valid <= _fft_cap]
            _fft_outliers = _fft_valid[_fft_valid > _fft_cap]

            fig_fft_v = go.Figure()
            fig_fft_v.add_trace(
                go.Violin(
                    y=_fft_plot,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    pointpos=-1.5,
                    jitter=0.3,
                    fillcolor="rgba(76,175,80,0.25)",
                    line_color="#4CAF50",
                    marker=dict(size=4, color="#4CAF50", opacity=0.5),
                    name="First Fish",
                    hovertemplate="%{y:.0f}s<extra></extra>",
                )
            )
            if len(_fft_outliers):
                fig_fft_v.add_trace(
                    go.Scatter(
                        x=[0] * len(_fft_outliers),
                        y=_fft_outliers,
                        mode="markers",
                        marker=dict(size=5, color="#F44336", symbol="x", opacity=0.7),
                        name=f"{len(_fft_outliers)} outlier(s) >{_fft_cap:.0f}s",
                        hovertemplate="%{y:.0f}s<extra></extra>",
                    )
                )
            # Reference lines at 30s and 60s
            for _ref, _col, _lbl in [(30, "#FFD700", "30s"), (60, "#FF9800", "60s")]:
                fig_fft_v.add_hline(
                    y=_ref,
                    line=dict(color=_col, dash="dash", width=1.5),
                    annotation_text=_lbl,
                    annotation_position="right",
                    annotation_font=dict(color=_col, size=11),
                )
            fig_fft_v.update_layout(
                title=f"Time to First Fish  (n={len(_fft_valid)})",
                yaxis_title="Seconds",
                xaxis=dict(showticklabels=False),
                margin=dict(t=40, b=20),
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                yaxis=dict(gridcolor="#1a1a3e"),
                legend=dict(orientation="h", y=-0.05),
                showlegend=len(_fft_outliers) > 0,
            )
            st.plotly_chart(fig_fft_v, use_container_width=True)
        else:
            st.info("No firstFishTime data yet")

    with _viol_c2:
        if len(_tct_valid):
            _tct_cap = min(float(np.percentile(_tct_valid, 99)), 1200)
            _tct_plot = _tct_valid[_tct_valid <= _tct_cap]
            _tct_outliers = _tct_valid[_tct_valid > _tct_cap]

            fig_tct_v = go.Figure()
            fig_tct_v.add_trace(
                go.Violin(
                    y=_tct_plot,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    pointpos=-1.5,
                    jitter=0.3,
                    fillcolor="rgba(212,118,59,0.25)",
                    line_color="#D4763B",
                    marker=dict(size=4, color="#D4763B", opacity=0.5),
                    name="Tutorial",
                    hovertemplate="%{y:.0f}s<extra></extra>",
                )
            )
            if len(_tct_outliers):
                fig_tct_v.add_trace(
                    go.Scatter(
                        x=[0] * len(_tct_outliers),
                        y=_tct_outliers,
                        mode="markers",
                        marker=dict(size=5, color="#F44336", symbol="x", opacity=0.7),
                        name=f"{len(_tct_outliers)} outlier(s) >{_tct_cap:.0f}s",
                        hovertemplate="%{y:.0f}s<extra></extra>",
                    )
                )
            for _ref, _col, _lbl in [
                (60, "#4CAF50", "1 min"),
                (180, "#FFD700", "3 min"),
                (300, "#FF9800", "5 min"),
            ]:
                fig_tct_v.add_hline(
                    y=_ref,
                    line=dict(color=_col, dash="dash", width=1.5),
                    annotation_text=_lbl,
                    annotation_position="right",
                    annotation_font=dict(color=_col, size=11),
                )
            fig_tct_v.update_layout(
                title=f"Time to Complete Tutorial  (n={len(_tct_valid)})",
                yaxis_title="Seconds",
                xaxis=dict(showticklabels=False),
                margin=dict(t=40, b=20),
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                yaxis=dict(gridcolor="#1a1a3e"),
                legend=dict(orientation="h", y=-0.05),
                showlegend=len(_tct_outliers) > 0,
            )
            st.plotly_chart(fig_tct_v, use_container_width=True)
        else:
            st.info("No tutorial completion time data yet")

    # ── Percentile table ─────────────────────────────────────────────────────
    _pct_levels = [10, 25, 50, 75, 90, 95, 99]
    _pct_rows = []
    for _p in _pct_levels:
        _row = {"Percentile": f"P{_p}"}
        if len(_fft_valid):
            _row["First Fish"] = fmt_time(float(np.percentile(_fft_valid, _p)))
        if len(_tct_valid):
            _row["Tutorial Complete"] = fmt_time(float(np.percentile(_tct_valid, _p)))
        _pct_rows.append(_row)
    if _pct_rows and len(_pct_rows[0]) > 1:
        st.dataframe(pd.DataFrame(_pct_rows), use_container_width=True, hide_index=True)

    # ── Combined ECDF (cumulative % by X seconds) ─────────────────────────────
    _ecdf_has_data = len(_fft_valid) > 0 or len(_tct_valid) > 0
    if _ecdf_has_data:
        st.markdown("**Cumulative % of players who hit each milestone by X seconds:**")
        fig_ecdf = go.Figure()

        if len(_fft_valid):
            _s_fft = np.sort(_fft_valid)
            _x_fft = np.linspace(0, min(float(_s_fft[-1]), 300), 600)
            _y_fft = 100.0 * np.searchsorted(_s_fft, _x_fft, side="right") / len(_s_fft)
            fig_ecdf.add_trace(
                go.Scatter(
                    x=_x_fft,
                    y=_y_fft,
                    mode="lines",
                    name="Caught First Fish",
                    line=dict(color="#4CAF50", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(76,175,80,0.08)",
                    hovertemplate="%{y:.1f}% within %{x:.0f}s<extra>First Fish</extra>",
                )
            )

        if len(_tct_valid):
            _s_tct = np.sort(_tct_valid)
            _x_tct = np.linspace(0, min(float(_s_tct[-1]), 900), 600)
            _y_tct = 100.0 * np.searchsorted(_s_tct, _x_tct, side="right") / len(_s_tct)
            fig_ecdf.add_trace(
                go.Scatter(
                    x=_x_tct,
                    y=_y_tct,
                    mode="lines",
                    name="Completed Tutorial",
                    line=dict(color="#D4763B", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(212,118,59,0.08)",
                    hovertemplate="%{y:.1f}% within %{x:.0f}s<extra>Tutorial</extra>",
                )
            )

        for _ref, _lbl, _col in [
            (30, "30s", "#4CAF50"),
            (60, "1 min", "#FFD700"),
            (120, "2 min", "#FF9800"),
            (300, "5 min", "#F44336"),
        ]:
            fig_ecdf.add_vline(
                x=_ref,
                line=dict(color=_col, dash="dot", width=1),
                annotation_text=_lbl,
                annotation_position="top right",
                annotation_font=dict(color=_col, size=10),
            )

        fig_ecdf.update_layout(
            xaxis_title="Seconds into Session",
            yaxis_title="Cumulative % of Players",
            yaxis=dict(range=[0, 100], gridcolor="#1a1a3e"),
            xaxis=dict(gridcolor="#1a1a3e"),
            margin=dict(t=20, b=20),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_ecdf, use_container_width=True)

    # ── Scatter: first fish vs tutorial time ──────────────────────────────────
    if len(_fft_valid) > 0 and len(_tct_valid) > 0:
        _both = comp_df[[_fft_col, _tct_col, "Player", "Tutorial Step"]].dropna()
        _both = _both[(_both[_fft_col] > 0) & (_both[_tct_col] > 0)]
        if len(_both) > 1:
            st.markdown(
                "**First Fish Time vs Tutorial Complete Time** (each dot = one player):"
            )
            fig_scat = px.scatter(
                _both,
                x=_fft_col,
                y=_tct_col,
                hover_name="Player",
                color="Tutorial Step",
                color_continuous_scale=["#1a1a3e", "#4A90D9", "#FFD700"],
                labels={_fft_col: "First Fish (s)", _tct_col: "Tutorial Complete (s)"},
                trendline="ols",
            )
            fig_scat.update_traces(
                marker=dict(size=8, opacity=0.8), selector=dict(mode="markers")
            )
            fig_scat.update_layout(
                margin=dict(t=20, b=20),
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                xaxis=dict(gridcolor="#1a1a3e"),
                yaxis=dict(gridcolor="#1a1a3e"),
                coloraxis_colorbar=dict(title="Step"),
            )
            st.plotly_chart(fig_scat, use_container_width=True)

    # ── Playtime Retention Curve ─────────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><h4>Playtime Retention Curve</h4></div>',
        unsafe_allow_html=True,
    )

    _playtimes_s = comp_df["Playtime (s)"].values
    if len(_playtimes_s) > 0:
        _max_s = int(
            np.percentile(_playtimes_s, 99)
        )  # cap at 99th pct to avoid huge x-axis
        _x_pts = np.linspace(0, _max_s, 500).astype(int)
        # Vectorised: sort once, use searchsorted — O(n log n) vs O(500n) loop
        _sorted_asc = np.sort(_playtimes_s)
        _pct_retained = (
            100.0
            * (len(_sorted_asc) - np.searchsorted(_sorted_asc, _x_pts, side="left"))
            / len(_sorted_asc)
        )

        _retention_fig = go.Figure()
        _retention_fig.add_trace(
            go.Scatter(
                x=_x_pts / 60,
                y=_pct_retained,
                mode="lines",
                line=dict(color="#4A90D9", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(74,144,217,0.15)",
                name="% still playing",
                hovertemplate="%{y:.1f}% played ≥ %{x:.1f} min<extra></extra>",
            )
        )

        # Vertical line at 5 minutes
        _five_min_pct = 100.0 * np.sum(_playtimes_s >= 300) / len(_playtimes_s)
        _retention_fig.add_vline(
            x=5,
            line=dict(color="#FFD700", width=2, dash="dash"),
            annotation_text=f"5 min — {_five_min_pct:.1f}% still playing",
            annotation_position="top right",
            annotation_font=dict(color="#FFD700", size=12),
        )

        _retention_fig.update_layout(
            xaxis_title="Session Playtime (minutes)",
            yaxis_title="% of Players",
            yaxis=dict(range=[0, 100], gridcolor="#1a1a3e"),
            xaxis=dict(gridcolor="#1a1a3e"),
            margin=dict(t=20, b=20),
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
        )
        st.plotly_chart(_retention_fig, use_container_width=True)
        st.caption(
            f"{_five_min_pct:.1f}% of players ({int(np.sum(_playtimes_s >= 300))}/{len(_playtimes_s)}) played for at least 5 minutes"
        )
    else:
        st.info("No playtime data available")

    # ── Fishing Location Breakdown (aggregate) ───────────────────────────────
    st.markdown(
        '<div class="section-header"><h4>Where Players Fish (Aggregate)</h4></div>',
        unsafe_allow_html=True,
    )

    agg_fishing_loc = {
        "Safe Zone": int(comp_df["Safe Zone Catches"].sum()),
        "PvP Zone": int(comp_df["PvP Zone Catches"].sum()),
        "Border": int(comp_df["Border Catches"].sum()),
        "Mutation Event": int(comp_df["Mutation Event Catches"].sum()),
    }
    agg_fl_df = pd.DataFrame(
        list(agg_fishing_loc.items()), columns=["Location", "Total Catches"]
    )
    agg_fl_df = agg_fl_df[agg_fl_df["Total Catches"] > 0]

    if not agg_fl_df.empty:
        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig = px.pie(
                agg_fl_df,
                values="Total Catches",
                names="Location",
                color_discrete_sequence=["#4CAF50", "#F44336", "#FF9800", "#9C27B0"],
            )
            fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_bar:
            fig = px.bar(
                agg_fl_df,
                x="Location",
                y="Total Catches",
                color="Location",
                color_discrete_map={
                    "Safe Zone": "#4CAF50",
                    "PvP Zone": "#F44336",
                    "Border": "#FF9800",
                    "Mutation Event": "#9C27B0",
                },
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(t=20, b=20),
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fishing location data")

    # ── Full Player Table ────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><h4>All Players — Full Data</h4></div>',
        unsafe_allow_html=True,
    )

    display_cols = [
        "Player",
        "First Join",
        "Rod Level",
        "Current Gold",
        "Fish Caught",
        "Fish Sold",
        "Species Discovered",
        "Gold Earned",
        "Gold Spent",
        "Playtime (min)",
        "Sessions",
        "Avg Session (s)",
        "Gold/hr",
        "Fish/hr",
        "Avg Fish Value",
        "PvP Kills",
        "Deaths",
        "Ships Sunk",
        "Zones Visited",
        "Tutorial Complete",
    ]
    display_df = comp_df[[c for c in display_cols if c in comp_df.columns]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
