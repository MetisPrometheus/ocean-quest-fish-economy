"""
Fish Economy Tester - Streamlit App
====================================
Run with: streamlit run fish_economy_app.py

Put this file in the same folder as FishData.lua
"""

import streamlit as st
import pandas as pd
import math
import random
import os
from collections import defaultdict

# ============================================================
# PAGE CONFIG - Must be first Streamlit command
# ============================================================
st.set_page_config(page_title="Fish Economy Tester", page_icon="🐟", layout="wide")

# ============================================================
# CONFIG
# ============================================================

ZONE_CONFIG = {
    "Pirate": {"rod_min": 1, "rod_max": 6, "color": "#8B4513"},
    "Greek": {"rod_min": 7, "rod_max": 15, "color": "#4169E1"},
    "Japanese": {"rod_min": 16, "rod_max": 24, "color": "#DC143C"},
    "Viking": {"rod_min": 25, "rod_max": 33, "color": "#228B22"},
}

MAIN_ZONES = ["Pirate", "Greek", "Japanese", "Viking"]
RARITIES = ["Poor", "Common", "Uncommon", "Rare", "Epic", "Legendary", "Mythic"]

RARITY_COLORS = {
    "Poor": "#808080",
    "Common": "#FFFFFF",
    "Uncommon": "#00FF00",
    "Rare": "#0080FF",
    "Epic": "#AA00FF",
    "Legendary": "#FFD700",
    "Mythic": "#FF4444",
}

GAUSSIAN_GAMMA = 1.4
GAUSSIAN_SIGMA_MIN = 0.10
GAUSSIAN_SIGMA_MAX = 0.40


# ============================================================
# DATA LOADING
# ============================================================


def parse_rarity(rarity_str: str) -> str:
    """Convert POOR/Poor/poor to 'Poor' format"""
    # Remove any whitespace and convert to title case
    return rarity_str.strip().capitalize()


def load_fishdata_from_file(filepath: str) -> dict:
    """Parse FishData.lua file and return fish info dict."""
    fish_data = {}
    current_zone = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("-- //"):
                current_zone = line.replace("-- //", "").strip()
                continue

            if line.startswith("[") and "item_id" in line:
                try:
                    fish_id = int(line.split("]")[0].replace("[", ""))

                    name_start = line.find('name = "') + 8
                    name_end = line.find('"', name_start)
                    name = line[name_start:name_end]

                    # Fixed: handle UPPERCASE rarities like POOR, COMMON, etc.
                    rarity_start = line.find("Rarity.") + 7
                    rarity_end = line.find(",", rarity_start)
                    rarity_raw = line[rarity_start:rarity_end].strip()
                    rarity = parse_rarity(rarity_raw)  # POOR -> Poor

                    weight_start = line.find("mean_weight = ") + 14
                    weight_end = line.find(",", weight_start)
                    mean_weight = float(line[weight_start:weight_end])

                    value_start = line.find("base_value = ") + 13
                    value_end = line.find(" }", value_start)
                    if value_end == -1:
                        value_end = line.find("}", value_start)
                    base_value = int(line[value_start:value_end].strip())

                    fish_data[fish_id] = {
                        "id": fish_id,
                        "name": name.replace("_", " "),
                        "rarity": rarity,
                        "mean_weight": mean_weight,
                        "base_value": base_value,
                        "zone": current_zone,
                    }
                except (ValueError, IndexError) as e:
                    continue

    return fish_data


def load_fishdata_from_string(content: str) -> dict:
    """Parse FishData.lua content string and return fish info dict."""
    fish_data = {}
    current_zone = None

    for line in content.split("\n"):
        line = line.strip()

        if line.startswith("-- //"):
            current_zone = line.replace("-- //", "").strip()
            continue

        if line.startswith("[") and "item_id" in line:
            try:
                fish_id = int(line.split("]")[0].replace("[", ""))

                name_start = line.find('name = "') + 8
                name_end = line.find('"', name_start)
                name = line[name_start:name_end]

                # Fixed: handle UPPERCASE rarities
                rarity_start = line.find("Rarity.") + 7
                rarity_end = line.find(",", rarity_start)
                rarity_raw = line[rarity_start:rarity_end].strip()
                rarity = parse_rarity(rarity_raw)

                weight_start = line.find("mean_weight = ") + 14
                weight_end = line.find(",", weight_start)
                mean_weight = float(line[weight_start:weight_end])

                value_start = line.find("base_value = ") + 13
                value_end = line.find(" }", value_start)
                if value_end == -1:
                    value_end = line.find("}", value_start)
                base_value = int(line[value_start:value_end].strip())

                fish_data[fish_id] = {
                    "id": fish_id,
                    "name": name.replace("_", " "),
                    "rarity": rarity,
                    "mean_weight": mean_weight,
                    "base_value": base_value,
                    "zone": current_zone,
                }
            except (ValueError, IndexError):
                continue

    return fish_data


def get_zone_ranges(fish_data: dict) -> dict:
    """Get min/max fish IDs for each zone."""
    zone_fish = defaultdict(list)
    for fid, info in fish_data.items():
        zone_fish[info["zone"]].append(fid)

    return {z: (min(ids), max(ids)) for z, ids in zone_fish.items() if ids}


def get_parent_zone(zone_name: str) -> str:
    """Get parent zone for subzones."""
    for main in MAIN_ZONES:
        if zone_name == main or zone_name.startswith(main + "Sub_"):
            return main
    return zone_name


# ============================================================
# ECONOMY CALCULATIONS
# ============================================================


def get_catch_time(
    rod_level: int, base: float, per_rod: float, minimum: float
) -> float:
    """Calculate time per catch based on rod level."""
    time = base + (rod_level * per_rod)
    return max(time, minimum)


def catches_per_hour(
    rod_level: int, base: float, per_rod: float, minimum: float
) -> float:
    """Calculate catches per hour."""
    return 3600 / get_catch_time(rod_level, base, per_rod, minimum)


def compute_drop_probs(
    fish_data: dict, zone_ranges: dict, zone: str, rod_level: int
) -> dict:
    """Calculate drop probabilities for each fish in a zone."""
    if zone not in zone_ranges:
        return {}

    zmin, zmax = zone_ranges[zone]
    parent = get_parent_zone(zone)
    cfg = ZONE_CONFIG.get(parent, {"rod_min": 1, "rod_max": 6})
    rod_min, rod_max = cfg["rod_min"], cfg["rod_max"]

    # Softlock: underleveled = Poor only
    if rod_level < rod_min:
        poor_ids = [
            fid for fid in range(zmin, zmax + 1) if fish_data[fid]["rarity"] == "Poor"
        ]
        if not poor_ids:
            return {}
        p = 1.0 / len(poor_ids)
        return {fid: p for fid in poor_ids}

    rod_level = max(rod_min, min(rod_max, rod_level))
    rng = zmax - zmin
    if rng <= 0:
        return {zmin: 1.0}

    t = (rod_level - rod_min) / (rod_max - rod_min) if rod_max > rod_min else 0
    mean_id = zmin + (t**GAUSSIAN_GAMMA) * rng
    sigma = max(
        rng * (GAUSSIAN_SIGMA_MIN + (GAUSSIAN_SIGMA_MAX - GAUSSIAN_SIGMA_MIN) * t), 1e-3
    )

    weights = {}
    total = 0
    for fid in range(zmin, zmax + 1):
        w = math.exp(-0.5 * ((fid - mean_id) / sigma) ** 2)
        if w > 1e-9:
            weights[fid] = w
            total += w

    return {fid: w / total for fid, w in weights.items()} if total > 0 else {}


def expected_value(
    fish_data: dict, zone_ranges: dict, zone: str, rod_level: int
) -> float:
    """Expected gold value per catch."""
    probs = compute_drop_probs(fish_data, zone_ranges, zone, rod_level)
    return sum(p * fish_data[fid]["base_value"] for fid, p in probs.items())


def gold_per_hour(
    fish_data: dict,
    zone_ranges: dict,
    zone: str,
    rod_level: int,
    catch_base: float,
    catch_per_rod: float,
    catch_min: float,
) -> float:
    """Expected gold per hour."""
    ev = expected_value(fish_data, zone_ranges, zone, rod_level)
    cph = catches_per_hour(rod_level, catch_base, catch_per_rod, catch_min)
    return ev * cph


def rarity_probabilities(
    fish_data: dict, zone_ranges: dict, zone: str, rod_level: int
) -> dict:
    """Get probability of each rarity tier."""
    probs = compute_drop_probs(fish_data, zone_ranges, zone, rod_level)
    rarity_probs = defaultdict(float)
    for fid, p in probs.items():
        rarity_probs[fish_data[fid]["rarity"]] += p
    return dict(rarity_probs)


# ============================================================
# MAIN APP
# ============================================================

st.title("🐟 Fish Economy Tester")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Settings")

    # Data loading options
    st.subheader("📁 Load Data")

    load_method = st.radio(
        "Load method:", ["Auto-detect", "Upload file", "Manual path"]
    )

    fish_data = None

    if load_method == "Auto-detect":
        # Try to find FishData.lua in current directory
        possible_paths = [
            "FishData.lua",
            "./FishData.lua",
            "../FishData.lua",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    fish_data = load_fishdata_from_file(path)
                    st.success(f"✅ Loaded from {path}")
                    st.caption(f"{len(fish_data)} fish found")
                    break
                except Exception as e:
                    st.error(f"Error loading {path}: {e}")

        if fish_data is None:
            st.warning("FishData.lua not found in current directory")
            st.caption(f"Current dir: {os.getcwd()}")

    elif load_method == "Upload file":
        uploaded_file = st.file_uploader("Upload FishData.lua", type=["lua", "txt"])
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode("utf-8")
                fish_data = load_fishdata_from_string(content)
                st.success(f"✅ Loaded {len(fish_data)} fish!")
            except Exception as e:
                st.error(f"Error: {e}")

    else:  # Manual path
        manual_path = st.text_input("Path to FishData.lua:", value="FishData.lua")
        if st.button("Load"):
            try:
                fish_data = load_fishdata_from_file(manual_path)
                st.success(f"✅ Loaded {len(fish_data)} fish!")
            except FileNotFoundError:
                st.error(f"File not found: {manual_path}")
            except Exception as e:
                st.error(f"Error: {e}")

    # Store in session state
    if fish_data:
        st.session_state["fish_data"] = fish_data
    elif "fish_data" in st.session_state:
        fish_data = st.session_state["fish_data"]

    st.divider()

    # Catch time settings
    st.subheader("⏱️ Catch Time")
    catch_base = st.number_input("Base time (sec)", value=8.0, min_value=0.1, step=0.5)
    catch_per_rod = st.number_input(
        "Per rod level", value=0.0, step=0.05, help="Negative = faster"
    )
    catch_min = st.number_input("Minimum time", value=1.5, min_value=0.1, step=0.1)

    st.caption(
        f"Rod 1: {get_catch_time(1, catch_base, catch_per_rod, catch_min):.1f}s → {catches_per_hour(1, catch_base, catch_per_rod, catch_min):.0f}/hr"
    )
    st.caption(
        f"Rod 33: {get_catch_time(33, catch_base, catch_per_rod, catch_min):.1f}s → {catches_per_hour(33, catch_base, catch_per_rod, catch_min):.0f}/hr"
    )


# ==================== MAIN CONTENT ====================

if fish_data is None or len(fish_data) == 0:
    st.warning("👈 Please load FishData.lua using the sidebar")

    st.markdown(
        """
    ### How to use:
    1. Make sure `FishData.lua` is in the same folder as this script
    2. Or use "Upload file" to upload it manually
    3. Or use "Manual path" to specify the full path
    
    ### Don't have FishData.lua yet?
    Run your main fish economy script first to generate it from the RBXMX file.
    """
    )

    st.stop()

# Data is loaded - show the app
zone_ranges = get_zone_ranges(fish_data)
zones = sorted(zone_ranges.keys())
main_zones_available = [z for z in MAIN_ZONES if z in zones]

st.success(f"✅ {len(fish_data)} fish loaded across {len(zones)} zones")

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Zone Analysis",
        "🎣 Rod Comparison",
        "📈 Progression",
        "🎲 Simulation",
        "📋 Fish Database",
    ]
)

# ==================== TAB 1: Zone Analysis ====================
with tab1:
    st.header("Zone Analysis")

    col1, col2 = st.columns([1, 3])

    with col1:
        selected_zone = st.selectbox("Select Zone", zones)
        parent = get_parent_zone(selected_zone)
        cfg = ZONE_CONFIG.get(parent, {"rod_min": 1, "rod_max": 6})

        st.info(f"**Rod Range:** {cfg['rod_min']} - {cfg['rod_max']}")

        zone_fish = [f for f in fish_data.values() if f["zone"] == selected_zone]
        st.metric("Total Fish", len(zone_fish))

        st.subheader("By Rarity")
        for r in RARITIES:
            count = len([f for f in zone_fish if f["rarity"] == r])
            if count > 0:
                st.markdown(
                    f"<span style='color:{RARITY_COLORS[r]}'>{r}: {count}</span>",
                    unsafe_allow_html=True,
                )

    with col2:
        # Build data table
        rod_range = range(max(1, cfg["rod_min"] - 2), min(34, cfg["rod_max"] + 3))
        data = []
        for rod in rod_range:
            gph = gold_per_hour(
                fish_data,
                zone_ranges,
                selected_zone,
                rod,
                catch_base,
                catch_per_rod,
                catch_min,
            )
            ev = expected_value(fish_data, zone_ranges, selected_zone, rod)
            cph = catches_per_hour(rod, catch_base, catch_per_rod, catch_min)
            status = (
                "Softlock"
                if rod < cfg["rod_min"]
                else ("Optimal" if rod <= cfg["rod_max"] else "Overleveled")
            )
            data.append(
                {
                    "Rod": rod,
                    "Gold/Hour": gph,
                    "EV/Catch": ev,
                    "Catches/Hr": cph,
                    "Status": status,
                }
            )

        df = pd.DataFrame(data)

        # Simple line chart
        st.subheader(f"Gold/Hour in {selected_zone}")
        st.line_chart(df.set_index("Rod")["Gold/Hour"])

        # Data table
        st.dataframe(df, use_container_width=True)

# ==================== TAB 2: Rod Comparison ====================
with tab2:
    st.header("Rod Comparison")

    rod_level = st.slider("Select Rod Level", 1, 33, 10)

    comparison_data = []
    for zone in main_zones_available:
        cfg = ZONE_CONFIG[zone]
        gph = gold_per_hour(
            fish_data,
            zone_ranges,
            zone,
            rod_level,
            catch_base,
            catch_per_rod,
            catch_min,
        )
        ev = expected_value(fish_data, zone_ranges, zone, rod_level)

        if rod_level < cfg["rod_min"]:
            status = "🔒 Softlocked"
        elif rod_level > cfg["rod_max"]:
            status = "⚠️ Overleveled"
        else:
            status = "✅ Optimal"

        comparison_data.append(
            {
                "Zone": zone,
                "Gold/Hour": gph,
                "EV/Catch": ev,
                "Status": status,
                "Rod Range": f"{cfg['rod_min']}-{cfg['rod_max']}",
            }
        )

    df = pd.DataFrame(comparison_data).sort_values("Gold/Hour", ascending=False)

    best = df.iloc[0]
    st.success(
        f"**Best zone for Rod {rod_level}:** {best['Zone']} ({best['Gold/Hour']:,.0f} gold/hr)"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df.set_index("Zone")["Gold/Hour"])

    with col2:
        st.dataframe(df, use_container_width=True)

# ==================== TAB 3: Progression ====================
with tab3:
    st.header("Optimal Progression")

    progression_data = []
    for rod in range(1, 34):
        best_zone = None
        best_gph = 0
        row = {"Rod": rod}

        for zone in main_zones_available:
            gph = gold_per_hour(
                fish_data, zone_ranges, zone, rod, catch_base, catch_per_rod, catch_min
            )
            row[zone] = gph
            if gph > best_gph:
                best_gph = gph
                best_zone = zone

        row["Best Zone"] = best_zone
        row["Best Gold/Hr"] = best_gph
        progression_data.append(row)

    df = pd.DataFrame(progression_data)

    # Chart all zones
    chart_df = df.set_index("Rod")[main_zones_available]
    st.line_chart(chart_df)

    # Table
    display_df = df[["Rod", "Best Zone", "Best Gold/Hr"]]
    st.dataframe(display_df, use_container_width=True, height=400)

# ==================== TAB 4: Simulation ====================
with tab4:
    st.header("Fishing Simulation")

    col1, col2 = st.columns(2)

    with col1:
        sim_zone = st.selectbox("Zone", zones, key="sim_zone")
        sim_rod = st.number_input("Rod Level", 1, 33, 10, key="sim_rod")

    with col2:
        sim_duration = st.number_input("Duration (minutes)", 1, 480, 60)
        sim_seed = st.number_input("Random Seed (0=random)", 0, 99999, 0)

    if st.button("🎣 Run Simulation", type="primary"):
        if sim_seed > 0:
            random.seed(sim_seed)

        probs = compute_drop_probs(fish_data, zone_ranges, sim_zone, sim_rod)

        if not probs:
            st.error("No fish available!")
        else:
            fish_ids = list(probs.keys())
            weights = [probs[fid] for fid in fish_ids]

            cph = catches_per_hour(sim_rod, catch_base, catch_per_rod, catch_min)
            total_catches = int(cph * (sim_duration / 60))

            catches = defaultdict(int)
            total_gold = 0
            rarity_counts = defaultdict(int)

            for _ in range(total_catches):
                fid = random.choices(fish_ids, weights=weights, k=1)[0]
                fish = fish_data[fid]
                catches[fid] += 1
                total_gold += fish["base_value"]
                rarity_counts[fish["rarity"]] += 1

            # Results
            col1, col2, col3 = st.columns(3)
            col1.metric("Catches", f"{total_catches:,}")
            col2.metric("Gold Earned", f"{total_gold:,}")
            col3.metric("Gold/Hour", f"{total_gold / (sim_duration/60):,.0f}")

            expected_gph = gold_per_hour(
                fish_data,
                zone_ranges,
                sim_zone,
                sim_rod,
                catch_base,
                catch_per_rod,
                catch_min,
            )
            st.caption(f"Expected: {expected_gph:,.0f} gold/hr")

            # Rarity breakdown
            st.subheader("By Rarity")
            rarity_df = pd.DataFrame(
                [
                    {
                        "Rarity": r,
                        "Count": rarity_counts[r],
                        "Pct": f"{100*rarity_counts[r]/total_catches:.1f}%",
                    }
                    for r in RARITIES
                    if rarity_counts[r] > 0
                ]
            )
            st.dataframe(rarity_df, use_container_width=True)

            # Top catches
            st.subheader("Top Catches")
            sorted_catches = sorted(catches.items(), key=lambda x: -x[1])[:10]
            catch_df = pd.DataFrame(
                [
                    {
                        "Fish": fish_data[fid]["name"],
                        "Rarity": fish_data[fid]["rarity"],
                        "Count": count,
                        "Value": fish_data[fid]["base_value"],
                    }
                    for fid, count in sorted_catches
                ]
            )
            st.dataframe(catch_df, use_container_width=True)

# ==================== TAB 5: Fish Database ====================
with tab5:
    st.header("Fish Database")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_zone = st.multiselect("Zone", zones)
    with col2:
        filter_rarity = st.multiselect("Rarity", RARITIES)
    with col3:
        search = st.text_input("Search name")

    db_data = []
    for fid, info in fish_data.items():
        if filter_zone and info["zone"] not in filter_zone:
            continue
        if filter_rarity and info["rarity"] not in filter_rarity:
            continue
        if search and search.lower() not in info["name"].lower():
            continue

        db_data.append(
            {
                "ID": fid,
                "Name": info["name"],
                "Zone": info["zone"],
                "Rarity": info["rarity"],
                "Value": info["base_value"],
                "Weight": info["mean_weight"],
            }
        )

    st.metric("Showing", len(db_data))
    st.dataframe(pd.DataFrame(db_data), use_container_width=True, height=500)
