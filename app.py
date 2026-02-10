"""
Fish Economy Dashboard
======================
- Upload RBXMX → Generate FishData.lua
- Test gold/hour, rod comparisons, simulations
- Export Lua for Roblox
- Rod Pricing Playground

Run with: streamlit run fish_economy_app.py
"""

import streamlit as st
import pandas as pd
import math
import random
import io
import xml.etree.ElementTree as ET
from collections import defaultdict

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Fish Economy Dashboard", page_icon="🐟", layout="wide")

# ============================================================
# CONFIG
# ============================================================

ZONE_CONFIG = {
    "Pirate": {"rod_min": 1, "rod_max": 6, "color": "#8B4513"},
    "Greek": {"rod_min": 7, "rod_max": 12, "color": "#4169E1"},
    "Japanese": {"rod_min": 13, "rod_max": 21, "color": "#DC143C"},
    "Viking": {"rod_min": 22, "rod_max": 30, "color": "#228B22"},
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

MUTATIONS = [
    "_Demonic",
    "_Cosmic",
    "_Divine",
    "_Void",
    "_Crystal",
    "_Slime",
    "_Storm",
    "_Gold",
    "_Ice",
    "_Magma",
    "_Stone",
    "_Wood",
    "_Base",
]

PREFIXES = {
    "Pirate": 1000,
    "PirateSub_1": 10000,
    "PirateSub_2": 11000,
    "PirateSub_3": 12000,
    "Greek": 2000,
    "GreekSub_1": 20000,
    "GreekSub_2": 21000,
    "GreekSub_3": 22000,
    "Japanese": 3000,
    "JapaneseSub_1": 30000,
    "JapaneseSub_2": 31000,
    "JapaneseSub_3": 32000,
    "Viking": 4000,
    "VikingSub_1": 40000,
    "VikingSub_2": 41000,
    "VikingSub_3": 42000,
}

ZONE_OUTPUT_ORDER = [
    "Pirate",
    "PirateSub_1",
    "PirateSub_2",
    "PirateSub_3",
    "Greek",
    "GreekSub_1",
    "GreekSub_2",
    "GreekSub_3",
    "Japanese",
    "JapaneseSub_1",
    "JapaneseSub_2",
    "JapaneseSub_3",
    "Viking",
    "VikingSub_1",
    "VikingSub_2",
    "VikingSub_3",
]

# Economy settings
WEIGHT_MIN = 1
WEIGHT_MAX = 50
GAUSSIAN_GAMMA = 1.4
GAUSSIAN_SIGMA_MIN = 0.10
GAUSSIAN_SIGMA_MAX = 0.40
CALIBRATION_MARGIN = 1.02
CALIBRATION_MAX_ITER = 5

RARITY_PRICE_MULT = {
    "Poor": 0.10,
    "Common": 1.0,
    "Uncommon": 1.3,
    "Rare": 1.8,
    "Epic": 2.3,
    "Legendary": 3.0,
    "Mythic": 4.0,
}

# Full rod data from RodData.lua (everything except price which we calculate)
ROD_ITEMS = {
    1: {
        "name": "Stick",
        "item_power": [20, 24, 28],
        "click_power": [6, 8, 10],
        "distance": [50, 60, 70],
        "luck": [1, 2, 3],
    },
    2: {
        "name": "Bamboo",
        "item_power": [30, 34, 38],
        "click_power": [9, 12, 16],
        "distance": [70, 80, 90],
        "luck": [3, 4, 5],
    },
    3: {
        "name": "Decrepit",
        "item_power": [24, 32, 40],
        "click_power": [12, 16, 21],
        "distance": [90, 100, 110],
        "luck": [4, 5, 6],
    },
    4: {
        "name": "Mech",
        "item_power": [42, 50, 58],
        "click_power": [16, 21, 27],
        "distance": [110, 120, 130],
        "luck": [5, 6, 7],
    },
    5: {
        "name": "Graceful",
        "item_power": [45, 53, 60],
        "click_power": [21, 27, 35],
        "distance": [130, 140, 150],
        "luck": [6, 7, 8],
    },
    6: {
        "name": "Mystic",
        "item_power": [62, 70, 78],
        "click_power": [27, 35, 45],
        "distance": [150, 160, 170],
        "luck": [7, 8, 9],
    },
    7: {
        "name": "Flesh",
        "item_power": [65, 73, 80],
        "click_power": [35, 46, 59],
        "distance": [170, 180, 190],
        "luck": [8, 9, 10],
    },
    8: {
        "name": "Ghostly",
        "item_power": [82, 90, 98],
        "click_power": [46, 61, 78],
        "distance": [190, 200, 210],
        "luck": [10, 11, 12],
    },
    9: {
        "name": "Angelic",
        "item_power": [104, 112, 120],
        "click_power": [60, 79, 100],
        "distance": [210, 220, 230],
        "luck": [12, 13, 14],
    },
    10: {
        "name": "Skeleton",
        "item_power": [124, 136, 148],
        "click_power": [78, 103, 130],
        "distance": [230, 240, 250],
        "luck": [14, 16, 18],
    },
}

ROD_RARITY_COMMENTS = {
    1: "-- 1\u20132) COMMON (21\u201340)",
    3: "-- 3\u20134) UNCOMMON (41\u201360)",
    5: "-- 5\u20136) RARE (61\u201380)",
    7: "-- 7-8) EPIC (81\u2013100)",
    9: "-- 9) LEGENDARY (101\u2013120)",
    10: "-- 10) MYTHIC (121\u2013150)",
}


# Rod level -> (rod_item_id, upgrade_tier 1/2/3)
# Levels 1-3 = rod item 1 tiers 1-3, levels 4-6 = rod item 2 tiers 1-3, etc.
# But level 1 tier 1 is "buying" the rod, tier 2 and 3 are upgrades
def rod_level_to_item_tier(level: int):
    """Convert rod level (1-30) to (item_id, tier). tier is 1-3."""
    item_id = ((level - 1) // 3) + 1
    tier = ((level - 1) % 3) + 1
    return item_id, tier


# ============================================================
# RBXMX PARSER
# ============================================================


def get_vector3(prop):
    x = prop.find("X")
    y = prop.find("Y")
    z = prop.find("Z")
    if x is not None and y is not None and z is not None:
        return (float(x.text), float(y.text), float(z.text))
    return None


def get_mesh_volume(tool_item):
    best_volume = None
    for item in tool_item.iter("Item"):
        if item.get("class") == "MeshPart":
            props = item.find("Properties")
            if props is None:
                continue
            for prop in props:
                if prop.get("name") == "size" and prop.tag == "Vector3":
                    size = get_vector3(prop)
                    if size:
                        volume = size[0] * size[1] * size[2]
                        if best_volume is None or volume > best_volume:
                            best_volume = volume
                    break
    return best_volume


def calculate_weight_from_volume(volume, vol_min, vol_max):
    if volume is None or vol_min is None or vol_max is None:
        return 6
    if vol_max <= vol_min:
        return WEIGHT_MIN
    log_vol = math.log(max(volume, 0.01))
    log_min = math.log(max(vol_min, 0.01))
    log_max = math.log(max(vol_max, 0.01))
    if log_max <= log_min:
        return WEIGHT_MIN
    t = (log_vol - log_min) / (log_max - log_min)
    weight = WEIGHT_MIN + (WEIGHT_MAX - WEIGHT_MIN) * t
    return round(weight, 1)


def parse_rbxmx(file_content: bytes):
    """Parse RBXMX file content and extract fish data."""
    root = ET.fromstring(file_content)

    result = defaultdict(lambda: defaultdict(set))
    volume_data = {}

    def get_name(item):
        props = item.find("Properties")
        if props is not None:
            for prop in props:
                if prop.get("name") == "Name":
                    return prop.text
        return None

    def get_class(item):
        return item.get("class")

    def get_rarity_from_name(folder_name):
        return next((r for r in RARITIES if r in folder_name), None)

    def is_zone_folder(name):
        if name in MAIN_ZONES:
            return True
        for zone in MAIN_ZONES:
            if name.startswith(zone + "Sub_"):
                return True
        return False

    def extract_fish_name(tool_name):
        name = tool_name
        for mut in MUTATIONS:
            if name.endswith(mut):
                name = name[: -len(mut)]
                break
        return name.replace("_", " ")

    def process_item(item, zone=None, rarity=None):
        name = get_name(item)
        item_class = get_class(item)

        if item_class == "Folder" and name:
            if is_zone_folder(name):
                zone = name
            elif get_rarity_from_name(name):
                rarity = get_rarity_from_name(name)
            for child in item.findall("Item"):
                process_item(child, zone=zone, rarity=rarity)

        elif item_class == "Tool" and name and zone and rarity:
            fish_name = extract_fish_name(name)
            result[zone][rarity].add(fish_name)
            if fish_name not in volume_data:
                vol = get_mesh_volume(item)
                if vol is not None:
                    volume_data[fish_name] = vol
        else:
            for child in item.findall("Item"):
                process_item(child, zone=zone, rarity=rarity)

    for item in root.findall(".//Item"):
        if get_class(item) == "Folder":
            name = get_name(item)
            if name and "Fishing Staging" in name:
                for child in item.findall("Item"):
                    process_item(child)
                break

    final = {}
    for zone in result:
        final[zone] = {}
        for rarity in result[zone]:
            final[zone][rarity] = sorted(result[zone][rarity])

    return final, volume_data


# ============================================================
# ECONOMY MODEL
# ============================================================


def get_parent_zone(zone_name: str) -> str:
    for main in MAIN_ZONES:
        if zone_name == main or zone_name.startswith(main + "Sub_"):
            return main
    return zone_name


def rod_level_factor(rod_level: int) -> float:
    return 20 + 5 * rod_level


def design_rod_level_for_zone(zone: str) -> int:
    parent = get_parent_zone(zone)
    cfg = ZONE_CONFIG.get(parent)
    if not cfg:
        return 1
    return (cfg["rod_min"] + cfg["rod_max"]) // 2


class FishEconomy:
    def __init__(self, fish_data, volume_data=None):
        self.fish_data = fish_data
        self.volume_data = volume_data or {}
        self.id_to_fish = {}
        self.zone_ranges = {}
        self.calibration_log = []

        if self.volume_data:
            volumes = list(self.volume_data.values())
            self.vol_min = min(volumes)
            self.vol_max = max(volumes)
        else:
            self.vol_min = None
            self.vol_max = None

        self._build_fish_index()
        self._calculate_initial_values()
        self._calibrate_zones()

    def _build_fish_index(self):
        zone_order = ZONE_OUTPUT_ORDER

        for zone in zone_order:
            if zone not in self.fish_data:
                continue

            prefix = PREFIXES.get(zone, 99000)
            fid = prefix
            min_id = None
            max_id = None

            for rarity in RARITIES:
                if rarity not in self.fish_data[zone]:
                    continue
                for species in self.fish_data[zone][rarity]:
                    clean = species.replace(" ", "_")
                    volume = self.volume_data.get(species)
                    weight = calculate_weight_from_volume(
                        volume, self.vol_min, self.vol_max
                    )

                    self.id_to_fish[fid] = {
                        "id": fid,
                        "zone": zone,
                        "name": clean,
                        "rarity": rarity,
                        "base_value": 0,
                        "mean_weight": weight,
                        "volume": volume,
                    }
                    if min_id is None:
                        min_id = fid
                    max_id = fid
                    fid += 1

            if min_id is not None:
                self.zone_ranges[zone] = (min_id, max_id)

    def _calculate_initial_values(self):
        for fid, info in self.id_to_fish.items():
            zone = info["zone"]
            rarity = info["rarity"]
            rod_lvl = design_rod_level_for_zone(zone)
            value = rod_level_factor(rod_lvl) * RARITY_PRICE_MULT.get(rarity, 1.0)

            if zone in self.zone_ranges:
                zmin, zmax = self.zone_ranges[zone]
                same_rarity_ids = [
                    f
                    for f in range(zmin, zmax + 1)
                    if self.id_to_fish[f]["rarity"] == rarity
                ]
                same_rarity_ids.sort()
                if fid in same_rarity_ids:
                    idx = same_rarity_ids.index(fid)
                    if len(same_rarity_ids) > 1:
                        value *= 1 + 0.05 * idx

            info["base_value"] = value

    def _compute_gaussian_params(self, zone: str, rod_level: int):
        if zone not in self.zone_ranges:
            return 0, 1.0

        zmin, zmax = self.zone_ranges[zone]
        parent = get_parent_zone(zone)
        cfg = ZONE_CONFIG.get(parent, {"rod_min": 1, "rod_max": 6})
        rod_min, rod_max = cfg["rod_min"], cfg["rod_max"]
        rod_level = max(rod_min, min(rod_max, rod_level))

        rng = zmax - zmin
        if rng <= 0:
            return float(zmin), 1.0

        t = 0.0 if rod_max == rod_min else (rod_level - rod_min) / (rod_max - rod_min)
        mean_pos = t**GAUSSIAN_GAMMA
        mean_id = zmin + mean_pos * rng
        sigma = rng * (
            GAUSSIAN_SIGMA_MIN + (GAUSSIAN_SIGMA_MAX - GAUSSIAN_SIGMA_MIN) * t
        )

        return mean_id, max(sigma, 1e-3)

    def drop_probs(self, zone: str, rod_level: int):
        if zone not in self.zone_ranges:
            return {}

        zmin, zmax = self.zone_ranges[zone]
        parent = get_parent_zone(zone)
        cfg = ZONE_CONFIG.get(parent, {"rod_min": 1, "rod_max": 6})
        rod_min = cfg["rod_min"]

        if rod_level < rod_min:
            poor_ids = [
                fid
                for fid in range(zmin, zmax + 1)
                if self.id_to_fish[fid]["rarity"] == "Poor"
            ]
            if not poor_ids:
                return {}
            p = 1.0 / len(poor_ids)
            return {fid: p for fid in poor_ids}

        mean_id, sigma = self._compute_gaussian_params(zone, rod_level)
        weights = {}
        total_w = 0.0
        for fid in range(zmin, zmax + 1):
            z = (fid - mean_id) / sigma
            gauss = math.exp(-0.5 * z * z)
            if gauss > 0:
                weights[fid] = gauss
                total_w += gauss

        if total_w <= 0:
            uniform_p = 1.0 / (zmax - zmin + 1)
            return {fid: uniform_p for fid in range(zmin, zmax + 1)}

        return {fid: w / total_w for fid, w in weights.items()}

    def expected_gold_per_catch(self, zone: str, rod_level: int) -> float:
        probs = self.drop_probs(zone, rod_level)
        return sum(p * self.id_to_fish[fid]["base_value"] for fid, p in probs.items())

    def _calibrate_zones(self):
        zone_order = [z for z in MAIN_ZONES if z in self.zone_ranges]

        for iteration in range(CALIBRATION_MAX_ITER):
            changed = False
            for i in range(len(zone_order) - 1):
                z_prev = zone_order[i]
                z_next = zone_order[i + 1]

                prev_end_lvl = ZONE_CONFIG[z_prev]["rod_max"]
                next_start_lvl = ZONE_CONFIG[z_next]["rod_min"]

                e_prev = self.expected_gold_per_catch(z_prev, prev_end_lvl)
                e_next = self.expected_gold_per_catch(z_next, next_start_lvl)

                if e_next < e_prev * CALIBRATION_MARGIN:
                    factor = (e_prev * CALIBRATION_MARGIN) / max(e_next, 1e-9)
                    self.calibration_log.append(
                        f"Iter {iteration}: Boosting {z_next} by x{factor:.3f}"
                    )

                    for fid, info in self.id_to_fish.items():
                        if get_parent_zone(info["zone"]) == z_next:
                            info["base_value"] *= factor
                    changed = True

            if not changed:
                break

    def generate_lua(self) -> str:
        lua_output = "local FishData = {\n\n"

        for zone in ZONE_OUTPUT_ORDER:
            if zone not in self.fish_data:
                continue

            prefix = PREFIXES.get(zone, 99000)
            species_index = prefix
            lua_output += f"\t-- // {zone}\n"

            for rarity in RARITIES:
                if rarity not in self.fish_data[zone]:
                    continue

                for species in self.fish_data[zone][rarity]:
                    clean_name = species.replace(" ", "_")
                    base_value = int(
                        round(self.id_to_fish[species_index]["base_value"])
                    )
                    mean_weight = self.id_to_fish[species_index]["mean_weight"]

                    lua_output += (
                        f"\t[{species_index}] = {{ "
                        f"item_id = {species_index}, "
                        f'name = "{clean_name}", '
                        f"rarity = _G.EnumModule.Rarity.{rarity.upper()}, "
                        f"mean_weight = {mean_weight}, "
                        f"base_value = {base_value} "
                        f"}},\n"
                    )
                    species_index += 1
            lua_output += "\n"

        lua_output += "}\n\nreturn FishData\n"
        return lua_output


# ============================================================
# FISHDATA.LUA LOADER (for dashboard)
# ============================================================


def parse_rarity(rarity_str: str) -> str:
    return rarity_str.strip().capitalize()


def load_fishdata_from_string(content: str) -> dict:
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

                rarity_start = line.find("Rarity.") + 7
                rarity_end = line.find(",", rarity_start)
                rarity = parse_rarity(line[rarity_start:rarity_end])

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


# ============================================================
# DASHBOARD HELPERS
# ============================================================


def get_zone_ranges(fish_data: dict) -> dict:
    zone_fish = defaultdict(list)
    for fid, info in fish_data.items():
        zone_fish[info["zone"]].append(fid)
    return {z: (min(ids), max(ids)) for z, ids in zone_fish.items() if ids}


def get_catch_time(base: float) -> float:
    return base


def catches_per_hour(base: float) -> float:
    return 3600 / get_catch_time(base)


def compute_drop_probs(
    fish_data: dict, zone_ranges: dict, zone: str, rod_level: int
) -> dict:
    if zone not in zone_ranges:
        return {}

    zmin, zmax = zone_ranges[zone]
    parent = get_parent_zone(zone)
    cfg = ZONE_CONFIG.get(parent, {"rod_min": 1, "rod_max": 6})
    rod_min, rod_max = cfg["rod_min"], cfg["rod_max"]

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
    probs = compute_drop_probs(fish_data, zone_ranges, zone, rod_level)
    return sum(p * fish_data[fid]["base_value"] for fid, p in probs.items())


def gold_per_hour(
    fish_data: dict,
    zone_ranges: dict,
    zone: str,
    rod_level: int,
    catch_base: float,
) -> float:
    ev = expected_value(fish_data, zone_ranges, zone, rod_level)
    cph = catches_per_hour(catch_base)
    return ev * cph


def get_optimal_zone_for_rod(fish_data, zone_ranges, rod_level):
    """Find the main zone with highest EV/catch for a given rod level."""
    best_zone = None
    best_ev = 0
    for zone_name, cfg in ZONE_CONFIG.items():
        if zone_name not in zone_ranges:
            continue
        # Only consider if rod is within this zone's range
        if rod_level < cfg["rod_min"] or rod_level > cfg["rod_max"]:
            continue
        ev = expected_value(fish_data, zone_ranges, zone_name, rod_level)
        if ev > best_ev:
            best_ev = ev
            best_zone = zone_name
    # Fallback: if no zone matches (rod below all minimums), pick the one with lowest rod_min
    if best_zone is None:
        for zone_name, cfg in ZONE_CONFIG.items():
            if zone_name in zone_ranges:
                ev = expected_value(fish_data, zone_ranges, zone_name, rod_level)
                if ev > best_ev:
                    best_ev = ev
                    best_zone = zone_name
    return best_zone, best_ev


# ============================================================
# LOCAL FISHDATA PATH - Change this to your local FishData.lua
# ============================================================
import os

LOCAL_FISHDATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "FishData.lua"
)

# ============================================================
# MAIN APP
# ============================================================

st.title("🐟 Fish Economy Dashboard")

# Auto-load local FishData.lua on first run
if "fish_data" not in st.session_state and os.path.exists(LOCAL_FISHDATA_PATH):
    try:
        with open(LOCAL_FISHDATA_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        fish_data_loaded = load_fishdata_from_string(content)
        if fish_data_loaded:
            st.session_state["fish_data"] = fish_data_loaded
            st.session_state["lua_code"] = content
            st.toast(
                f"✅ Auto-loaded {len(fish_data_loaded)} fish from {LOCAL_FISHDATA_PATH}"
            )
    except Exception:
        pass  # Silently fail, user can upload manually

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("⏱️ Catch Time")
    catch_base = st.number_input(
        "Seconds per fish", value=10.0, min_value=0.1, step=0.5
    )


# ==================== TABS ====================
tab_gen, tab_zone, tab_rod, tab_prog, tab_sim, tab_db, tab_rodprice = st.tabs(
    [
        "🔧 Generate Lua",
        "📊 Zone Analysis",
        "🎣 Rod Comparison",
        "📈 Progression",
        "🎲 Simulation",
        "📋 Fish Database",
        "💰 Rod Pricing",
    ]
)


# ==================== TAB: GENERATE LUA ====================
with tab_gen:
    st.header("Generate FishData.lua from RBXMX")

    st.markdown(
        """
    **How to use:**
    1. Upload your `.rbxmx` file (exported from Roblox Studio)
    2. The parser will extract all fish, zones, rarities, and mesh volumes
    3. Download or copy the generated `FishData.lua`
    """
    )

    uploaded_rbxmx = st.file_uploader(
        "Upload RBXMX file", type=["rbxmx"], key="rbxmx_upload"
    )

    if uploaded_rbxmx is not None:
        try:
            with st.spinner("Parsing RBXMX..."):
                content = uploaded_rbxmx.read()
                raw_fish_data, volume_data = parse_rbxmx(content)

            st.success(f"✅ Parsed successfully!")

            # Show stats
            col1, col2, col3 = st.columns(3)
            total_fish = sum(
                len(raw_fish_data[z][r])
                for z in raw_fish_data
                for r in raw_fish_data[z]
            )
            col1.metric("Total Fish", total_fish)
            col2.metric("Zones", len(raw_fish_data))
            col3.metric("With Volume Data", len(volume_data))

            # Zone breakdown
            st.subheader("Zones Found")
            zone_summary = []
            for zone in ZONE_OUTPUT_ORDER:
                if zone in raw_fish_data:
                    count = sum(
                        len(raw_fish_data[zone][r]) for r in raw_fish_data[zone]
                    )
                    zone_summary.append({"Zone": zone, "Fish Count": count})
            st.dataframe(pd.DataFrame(zone_summary), use_container_width=True)

            # Generate economy
            with st.spinner("Calibrating economy..."):
                economy = FishEconomy(raw_fish_data, volume_data)

            # Show calibration log
            if economy.calibration_log:
                with st.expander("Calibration Log"):
                    for log in economy.calibration_log:
                        st.text(log)

            # Generate Lua
            lua_code = economy.generate_lua()

            st.subheader("Generated FishData.lua")

            # Download button
            st.download_button(
                label="📥 Download FishData.lua",
                data=lua_code,
                file_name="FishData.lua",
                mime="text/plain",
                type="primary",
            )

            # Copy-paste area
            with st.expander("📋 View/Copy Lua Code"):
                st.code(lua_code, language="lua")

            # Store for dashboard use
            st.session_state["lua_code"] = lua_code
            st.session_state["fish_data"] = load_fishdata_from_string(lua_code)
            st.session_state["economy"] = economy

            st.info(
                "✅ Fish data loaded into dashboard! Check the other tabs to analyze."
            )

        except Exception as e:
            st.error(f"Error parsing RBXMX: {e}")
            st.exception(e)

    st.divider()

    # Alternative: upload existing FishData.lua
    st.subheader("Or Upload Existing FishData.lua")
    uploaded_lua = st.file_uploader(
        "Upload FishData.lua", type=["lua", "txt"], key="lua_upload"
    )

    if uploaded_lua is not None:
        try:
            content = uploaded_lua.read().decode("utf-8")
            fish_data = load_fishdata_from_string(content)
            st.session_state["fish_data"] = fish_data
            st.session_state["lua_code"] = content
            st.success(f"✅ Loaded {len(fish_data)} fish from FishData.lua!")
        except Exception as e:
            st.error(f"Error loading: {e}")


# ==================== CHECK IF DATA LOADED ====================
fish_data = st.session_state.get("fish_data")

if not fish_data:
    for tab in [tab_zone, tab_rod, tab_prog, tab_sim, tab_db, tab_rodprice]:
        with tab:
            st.warning("👈 Generate or upload FishData.lua in the first tab")
else:
    zone_ranges = get_zone_ranges(fish_data)
    zones = sorted(zone_ranges.keys())
    main_zones_available = [z for z in MAIN_ZONES if z in zones]

    # ==================== TAB: ZONE ANALYSIS ====================
    with tab_zone:
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
            rod_range = range(max(1, cfg["rod_min"] - 2), min(31, cfg["rod_max"] + 3))
            data = []
            for rod in rod_range:
                gph = gold_per_hour(
                    fish_data,
                    zone_ranges,
                    selected_zone,
                    rod,
                    catch_base,
                )
                ev = expected_value(fish_data, zone_ranges, selected_zone, rod)
                cph = catches_per_hour(catch_base)
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
            st.subheader(f"Gold/Hour in {selected_zone}")
            st.line_chart(df.set_index("Rod")["Gold/Hour"])
            st.dataframe(df, use_container_width=True)

    # ==================== TAB: ROD COMPARISON ====================
    with tab_rod:
        st.header("Rod Comparison")

        rod_level = st.slider("Select Rod Level", 1, 30, 10)

        comparison_data = []
        for zone in main_zones_available:
            cfg = ZONE_CONFIG[zone]
            gph = gold_per_hour(
                fish_data,
                zone_ranges,
                zone,
                rod_level,
                catch_base,
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

    # ==================== TAB: PROGRESSION ====================
    with tab_prog:
        st.header("Optimal Progression")

        progression_data = []
        for rod in range(1, 31):
            best_zone = None
            best_gph = 0
            row = {"Rod": rod}

            for zone in main_zones_available:
                gph = gold_per_hour(
                    fish_data,
                    zone_ranges,
                    zone,
                    rod,
                    catch_base,
                )
                row[zone] = gph
                if gph > best_gph:
                    best_gph = gph
                    best_zone = zone

            row["Best Zone"] = best_zone
            row["Best Gold/Hr"] = best_gph
            progression_data.append(row)

        df = pd.DataFrame(progression_data)

        if main_zones_available:
            chart_df = df.set_index("Rod")[main_zones_available]
            st.line_chart(chart_df)

        display_df = df[["Rod", "Best Zone", "Best Gold/Hr"]]
        st.dataframe(display_df, use_container_width=True, height=400)

    # ==================== TAB: SIMULATION ====================
    with tab_sim:
        st.header("Fishing Simulation")

        col1, col2 = st.columns(2)
        with col1:
            sim_zone = st.selectbox("Zone", zones, key="sim_zone")
            sim_rod = st.number_input("Rod Level", 1, 30, 10, key="sim_rod")
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

                cph = catches_per_hour(catch_base)
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
                )
                st.caption(f"Expected: {expected_gph:,.0f} gold/hr")

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

    # ==================== TAB: DATABASE ====================
    with tab_db:
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

    # ==================== TAB: ROD PRICING PLAYGROUND ====================
    with tab_rodprice:
        st.header("💰 Rod Pricing Playground")

        st.caption(
            "Price = EV/catch × catches/min × minutes. "
            "Lvl 1 auto-priced at ~1 fish. Optimal zone per rod level."
        )

        # Zone-to-rod-level mapping
        ZONE_ROD_RANGES = {
            "Pirate": (1, 6),
            "Greek": (7, 12),
            "Japanese": (13, 21),
            "Viking": (22, 30),
        }
        DEFAULT_STARTS = {"Pirate": 1.0, "Greek": 7.0, "Japanese": 20.0, "Viking": 47.0}
        DEFAULT_INCS = {"Pirate": 1.0, "Greek": 2.0, "Japanese": 3.0, "Viking": 3.0}

        def gen_zone_minutes(starts, incs):
            mins = [0.0] * 31
            for zone in ["Pirate", "Greek", "Japanese", "Viking"]:
                lo, hi = ZONE_ROD_RANGES[zone]
                # first_real = first non-auto level in this zone
                first_real = lo + 1 if lo == 1 else lo
                for lvl in range(lo, hi + 1):
                    if lvl == 1:
                        continue
                    offset = lvl - first_real
                    mins[lvl] = round(starts[zone] + offset * incs[zone], 1)
            return mins

        def sync_widget_keys_from_minutes():
            """Write rod_minutes values directly into widget keys so inputs update."""
            stored = st.session_state["rod_minutes"]
            for lvl in range(2, 31):
                val = stored[lvl] if lvl < len(stored) else 10.0
                st.session_state[f"rm_{lvl}"] = float(val)

        if "rod_minutes" not in st.session_state:
            st.session_state["rod_minutes"] = gen_zone_minutes(
                DEFAULT_STARTS, DEFAULT_INCS
            )

        # Top row
        col_spf, col_p1, col_p2 = st.columns([1.5, 1, 1])
        with col_spf:
            rod_price_sec_per_fish = st.number_input(
                "Seconds per fish",
                value=10.0,
                min_value=1.0,
                max_value=60.0,
                step=0.5,
                key="rod_price_spf",
            )
        catches_per_min = 60.0 / rod_price_sec_per_fish
        with col_p1:
            if st.button("🔄 Reset Default", use_container_width=True):
                st.session_state["rod_minutes"] = gen_zone_minutes(
                    DEFAULT_STARTS, DEFAULT_INCS
                )
                sync_widget_keys_from_minutes()
                st.rerun()
        with col_p2:
            if st.button("📈 Exponential", use_container_width=True):
                exp = [0.0] + [round(2 * (1.16**i), 1) for i in range(30)]
                st.session_state["rod_minutes"] = exp
                sync_widget_keys_from_minutes()
                st.rerun()

        # Zone increment editor
        with st.expander("⚡ Zone Increment Editor"):
            st.caption("Set start minute & per-level increment, then Apply.")
            zc1, zc2, zc3, zc4, zc5 = st.columns([1, 1, 1, 1, 0.6])
            zinc_s, zinc_i = {}, {}
            for zone, col in [
                ("Pirate", zc1),
                ("Greek", zc2),
                ("Japanese", zc3),
                ("Viking", zc4),
            ]:
                with col:
                    st.markdown(f"**{zone}**")
                    zinc_s[zone] = st.number_input(
                        "Start",
                        min_value=0.1,
                        max_value=999.0,
                        value=DEFAULT_STARTS[zone],
                        step=0.5,
                        key=f"zs_{zone}",
                    )
                    zinc_i[zone] = st.number_input(
                        "+/lvl",
                        min_value=0.1,
                        max_value=99.0,
                        value=DEFAULT_INCS[zone],
                        step=0.5,
                        key=f"zi_{zone}",
                    )
            with zc5:
                st.markdown(" ")
                st.markdown(" ")
                if st.button(
                    "Apply", type="primary", use_container_width=True, key="za"
                ):
                    st.session_state["rod_minutes"] = gen_zone_minutes(zinc_s, zinc_i)
                    sync_widget_keys_from_minutes()
                    st.rerun()

        st.divider()

        # Compact grid
        hc1, hc2, hc3, hc4 = st.columns([1.5, 1, 1, 1])
        hc1.markdown("**Rod**")
        hc2.markdown("**T1 (min)**")
        hc3.markdown("**T2 (min)**")
        hc4.markdown("**T3 (min)**")

        minutes_map = {1: 0.0}
        for rod_item_id in range(1, 11):
            rod_info = ROD_ITEMS[rod_item_id]
            c_name, c1, c2, c3 = st.columns([1.5, 1, 1, 1])
            c_name.markdown(f"**{rod_item_id}. {rod_info['name']}**")

            for tier, col in [(1, c1), (2, c2), (3, c3)]:
                rod_level = (rod_item_id - 1) * 3 + tier
                if rod_level == 1:
                    col.caption("auto")
                else:
                    stored = st.session_state["rod_minutes"]
                    dv = stored[rod_level] if rod_level < len(stored) else 10.0
                    val = col.number_input(
                        f"l{rod_level}",
                        min_value=0.1,
                        max_value=9999.0,
                        value=float(dv),
                        step=0.5,
                        key=f"rm_{rod_level}",
                        label_visibility="collapsed",
                    )
                    minutes_map[rod_level] = val

        new_mins = [0.0] * 31
        for lvl in range(1, 31):
            new_mins[lvl] = minutes_map.get(lvl, 0.0)
        st.session_state["rod_minutes"] = new_mins

        st.divider()

        # Calculate results
        results = []
        cumulative_gold = 0
        cumulative_minutes = 0

        for level_idx in range(30):
            rod_level = level_idx + 1
            rod_item_id, tier = rod_level_to_item_tier(rod_level)
            rod_name = ROD_ITEMS[rod_item_id]["name"]

            game_rod_level = min(rod_level, 30)
            optimal_zone, ev_per_catch = get_optimal_zone_for_rod(
                fish_data, zone_ranges, game_rod_level
            )

            if optimal_zone is None:
                optimal_zone = "N/A"
                ev_per_catch = 0

            if rod_level == 1:
                price = round(ev_per_catch)
                minutes = 0
            else:
                minutes = new_mins[rod_level] if rod_level < len(new_mins) else 10.0
                gold_per_min = ev_per_catch * catches_per_min
                price = round(gold_per_min * minutes)

            cumulative_gold += price
            cumulative_minutes += minutes

            results.append(
                {
                    "Lvl": rod_level,
                    "Rod": rod_name,
                    "T": tier,
                    "Zone": optimal_zone,
                    "EV/Catch": round(ev_per_catch, 1),
                    "Min": round(minutes, 1),
                    "Price": price,
                    "Cumul. Gold": cumulative_gold,
                    "Cumul. Hrs": round(cumulative_minutes / 60, 1),
                }
            )

        results_df = pd.DataFrame(results)

        # Summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Gold", f"{cumulative_gold:,}")
        col2.metric("Total Grind", f"{cumulative_minutes / 60:.1f} hrs")
        col3.metric(
            f"@ {rod_price_sec_per_fish}s/fish", f"{catches_per_min:.1f} catches/min"
        )

        st.line_chart(results_df.set_index("Lvl")["Price"])
        st.dataframe(results_df, use_container_width=True, height=400)

        # ---- FULL RODDATA EXPORT ----
        st.subheader("📋 Export RodData.lua")

        lua_lines = []
        lua_lines.append("local RodData = {")

        # Pre-compute max price width for consistent alignment
        all_prices = [r["Price"] for r in results]
        price_width = max(len(str(p)) for p in all_prices) if all_prices else 6

        for rod_item_id in range(1, 11):
            rod = ROD_ITEMS[rod_item_id]
            # Add rarity comment if applicable
            if rod_item_id in ROD_RARITY_COMMENTS:
                if rod_item_id in (3, 5, 7):
                    lua_lines.append(f"\t")
                lua_lines.append(f"\t{ROD_RARITY_COMMENTS[rod_item_id]}")

            # Gather calculated prices for this rod's 3 tiers
            tier_prices = []
            for tier in range(1, 4):
                level_idx = (rod_item_id - 1) * 3 + (tier - 1)
                tier_prices.append(results[level_idx]["Price"])

            # Format arrays with alignment
            def fmt_arr(arr, width=4):
                return "{ " + ", ".join(f"{v:>{width}}" for v in arr) + " }"

            ip = fmt_arr(rod["item_power"])
            cp = fmt_arr(rod["click_power"])
            dist = fmt_arr(rod["distance"])
            lk = fmt_arr(rod["luck"])
            pr = fmt_arr(tier_prices, width=price_width)

            name_padded = f'"{rod["name"]}"'
            # Build line matching original format: [1], [2], ... [10]
            line = (
                f"\t[{rod_item_id}]"
                f'{" " if rod_item_id < 10 else ""}'
                f" = {{ "
                f"item_id={rod_item_id}, "
                f'{" " if rod_item_id < 10 else ""}'
                f"name={name_padded:<12s}, "
                f"item_power={ip}, "
                f"click_power={cp}, "
                f"distance={dist}, "
                f"luck={lk}, "
                f"price={pr}"
                f" }},"
            )
            lua_lines.append(line)

        lua_lines.append("}")
        lua_lines.append("function RodData.PostInitialize()")
        lua_lines.append('\tprint("Initializing images for", script.Name)')
        lua_lines.append(
            "\t_G.MScript.ImageAssigner.AssignImages(RodData, { _G.MScript.RodIcons })"
        )
        lua_lines.append("end")
        lua_lines.append("return RodData")

        lua_output = "\n".join(lua_lines)
        st.code(lua_output, language="lua")

        st.download_button(
            "📥 Download RodData.lua",
            data=lua_output,
            file_name="RodData.lua",
            mime="text/plain",
        )
