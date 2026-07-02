"""
Ocean Quest — XP & Economy Simulator
====================================
Tunes the new fishing economy:
  - Fish gold values DERIVED from: Pirate Common anchor × zone inflation × rarity mult
  - Migration ratios computed at zone boundaries (stay-in-old vs move-to-new)
  - XP curve + rod prices + zone gates all live-tunable
  - Math is EXACT (drop rates × rarity values, no Monte Carlo)

Run:  streamlit run xp_economy_simulator.py
"""

import math
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

RARITIES = ["Poor", "Common", "Uncommon", "Rare", "Epic", "Legendary", "Mythic"]
ZONES = ["Pirate", "Greek", "Japanese", "Viking"]
TIERS = [1, 2, 3, 4, 5]

RARITY_COLORS = {
    "Poor": "#969696",
    "Common": "#FFFFFF",
    "Uncommon": "#50C850",
    "Rare": "#3C8CFF",
    "Epic": "#B450FF",
    "Legendary": "#FFA01E",
    "Mythic": "#FF3C3C",
}
ZONE_COLORS = {
    "Pirate": "#D4763B",
    "Greek": "#4A90D9",
    "Japanese": "#D94A6B",
    "Viking": "#6B4AD9",
}

# Drop rates from FishingConfig.DROP_RATES_BY_ROD_TIER (shared across all zones)
DEFAULT_DROP_RATES = {
    1: {
        "Poor": 0.52,
        "Common": 0.45,
        "Uncommon": 0.03,
        "Rare": 0.0,
        "Epic": 0.0,
        "Legendary": 0.0,
        "Mythic": 0.0,
    },
    2: {
        "Poor": 0.12,
        "Common": 0.45,
        "Uncommon": 0.395,
        "Rare": 0.03,
        "Epic": 0.005,
        "Legendary": 0.0,
        "Mythic": 0.0,
    },
    3: {
        "Poor": 0.04,
        "Common": 0.10,
        "Uncommon": 0.50,
        "Rare": 0.325,
        "Epic": 0.035,
        "Legendary": 0.0,
        "Mythic": 0.0,
    },
    4: {
        "Poor": 0.02,
        "Common": 0.03,
        "Uncommon": 0.13,
        "Rare": 0.48,
        "Epic": 0.27,
        "Legendary": 0.06,
        "Mythic": 0.02,
    },
    5: {
        "Poor": 0.02,
        "Common": 0.03,
        "Uncommon": 0.08,
        "Rare": 0.20,
        "Epic": 0.38,
        "Legendary": 0.23,
        "Mythic": 0.06,
    },
}

# Pricing defaults — Mythic = 8× Common, Z = 16
DEFAULT_PIRATE_COMMON = 40.0
DEFAULT_ZONE_INFLATION = 16.0
DEFAULT_RARITY_MULT = {
    "Poor": 0.1,
    "Common": 1.0,
    "Uncommon": 1.5,
    "Rare": 2.5,
    "Epic": 4.0,
    "Legendary": 6.0,
    "Mythic": 8.0,
}

DEFAULT_XP_PER_RARITY = {
    "Poor": 2,
    "Common": 5,
    "Uncommon": 10,
    "Rare": 20,
    "Epic": 35,
    "Legendary": 65,
    "Mythic": 120,
}
DEFAULT_XP_ZONE_MULT = {"Pirate": 1.0, "Greek": 5.0, "Japanese": 25.0, "Viking": 125.0}

DEFAULT_ROD_XP_MULT = {
    101: 1.0,
    102: 1.3,
    103: 1.7,
    104: 2.1,
    105: 2.5,
    106: 5.0,
    107: 6.0,
    108: 7.0,
    109: 8.0,
    110: 9.0,
    111: 18.0,
    112: 21.0,
    113: 24.0,
    114: 27.0,
    115: 30.0,
    116: 60.0,
    117: 70.0,
    118: 80.0,
    119: 90.0,
    120: 110.0,
}

DEFAULT_ROD_PRICES = {
    101: 0,
    102: 10,
    103: 600,
    104: 3500,
    105: 12000,
    106: 35000,
    107: 70000,
    108: 130000,
    109: 250000,
    110: 400000,
    111: 650000,
    112: 900000,
    113: 1400000,
    114: 2100000,
    115: 3000000,
    116: 4200000,
    117: 5500000,
    118: 6500000,
    119: 7800000,
    120: 9200000,
}

ROD_NAMES = {
    101: "Wooden",
    102: "Bamboo",
    103: "Lantern",
    104: "Worn",
    105: "Polished",
    106: "Fancy",
    107: "Lava",
    108: "Sun",
    109: "Frost",
    110: "Coral",
    111: "Shadow",
    112: "Magma",
    113: "Crystal",
    114: "Petal",
    115: "Infernal",
    116: "Icefang",
    117: "Ghoul",
    118: "Magical",
    119: "Floral",
    120: "Meteor",
}

ROD_TO_ZONE_TIER = {}
for rid in range(101, 121):
    idx = rid - 101
    ROD_TO_ZONE_TIER[rid] = (ZONES[idx // 5], (idx % 5) + 1)

DEFAULT_ZONE_LEVEL_REQ = {"Pirate": 1, "Greek": 25, "Japanese": 50, "Viking": 85}

# Fish lists per (zone, rarity) — from FishingConfig.FISH_LIST_BY_ZONE_AND_RARITY
FISH_LIST = {
    "Pirate": {
        "Poor": [1000, 1001, 1002],
        "Common": [1003, 1004, 1005, 1006, 1007, 10000, 11000],
        "Uncommon": [1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 10001, 11001],
        "Rare": [1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025],
        "Epic": [1026, 1027, 1028, 1029, 1030, 1031, 1032],
        "Legendary": [1033, 1034, 1035, 1036],
        "Mythic": [1037, 1038, 1039],
    },
    "Greek": {
        "Poor": [2000, 2001, 2002],
        "Common": [2003, 2004, 2005, 2006, 2007, 2008, 21000, 22000],
        "Uncommon": [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 20000],
        "Rare": [2018, 2019, 2020, 2021, 2022, 2023, 20001, 21001, 22001],
        "Epic": [2024, 2025, 2026, 2027, 2028, 2029, 2030],
        "Legendary": [2031, 2032, 2033, 2034],
        "Mythic": [2035, 2036, 2037, 2038],
    },
    "Japanese": {
        "Poor": [3000, 3001],
        "Common": [3002, 3003, 3004, 3005, 3006],
        "Uncommon": [3007, 3008, 3009, 3010, 3011, 3012, 30000, 31000, 32000],
        "Rare": [3013, 3014, 3015, 3016, 3017, 3018, 30001, 31001, 32001],
        "Epic": [3019, 3020, 3021, 3022, 3023, 3024, 3025],
        "Legendary": [3026, 3027, 3028],
        "Mythic": [3029, 3030, 3031],
    },
    "Viking": {
        "Poor": [4000, 4001],
        "Common": [4002, 4003, 4004, 4005, 40000, 41000],
        "Uncommon": [4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014],
        "Rare": [4015, 4016, 4017, 4018, 4019, 4020, 4021, 40001, 41001],
        "Epic": [4022, 4023, 4024, 4025, 4026, 4027, 4028],
        "Legendary": [4029, 4030, 4031, 4032],
        "Mythic": [4033, 4034, 4035],
    },
}


# ════════════════════════════════════════════════════════════════════════
# CORE COMPUTATIONS
# ════════════════════════════════════════════════════════════════════════


def derive_zone_rarity_value(zone, rarity, pirate_common, zone_inflation, rarity_mult):
    """Base value for (zone, rarity) bucket. Within-rarity spread applied separately."""
    zone_idx = ZONES.index(zone)
    return pirate_common * (zone_inflation**zone_idx) * rarity_mult[rarity]


def derive_fish_values(pirate_common, zone_inflation, rarity_mult, within_spread=0.05):
    """For every fish, compute its base_value with within-bucket spread."""
    out = {}
    for zone in ZONES:
        for rarity in RARITIES:
            base = derive_zone_rarity_value(
                zone, rarity, pirate_common, zone_inflation, rarity_mult
            )
            for idx, fish_id in enumerate(FISH_LIST[zone][rarity]):
                out[fish_id] = base * (1 + within_spread * idx)
    return out


def avg_value_per_rarity(zone, fish_values):
    out = {}
    for rarity in RARITIES:
        ids = FISH_LIST[zone][rarity]
        vals = [fish_values[fid] for fid in ids if fid in fish_values]
        out[rarity] = float(np.mean(vals)) if vals else 0.0
    return out


def expected_per_catch(
    zone, tier, drop_rates, fish_values, xp_per_rarity, xp_zone_mult, rod_xp_mult_value
):
    """
    Expected XP & gold per catch given:
      - zone (fish values + zone XP mult come from here)
      - tier (drop table)
      - rod_xp_mult_value (the rod's XP multiplier, scalar)
    """
    drops = drop_rates[tier]
    avg_vals = avg_value_per_rarity(zone, fish_values)
    zmult = xp_zone_mult[zone]

    exp_xp, exp_gold = 0.0, 0.0
    breakdown = {}
    for rarity in RARITIES:
        p = drops.get(rarity, 0.0)
        xp = xp_per_rarity[rarity] * zmult * rod_xp_mult_value
        gold = avg_vals[rarity]
        exp_xp += p * xp
        exp_gold += p * gold
        breakdown[rarity] = {"p": p, "xp": xp, "gold": gold}
    return exp_xp, exp_gold, breakdown


def xp_curve(base, power, cap=200):
    return [int(math.floor(base * (n**power))) for n in range(1, cap + 1)]


def cumulative_xp_to_reach(curve):
    cum = [0.0] * (len(curve) + 2)
    for n in range(1, len(curve) + 1):
        cum[n + 1] = cum[n] + curve[n - 1]
    return cum


def level_from_xp(total_xp, curve):
    accum = 0
    for n in range(1, len(curve) + 1):
        need = curve[n - 1]
        if accum + need > total_xp:
            return n, total_xp - accum, need
        accum += need
    return len(curve), total_xp - accum, curve[-1]


# ════════════════════════════════════════════════════════════════════════
# TARGET PACING
# ════════════════════════════════════════════════════════════════════════


def compute_target_times(zone_budget_min, ramp):
    """
    Given a total minute-budget per zone and a within-zone geometric ramp,
    return a dict {rod_id: target_minutes_since_previous_rod}.

    Each zone's budget is split across its 5 rod purchases (the 5 transitions
    INTO tier 1, 2, 3, 4, 5 of that zone). Within the zone, transition N takes
    `ramp` times longer than transition N-1.

    For Pirate: 5 transitions including the "buy rod 101" implicit one. We
    treat 101 as free (time = 0) and split Pirate's budget across transitions
    into rods 102, 103, 104, 105.
    For other zones, all 5 rods (tier 1-5) consume the budget — including
    the cross-zone jump into tier 1.

    Returns: {rod_id: target_minutes_for_this_purchase}
    """
    target = {101: 0.0}

    # Pirate: 4 transitions (102, 103, 104, 105)
    pirate_weights = [ramp**i for i in range(4)]
    total_weight = sum(pirate_weights)
    unit = zone_budget_min["Pirate"] / total_weight
    for i, rid in enumerate([102, 103, 104, 105]):
        target[rid] = unit * pirate_weights[i]

    # Greek, Japan, Viking: 5 transitions each (tier 1 thru 5)
    for zi, zone in enumerate(["Greek", "Japanese", "Viking"]):
        weights = [ramp**i for i in range(5)]
        total_weight = sum(weights)
        unit = zone_budget_min[zone] / total_weight
        base_rid = 101 + (zi + 1) * 5  # 106, 111, 116
        for i in range(5):
            rid = base_rid + i
            target[rid] = unit * weights[i]

    return target


def cumulative_target(target_per_rod):
    """{rod_id: cumulative_minutes_at_this_rod}"""
    out = {}
    s = 0.0
    for rid in range(101, 121):
        s += target_per_rod[rid]
        out[rid] = s
    return out


# ════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="XP & Economy Simulator", layout="wide", page_icon="🎣")

st.title("🎣 XP & Economy Simulator")
st.caption(
    "Tune the new fishing economy. Fish gold values are DERIVED — set the anchor, zone inflation, and rarity ramp, and everything falls out."
)


# ── SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Tuning")

    st.subheader("Player Behavior")
    catches_per_min = st.slider("Catches per minute", 1.0, 30.0, 8.0, 0.5)
    minutes_per_session = st.slider("Avg session length (min)", 1, 120, 20)

    st.divider()
    st.subheader("💰 Pricing")
    st.caption("`value = pirate_common × Z^zone_idx × rarity_mult × (1+spread×idx)`")
    pirate_common = st.number_input(
        "Pirate Common anchor", min_value=1.0, value=DEFAULT_PIRATE_COMMON, step=5.0
    )
    zone_inflation = st.slider(
        "Zone inflation (Z)",
        2.0,
        30.0,
        DEFAULT_ZONE_INFLATION,
        0.5,
        help="Multiplier per zone (Greek = Z × Pirate, etc.)",
    )
    within_spread = st.slider("Within-rarity spread", 0.0, 0.20, 0.05, 0.01)

    st.markdown("**Rarity multipliers**")
    rarity_mult = {}
    for r in RARITIES:
        rarity_mult[r] = st.number_input(
            r,
            min_value=0.0,
            value=DEFAULT_RARITY_MULT[r],
            step=0.1,
            format="%.2f",
            key=f"rmult_{r}",
        )

    st.divider()
    st.subheader("📈 XP Curve")
    xp_base = st.slider("XP base", 1.0, 200.0, 10.0, 1.0)
    xp_power = st.slider("XP power", 0.5, 3.5, 1.6, 0.05)
    level_cap = st.slider("Level cap", 50, 200, 100, 5)

    st.subheader("✨ XP per Rarity (base)")
    xp_per_rarity = {}
    for r in RARITIES:
        xp_per_rarity[r] = st.number_input(
            f"XP {r}",
            min_value=0.0,
            value=float(DEFAULT_XP_PER_RARITY[r]),
            step=1.0,
            key=f"xp_{r}",
        )

    st.subheader("🗺️ XP Zone Multiplier")
    xp_zone_mult = {}
    for z in ZONES:
        xp_zone_mult[z] = st.number_input(
            f"XP×{z}",
            min_value=0.1,
            value=DEFAULT_XP_ZONE_MULT[z],
            step=0.5,
            key=f"xpzone_{z}",
        )

    st.divider()
    st.subheader("🔒 Zone Level Requirements")
    zone_req = {}
    for z in ZONES:
        zone_req[z] = st.number_input(
            z, min_value=1, value=DEFAULT_ZONE_LEVEL_REQ[z], step=1, key=f"req_{z}"
        )

    st.divider()
    st.subheader("⏳ Target Pacing")
    st.caption(
        "How long should a player spend in each zone before moving on? Tier 1→5 ramps geometrically within each zone (tier 1 fast, tier 5 slow)."
    )
    DEFAULT_ZONE_BUDGET_MIN = {
        "Pirate": 18,
        "Greek": 115,
        "Japanese": 425,
        "Viking": 2220,
    }
    zone_budget_min = {}
    for z in ZONES:
        zone_budget_min[z] = st.number_input(
            f"{z} (min)",
            min_value=1.0,
            value=float(DEFAULT_ZONE_BUDGET_MIN[z]),
            step=5.0,
            key=f"budget_{z}",
            help="Total minutes a player should spend going from this zone's tier-1 to its tier-5 rod (excluding the cross-zone jump).",
        )
    pacing_ramp = st.slider(
        "Within-zone ramp",
        1.0,
        5.0,
        2.5,
        0.1,
        help="Geometric ramp factor — tier-N takes `ramp×` longer than tier-(N-1). 1.0 = flat, 2.5 = each rod 2.5× the previous.",
    )

    total_target = sum(zone_budget_min.values())
    st.caption(
        f"**Total target playtime: {total_target:,.0f} min  ({total_target/60:.1f} hr)**"
    )


# ── DERIVE ──────────────────────────────────────────────────────────────
fish_values = derive_fish_values(
    pirate_common, zone_inflation, rarity_mult, within_spread
)
curve = xp_curve(xp_base, xp_power, cap=level_cap)
cum = cumulative_xp_to_reach(curve)
target_per_rod = compute_target_times(zone_budget_min, pacing_ramp)
target_cum = cumulative_target(target_per_rod)


# ── TABS ────────────────────────────────────────────────────────────────
(
    tab_pacing,
    tab_mig,
    tab_prices,
    tab_drops,
    tab_rods,
    tab_rodprices,
    tab_curve,
    tab_prog,
    tab_export,
) = st.tabs(
    [
        "⏳ Pacing",
        "🚦 Migration",
        "💰 Prices",
        "🎯 Drops",
        "🎣 Rod XP",
        "🛒 Rod Prices",
        "📈 Curve",
        "🚀 Progression",
        "📤 Export",
    ]
)


# ════════════════════════════════════════════════════════════════════════
# TAB: DROP TABLES
# ════════════════════════════════════════════════════════════════════════
with tab_drops:
    st.subheader("Drop Tables by Rod Tier (shared across all zones)")
    drop_rates = {}
    cols = st.columns(5)
    for i, tier in enumerate(TIERS):
        with cols[i]:
            st.markdown(f"**Tier {tier}**")
            drop_rates[tier] = {}
            for r in RARITIES:
                drop_rates[tier][r] = st.number_input(
                    r,
                    min_value=0.0,
                    max_value=1.0,
                    value=DEFAULT_DROP_RATES[tier][r],
                    step=0.005,
                    format="%.4f",
                    key=f"drop_{tier}_{r}",
                )
            total = sum(drop_rates[tier].values())
            if abs(total - 1.0) > 0.001:
                st.warning(f"Σ = {total:.4f}")
            else:
                st.success(f"Σ = {total:.4f} ✓")

    st.divider()
    fig = go.Figure()
    for r in RARITIES:
        fig.add_trace(
            go.Bar(
                name=r,
                x=[f"Tier {t}" for t in TIERS],
                y=[drop_rates[t][r] * 100 for t in TIERS],
                marker_color=RARITY_COLORS[r],
            )
        )
    fig.update_layout(
        barmode="stack",
        yaxis_title="Drop %",
        height=350,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# TAB: PRICES
# ════════════════════════════════════════════════════════════════════════
with tab_prices:
    st.subheader("Derived Fish Prices")
    st.caption(
        f"Anchor: {pirate_common:.0f}g · Z: {zone_inflation:.1f}× · Spread: {within_spread*100:.0f}%"
    )

    bucket_df = pd.DataFrame(index=ZONES, columns=RARITIES, dtype=float)
    for zone in ZONES:
        avg = avg_value_per_rarity(zone, fish_values)
        for r in RARITIES:
            bucket_df.loc[zone, r] = avg[r]
    st.dataframe(
        bucket_df.style.format("{:,.0f}").background_gradient(cmap="YlOrRd", axis=None),
        use_container_width=True,
    )

    st.divider()
    st.markdown("**Inspect a bucket**")
    c1, c2 = st.columns(2)
    with c1:
        inspect_zone = st.selectbox("Zone", ZONES)
    with c2:
        inspect_rarity = st.selectbox("Rarity", RARITIES, index=1)
    rows = [
        {"Idx": idx, "Fish ID": fid, "base_value": int(round(fish_values[fid]))}
        for idx, fid in enumerate(FISH_LIST[inspect_zone][inspect_rarity])
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    fig = go.Figure()
    for zone in ZONES:
        avg = avg_value_per_rarity(zone, fish_values)
        fig.add_trace(
            go.Scatter(
                x=RARITIES,
                y=[avg[r] for r in RARITIES],
                mode="lines+markers",
                name=zone,
                line=dict(color=ZONE_COLORS[zone], width=3),
                marker=dict(size=10),
            )
        )
    fig.update_layout(
        yaxis_type="log",
        yaxis_title="Gold (log)",
        height=400,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# TAB: ROD XP MULTS
# ════════════════════════════════════════════════════════════════════════
with tab_rods:
    st.subheader("Per-Rod XP Multiplier")
    p1, p2, p3 = st.columns(3)
    if p1.button("Reset"):
        for rid in range(101, 121):
            st.session_state[f"rodxp_{rid}"] = DEFAULT_ROD_XP_MULT[rid]
        st.rerun()
    if p2.button("Flat 1.0×"):
        for rid in range(101, 121):
            st.session_state[f"rodxp_{rid}"] = 1.0
        st.rerun()
    if p3.button("Linear 1→10"):
        for rid in range(101, 121):
            st.session_state[f"rodxp_{rid}"] = 1.0 + (rid - 101) * (9.0 / 19.0)
        st.rerun()

    rod_xp_mult = {}
    zcols = st.columns(4)
    for zi, zone in enumerate(ZONES):
        with zcols[zi]:
            st.markdown(f"**{zone}**")
            for tier in TIERS:
                rid = 101 + zi * 5 + (tier - 1)
                default = st.session_state.get(f"rodxp_{rid}", DEFAULT_ROD_XP_MULT[rid])
                rod_xp_mult[rid] = st.number_input(
                    f"{rid} {ROD_NAMES[rid]}",
                    min_value=0.1,
                    max_value=500.0,
                    value=float(default),
                    step=0.1,
                    key=f"rodxp_{rid}",
                )


# ════════════════════════════════════════════════════════════════════════
# TAB: ROD PRICES
# ════════════════════════════════════════════════════════════════════════
with tab_rodprices:
    # Apply any pending price overrides from other tabs (e.g. Pacing "Solve rod prices")
    # before the rodprice_* widgets get instantiated this run.
    pending_prices = st.session_state.pop("_pending_rod_prices", None)
    if pending_prices:
        for rid, price in pending_prices.items():
            st.session_state[f"rodprice_{rid}"] = price

    st.subheader("Rod Prices (gold)")
    p1, p2, p3, p4 = st.columns(4)
    if p1.button("Reset prices"):
        for rid in range(101, 121):
            st.session_state[f"rodprice_{rid}"] = DEFAULT_ROD_PRICES[rid]
        st.rerun()
    if p2.button("Geometric ×1.8/rod"):
        v = 10
        for rid in range(101, 121):
            st.session_state[f"rodprice_{rid}"] = int(v)
            v *= 1.8
        st.rerun()
    if p3.button("Geometric ×2.0/rod"):
        v = 10
        for rid in range(101, 121):
            st.session_state[f"rodprice_{rid}"] = int(v)
            v *= 2.0
        st.rerun()
    if p4.button(f"Z-match (×{zone_inflation:.0f}/zone)"):
        per_rod_in_zone = zone_inflation ** (1 / 5)
        v = 10
        for rid in range(101, 121):
            zone, tier = ROD_TO_ZONE_TIER[rid]
            if rid > 101 and tier == 1:
                v *= zone_inflation / (per_rod_in_zone**4)
            st.session_state[f"rodprice_{rid}"] = int(v)
            v *= per_rod_in_zone
        st.rerun()

    rod_prices = {}
    zcols = st.columns(4)
    for zi, zone in enumerate(ZONES):
        with zcols[zi]:
            st.markdown(f"**{zone}**")
            for tier in TIERS:
                rid = 101 + zi * 5 + (tier - 1)
                default = st.session_state.get(
                    f"rodprice_{rid}", DEFAULT_ROD_PRICES[rid]
                )
                rod_prices[rid] = st.number_input(
                    f"{rid} {ROD_NAMES[rid]}",
                    min_value=0,
                    value=int(default),
                    step=100,
                    key=f"rodprice_{rid}",
                )


# ════════════════════════════════════════════════════════════════════════
# BUILD ROD STATS
# ════════════════════════════════════════════════════════════════════════
rod_stats = []
for rid in range(101, 121):
    zone, tier = ROD_TO_ZONE_TIER[rid]
    exp_xp, exp_gold, breakdown = expected_per_catch(
        zone,
        tier,
        drop_rates,
        fish_values,
        xp_per_rarity,
        xp_zone_mult,
        rod_xp_mult[rid],
    )
    rod_stats.append(
        {
            "rod_id": rid,
            "name": ROD_NAMES[rid],
            "zone": zone,
            "tier": tier,
            "rod_xp_mult": rod_xp_mult[rid],
            "exp_xp_per_catch": exp_xp,
            "exp_gold_per_catch": exp_gold,
            "exp_xp_per_hour": exp_xp * catches_per_min * 60,
            "exp_gold_per_hour": exp_gold * catches_per_min * 60,
            "price": rod_prices[rid],
            "breakdown": breakdown,
        }
    )
rod_df = pd.DataFrame(
    [{k: v for k, v in r.items() if k != "breakdown"} for r in rod_stats]
)
rs_by_id = {r["rod_id"]: r for r in rod_stats}


# ════════════════════════════════════════════════════════════════════════
# SIMULATE ACTUAL PROGRESSION (used by Pacing + Progression tabs)
# ════════════════════════════════════════════════════════════════════════
def simulate_progression():
    """
    Walk a player from rod 101 → 120 using current tuning.
    For each transition, compute time spent + cumulative time + bottleneck.
    Returns a list of dicts, one per rod-purchase event.
    """
    rows = []
    cur_rod, cur_xp, cur_gold, cur_minutes = 101, 0, 0, 0

    for rid in range(101, 121):
        target = rs_by_id[rid]
        zone_of_rod = target["zone"]
        zone_unlock_lvl = zone_req[zone_of_rod]
        prev_rod = rs_by_id[cur_rod]
        xp_per_min = prev_rod["exp_xp_per_hour"] / 60
        gold_per_min = prev_rod["exp_gold_per_hour"] / 60

        gold_gap = max(0, target["price"] - cur_gold)
        time_for_gold = (gold_gap / gold_per_min) if gold_per_min > 0 else float("inf")
        xp_to_unlock = cum[zone_unlock_lvl] - cur_xp
        time_for_level = (xp_to_unlock / xp_per_min) if xp_per_min > 0 else float("inf")
        if xp_to_unlock <= 0:
            time_for_level = 0
        time_needed = max(time_for_gold, time_for_level)

        cur_minutes += time_needed
        cur_xp += time_needed * xp_per_min
        cur_gold += time_needed * gold_per_min
        cur_gold -= target["price"]
        new_level, _, _ = level_from_xp(cur_xp, curve)

        rows.append(
            {
                "rod_id": rid,
                "name": ROD_NAMES[rid],
                "zone": zone_of_rod,
                "lvl_gate": zone_unlock_lvl,
                "price": int(target["price"]),
                "time_needed": time_needed,
                "total_minutes": cur_minutes,
                "time_for_gold": time_for_gold,
                "time_for_level": time_for_level,
                "level_on_buy": new_level,
                "gold_after": int(cur_gold),
                "bottleneck": (
                    "Level"
                    if time_for_level > time_for_gold
                    else ("Gold" if time_for_gold > 0 else "Free")
                ),
            }
        )
        cur_rod = rid
    return rows


sim_rows = simulate_progression()
sim_by_id = {r["rod_id"]: r for r in sim_rows}


# ════════════════════════════════════════════════════════════════════════
# TAB: PACING (TARGET vs ACTUAL)
# ════════════════════════════════════════════════════════════════════════
with tab_pacing:
    st.subheader("Target vs Actual Pacing")
    st.caption(
        "**Target** comes from the per-zone time budgets in the sidebar. "
        "**Actual** comes from the current tuning (drop rates × prices × gold values). "
        "Green = within target, yellow = ~2× off, red = >3× off."
    )

    # Build comparison table
    pace_rows = []
    for rid in range(101, 121):
        actual = sim_by_id[rid]
        target = target_per_rod[rid]
        delta = actual["time_needed"] - target
        pct = (
            (actual["time_needed"] / max(target, 0.01)) * 100
            if target > 0
            else (100 if actual["time_needed"] == 0 else 999999)
        )
        if rid == 101:
            status = "—"
        elif 50 <= pct <= 150:
            status = "✅"
        elif 33 <= pct < 50 or 150 < pct <= 300:
            status = "⚠️"
        else:
            status = "🔴"
        pace_rows.append(
            {
                "Rod": f"{rid} ({ROD_NAMES[rid]})",
                "Zone": actual["zone"],
                "Target (min)": round(target, 1),
                "Actual (min)": round(actual["time_needed"], 1),
                "Δ (min)": round(delta, 1),
                "% of target": f"{pct:.0f}%" if rid != 101 else "—",
                "Status": status,
                "Cumul target (min)": round(target_cum[rid], 1),
                "Cumul actual (min)": round(actual["total_minutes"], 1),
                "Bottleneck": actual["bottleneck"] if rid != 101 else "—",
            }
        )

    pace_df = pd.DataFrame(pace_rows)

    def hl_status(val):
        if val == "✅":
            return "background-color: #2a5a3a"
        if val == "⚠️":
            return "background-color: #6a5028"
        if val == "🔴":
            return "background-color: #6a2828"
        return ""

    st.dataframe(
        pace_df.style.map(hl_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
        height=720,
    )

    # Charts
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Target",
                x=[f"{r['Rod']}" for r in pace_rows],
                y=[r["Target (min)"] for r in pace_rows],
                marker_color="#888",
                opacity=0.7,
            )
        )
        fig.add_trace(
            go.Bar(
                name="Actual",
                x=[f"{r['Rod']}" for r in pace_rows],
                y=[r["Actual (min)"] for r in pace_rows],
                marker_color=[ZONE_COLORS[r["Zone"]] for r in pace_rows],
            )
        )
        fig.update_layout(
            title="Time per rod transition",
            yaxis_title="Minutes",
            barmode="group",
            height=400,
            xaxis_tickangle=-45,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                name="Target cumulative",
                x=[r["Rod"] for r in pace_rows],
                y=[r["Cumul target (min)"] for r in pace_rows],
                mode="lines+markers",
                line=dict(color="#888", width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                name="Actual cumulative",
                x=[r["Rod"] for r in pace_rows],
                y=[r["Cumul actual (min)"] for r in pace_rows],
                mode="lines+markers",
                line=dict(color="#FFA01E", width=2),
            )
        )
        fig.update_layout(
            title="Cumulative playtime",
            yaxis_title="Minutes",
            height=400,
            xaxis_tickangle=-45,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-zone totals
    st.divider()
    st.markdown("**Per-zone totals**")
    zone_totals = []
    for zone in ZONES:
        zone_rows = [r for r in pace_rows if r["Zone"] == zone]
        tgt = sum(r["Target (min)"] for r in zone_rows)
        act = sum(r["Actual (min)"] for r in zone_rows)
        zone_totals.append(
            {
                "Zone": zone,
                "Target (min)": round(tgt, 1),
                "Target (hr)": round(tgt / 60, 2),
                "Actual (min)": round(act, 1),
                "Actual (hr)": round(act / 60, 2),
                "Ratio": f"{act/max(tgt,0.01):.2f}×",
            }
        )
    st.dataframe(pd.DataFrame(zone_totals), use_container_width=True, hide_index=True)

    # Solve buttons
    st.divider()
    st.markdown("### 🔧 Auto-tune")
    st.caption(
        "**Solve rod prices** sets each rod's price so the time-to-earn matches the target. "
        "**Solve XP curve** finds the (base, power) pair that aligns level-25/50/85 unlocks "
        "with the target time at rods 106/111/116."
    )

    sc1, sc2 = st.columns(2)

    if sc1.button("💰 Solve rod prices to match target"):
        # For each rod, target_time = (price - cur_gold) / gold_per_min
        # ⟹ price = target_time * gold_per_min + cur_gold_at_that_point
        # Walk through, using actual gold/min from previous rod.
        cur_gold_walk = 0.0
        pending = {}
        for rid in range(101, 121):
            target = target_per_rod[rid]
            if rid == 101:
                new_price = 0
            else:
                prev_rod_for_gold = rs_by_id[rid - 1]
                gold_per_min = prev_rod_for_gold["exp_gold_per_hour"] / 60
                new_price = max(0, int(round(target * gold_per_min + cur_gold_walk)))
                # Update walking gold balance: earned target*g/min, spent price
                cur_gold_walk += target * gold_per_min - new_price
            pending[rid] = new_price
        st.session_state["_pending_rod_prices"] = pending
        st.success("Rod prices updated. Reload tabs to see new values.")
        st.rerun()

    if sc2.button("📈 Solve XP curve (base, power)"):
        # We want the player at level 25 by the time they're buying rod 106,
        # level 50 at rod 111, level 85 at rod 116.
        # Find (base, power) that hits cum[25] ≈ XP-earned-by-rod-106-target,
        # similarly for 50@111 and 85@116. Three constraints, two parameters.
        # Use scipy-free grid search.
        target_25 = target_cum[105] + target_per_rod[106]  # arrived at rod 106
        target_50 = target_cum[110] + target_per_rod[111]
        target_85 = target_cum[115] + target_per_rod[116]

        # Realistic XP earned by each milestone: player uses each rod r for
        # target_per_rod[r+1] minutes (time grinding with r to afford r+1),
        # earning at that rod's XP/min. Sum across all rods used so far.
        def xp_earned_through(last_rod_used):
            total = 0.0
            for r in range(101, last_rod_used + 1):
                mins = target_per_rod[r + 1] if r < 120 else 0.0
                total += mins * rs_by_id[r]["exp_xp_per_hour"] / 60
            return total

        xp_at_25 = xp_earned_through(105)  # arrived at rod 106 → used 101..105
        xp_at_50 = xp_earned_through(110)  # arrived at rod 111 → used 101..110
        xp_at_85 = xp_earned_through(115)  # arrived at rod 116 → used 101..115

        # We want xp_curve(base, power) to produce cum[25]≈xp_at_25, cum[50]≈xp_at_50, cum[85]≈xp_at_85
        # cumulative sum of base * n^power from n=1..N-1
        # Use grid search to minimize squared log-error
        best = (xp_base, xp_power, float("inf"))
        for p_test in np.linspace(1.2, 4.0, 60):
            # Given a power p, solve for base such that cum[50] = xp_at_50 (middle anchor)
            # cum[50] = base * sum(n^p for n=1..49)
            denom = sum(n**p_test for n in range(1, 50))
            if denom == 0:
                continue
            b_test = xp_at_50 / denom
            if b_test < 1:
                continue
            # Compute error at 25 and 85
            c25 = b_test * sum(n**p_test for n in range(1, 25))
            c85 = b_test * sum(n**p_test for n in range(1, 85))
            err = (math.log(c25 / xp_at_25)) ** 2 + (math.log(c85 / xp_at_85)) ** 2
            if err < best[2]:
                best = (b_test, p_test, err)

        new_base, new_power, _ = best
        # Snap to nicer step sizes
        new_base = round(new_base, 1)
        new_power = round(new_power, 2)
        # We can't programmatically update sliders mid-rerun easily, so just print the solve
        st.success(
            f"Suggested XP curve: **base = {new_base:.1f}, power = {new_power:.2f}**"
        )
        st.info(
            f"Time targets: lvl 25 at {target_25:.0f} min, lvl 50 at {target_50:.0f} min, lvl 85 at {target_85:.0f} min.  \n"
            f"XP earned by then (using actual rod progression): "
            f"{xp_at_25:,.0f} / {xp_at_50:,.0f} / {xp_at_85:,.0f}.  \n"
            f"Set sidebar XP base/power to the suggested values manually."
        )


# ════════════════════════════════════════════════════════════════════════
# TAB: MIGRATION (THE PRIMARY TUNING SURFACE)
# ════════════════════════════════════════════════════════════════════════
with tab_mig:
    st.subheader("Migration Ratios — the funnel that pulls players to new zones")
    st.caption(
        "When a player crosses a zone boundary they have a choice: "
        "**stay** in old zone with old tier-5 rod, or **move** to new zone with new tier-1 rod. "
        "Target: 2-3× boost in both gold/hr and XP/hr when moving."
    )

    boundary_rows = []
    for i in range(3):
        old_zone, new_zone = ZONES[i], ZONES[i + 1]
        stay_rod = 101 + i * 5 + 4
        move_rod = 101 + (i + 1) * 5

        stay_xp, stay_gold, _ = expected_per_catch(
            old_zone,
            5,
            drop_rates,
            fish_values,
            xp_per_rarity,
            xp_zone_mult,
            rod_xp_mult[stay_rod],
        )
        move_xp, move_gold, _ = expected_per_catch(
            new_zone,
            1,
            drop_rates,
            fish_values,
            xp_per_rarity,
            xp_zone_mult,
            rod_xp_mult[move_rod],
        )
        gold_ratio = move_gold / max(stay_gold, 0.01)
        xp_ratio = move_xp / max(stay_xp, 0.01)

        boundary_rows.append(
            {
                "Transition": f"{old_zone} → {new_zone}",
                "Stay rod": f"{stay_rod} ({ROD_NAMES[stay_rod]})",
                "Move rod": f"{move_rod} ({ROD_NAMES[move_rod]})",
                "Stay gold/hr": int(stay_gold * catches_per_min * 60),
                "Move gold/hr": int(move_gold * catches_per_min * 60),
                "Gold ratio": gold_ratio,
                "Stay XP/hr": int(stay_xp * catches_per_min * 60),
                "Move XP/hr": int(move_xp * catches_per_min * 60),
                "XP ratio": xp_ratio,
            }
        )

    bdf = pd.DataFrame(boundary_rows)
    display = bdf.copy()
    display["Gold ratio"] = display["Gold ratio"].apply(lambda x: f"{x:.2f}×")
    display["XP ratio"] = display["XP ratio"].apply(lambda x: f"{x:.2f}×")
    for col in ["Stay gold/hr", "Move gold/hr", "Stay XP/hr", "Move XP/hr"]:
        display[col] = display[col].apply(lambda x: f"{x:,}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Stay",
                x=[r["Transition"] for r in boundary_rows],
                y=[r["Stay gold/hr"] for r in boundary_rows],
                marker_color="#888",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Move",
                x=[r["Transition"] for r in boundary_rows],
                y=[r["Move gold/hr"] for r in boundary_rows],
                marker_color="#4CAF50",
            )
        )
        fig.update_layout(
            title="Gold/hr at zone boundaries",
            yaxis_type="log",
            height=380,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Stay",
                x=[r["Transition"] for r in boundary_rows],
                y=[r["Stay XP/hr"] for r in boundary_rows],
                marker_color="#888",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Move",
                x=[r["Transition"] for r in boundary_rows],
                y=[r["Move XP/hr"] for r in boundary_rows],
                marker_color="#FFA01E",
            )
        )
        fig.update_layout(
            title="XP/hr at zone boundaries",
            yaxis_type="log",
            height=380,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    for row in boundary_rows:
        g, x = row["Gold ratio"], row["XP ratio"]
        msg = f"**{row['Transition']}** — Gold {g:.2f}× · XP {x:.2f}×  "
        if g < 1.5:
            st.error(msg + "🚨 Gold funnel BROKEN")
        elif g < 2.0:
            st.warning(msg + "⚠️ Gold funnel weak")
        elif g > 5.0:
            st.warning(msg + "⚠️ Gold funnel excessive — old zone feels useless")
        else:
            st.success(msg + "✅ Gold funnel in range")

    st.divider()
    st.markdown("**All 20 rods — gold/hr and XP/hr in their home zone**")
    sdf = rod_df.copy()
    sdf["exp_gold_per_hour"] = sdf["exp_gold_per_hour"].round(0).astype(int)
    sdf["exp_xp_per_hour"] = sdf["exp_xp_per_hour"].round(0).astype(int)
    sdf["exp_gold_per_catch"] = sdf["exp_gold_per_catch"].round(1)
    sdf["exp_xp_per_catch"] = sdf["exp_xp_per_catch"].round(1)
    st.dataframe(
        sdf.rename(
            columns={
                "rod_id": "ID",
                "name": "Name",
                "zone": "Zone",
                "tier": "T",
                "rod_xp_mult": "XP×",
                "exp_xp_per_catch": "XP/catch",
                "exp_gold_per_catch": "Gold/catch",
                "exp_xp_per_hour": "XP/hr",
                "exp_gold_per_hour": "Gold/hr",
                "price": "Price",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=720,
    )


# ════════════════════════════════════════════════════════════════════════
# TAB: LEVEL CURVE
# ════════════════════════════════════════════════════════════════════════
with tab_curve:
    st.subheader("Level Curve")
    st.caption(f"`XP_TO_NEXT_LEVEL[n] = floor({xp_base} × n^{xp_power})`")

    levels = list(range(1, level_cap + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=levels,
            y=curve,
            mode="lines",
            name="XP for next level",
            line=dict(color="#4A90D9", width=2),
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, level_cap + 2)),
            y=cum[1 : level_cap + 2],
            mode="lines",
            name="Cumulative XP",
            line=dict(color="#FFA01E", width=2, dash="dash"),
            yaxis="y2",
        )
    )
    for z, lvl in zone_req.items():
        if lvl <= level_cap:
            fig.add_vline(
                x=lvl,
                line=dict(color=ZONE_COLORS[z], width=1, dash="dot"),
                annotation_text=f"{z}",
                annotation_position="top",
            )
    fig.update_layout(
        xaxis_title="Level",
        yaxis=dict(title="XP for level", side="left"),
        yaxis2=dict(title="Cumulative XP", side="right", overlaying="y"),
        height=450,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Time to reach key levels** (using tier-5 rod of current zone)")
    targets = sorted(set(list(zone_req.values()) + [10, 25, 50, 85, 100]))
    targets = [t for t in targets if t <= level_cap]
    rows = []
    for tlvl in targets:
        xp_needed = cum[tlvl]
        cur_zone = "Pirate"
        for z in ZONES:
            if tlvl >= zone_req[z]:
                cur_zone = z
        rid = 101 + ZONES.index(cur_zone) * 5 + 4
        xp_per_min = rs_by_id[rid]["exp_xp_per_hour"] / 60
        mins = xp_needed / xp_per_min if xp_per_min > 0 else float("inf")
        rows.append(
            {
                "Target": tlvl,
                "Zone": cur_zone,
                "Rod": rid,
                "Cumulative XP": int(xp_needed),
                "XP/min": f"{xp_per_min:.1f}",
                "Minutes": f"{mins:.1f}",
                "Hours": f"{mins / 60:.2f}",
                f"Sessions ({minutes_per_session}m)": f"{mins / minutes_per_session:.1f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════
# TAB: PROGRESSION
# ════════════════════════════════════════════════════════════════════════
with tab_prog:
    st.subheader("Progression & Affordability")
    st.caption(
        "Same simulation as the Pacing tab, viewed as a flat table with bottleneck analysis."
    )

    prog_display = []
    for r in sim_rows:
        prog_display.append(
            {
                "Buy rod": f"{r['rod_id']} ({r['name']})",
                "Zone": r["zone"],
                "Lvl gate": r["lvl_gate"],
                "Price": r["price"],
                "Time spent (min)": round(r["time_needed"], 1),
                "Total time (min)": round(r["total_minutes"], 1),
                "Total (hr)": round(r["total_minutes"] / 60, 2),
                "Lvl on buy": r["level_on_buy"],
                "Gold after": r["gold_after"],
                "Bottleneck": r["bottleneck"],
            }
        )
    sim_df = pd.DataFrame(prog_display)

    def hl(val):
        if val == "Level":
            return "background-color: #5a3a3a"
        if val == "Gold":
            return "background-color: #3a4a5a"
        return ""

    st.dataframe(
        sim_df.style.map(hl, subset=["Bottleneck"]),
        use_container_width=True,
        hide_index=True,
        height=720,
    )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=sim_df["Buy rod"],
                y=sim_df["Total time (min)"],
                marker_color=[ZONE_COLORS[z] for z in sim_df["Zone"]],
            )
        )
        fig.update_layout(
            title="Cumulative time to buy each rod", height=400, xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=sim_df["Buy rod"],
                y=sim_df["Time spent (min)"],
                marker_color=[ZONE_COLORS[z] for z in sim_df["Zone"]],
            )
        )
        fig.update_layout(
            title="Time between each rod", height=400, xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

    walls = sim_df[sim_df["Time spent (min)"] > 60]
    skips = sim_df[sim_df["Time spent (min)"] < 1]
    if not walls.empty:
        st.error(f"🧱 **Walls** (>60 min): {', '.join(walls['Buy rod'].tolist())}")
    if not skips.empty:
        st.warning(f"⚡ **Free rods** (<1 min): {', '.join(skips['Buy rod'].tolist())}")


# ════════════════════════════════════════════════════════════════════════
# TAB: EXPORT
# ════════════════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("Export Tuned Values")

    st.markdown(
        "**FishData.lua** (base_values only — paste names/weights manually or via RBXMX re-import)"
    )
    lua = "-- Generated by XP & Economy Simulator\n"
    lua += f"-- Anchor: {pirate_common} · Z: {zone_inflation} · Mults: {[rarity_mult[r] for r in RARITIES]}\n\n"
    lua += "local FishData = {\n\n"
    for zone in ZONES:
        lua += f"\t-- // {zone}\n"
        for rarity in RARITIES:
            for fid in FISH_LIST[zone][rarity]:
                bv = int(round(fish_values[fid]))
                lua += (
                    f"\t[{fid}] = {{ item_id = {fid}, "
                    f"rarity = _G.EnumModule.Rarity.{rarity.upper()}, "
                    f"base_value = {bv} }},\n"
                )
            lua += "\n"
    lua += "}\n\nreturn FishData\n"
    st.download_button(
        "📥 Download FishData.lua",
        data=lua,
        file_name="FishData.lua",
        mime="text/plain",
    )
    with st.expander("Preview"):
        st.code(
            lua[:3000] + ("\n... (truncated)" if len(lua) > 3000 else ""),
            language="lua",
        )

    st.divider()
    st.markdown("**FishingConfig snippets**")

    st.code(
        f"-- XP curve\nFishingConfig.XP_TO_NEXT_LEVEL = {{}}\ndo\n"
        f"\tfor n = 1, 200 do\n"
        f"\t\tFishingConfig.XP_TO_NEXT_LEVEL[n] = math.floor({xp_base} * (n ^ {xp_power}))\n"
        f"\tend\nend",
        language="lua",
    )

    lua_xp = "FishingConfig.XP_PER_RARITY = {\n"
    for r in RARITIES:
        lua_xp += f"\t{r:<10} = {int(xp_per_rarity[r])},\n"
    lua_xp += "}"
    st.code(lua_xp, language="lua")

    lua_xpz = "-- XP zone multiplier (not in current FishingConfig — would need to be added)\nXP_ZONE_MULT = {\n"
    for z in ZONES:
        lua_xpz += f"\t{z:<9} = {xp_zone_mult[z]},\n"
    lua_xpz += "}"
    st.code(lua_xpz, language="lua")

    lua_rxp = "FishingConfig.XP_MULTIPLIER_BY_ROD_ID = {\n"
    for rid in range(101, 121):
        lua_rxp += f"\t[{rid}] = {rod_xp_mult[rid]:.2f},\n"
    lua_rxp += "}"
    st.code(lua_rxp, language="lua")

    lua_req = "FishingConfig.ZONE_LEVEL_REQUIREMENTS = {\n"
    for z in ZONES:
        lua_req += f"\t{z:<9} = {zone_req[z]},\n"
    lua_req += "}"
    st.code(lua_req, language="lua")

    lua_rp = "-- Rod prices (paste into RodData.lua):\n"
    for rid in range(101, 121):
        lua_rp += f"-- [{rid}] {ROD_NAMES[rid]:<10s} price = {rod_prices[rid]}\n"
    st.code(lua_rp, language="lua")

    st.divider()
    state = {
        "pricing": {
            "pirate_common_anchor": pirate_common,
            "zone_inflation": zone_inflation,
            "within_spread": within_spread,
            "rarity_mult": rarity_mult,
        },
        "xp": {
            "base": xp_base,
            "power": xp_power,
            "per_rarity": xp_per_rarity,
            "zone_mult": xp_zone_mult,
            "rod_mult": rod_xp_mult,
        },
        "rod_prices": rod_prices,
        "drop_rates": drop_rates,
        "zone_req": zone_req,
        "catches_per_min": catches_per_min,
    }
    st.download_button(
        "📥 Download tuning.json",
        data=json.dumps(state, indent=2),
        file_name="ocean_quest_tuning.json",
        mime="application/json",
    )
