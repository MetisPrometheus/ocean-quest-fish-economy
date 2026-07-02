"""
Ocean Quest level curve simulator.

Given:
  - XP_PER_RARITY_BY_ZONE
  - DROP_RATES_BY_ROD_TIER
  - XP_MULTIPLIER_BY_TIER
  - XP_TO_NEXT_LEVEL curve
  - casts_per_min
  - rod-progression model (when does the player swap rods?)

Compute: minutes to reach each level, and minutes spent in each zone.

Player rod progression model:
  Players don't sit on T1 — they upgrade fast in early zones. Model: spend
  X% of zone-time on each rod tier. Default: 10%/15%/25%/25%/25% (T1->T5).
  This means a player spends most of the zone with T3-T5 rods.

  Refine later by modeling rod cost vs fish revenue if needed.
"""

import math

# =====================================================================
# CONFIG (mirrored from FishingConfig.lua)
# =====================================================================

XP_PER_RARITY_BY_ZONE = {
    "Pirate": {
        "Poor": 5,
        "Common": 12,
        "Uncommon": 22,
        "Rare": 33,
        "Epic": 50,
        "Legendary": 78,
        "Mythic": 115,
    },
    "Greek": {
        "Poor": 120,
        "Common": 140,
        "Uncommon": 200,
        "Rare": 300,
        "Epic": 420,
        "Legendary": 580,
        "Mythic": 800,
    },
    "Japanese": {
        "Poor": 650,
        "Common": 760,
        "Uncommon": 950,
        "Rare": 1150,
        "Epic": 1450,
        "Legendary": 1800,
        "Mythic": 2400,
    },
    "Viking": {
        "Poor": 1650,
        "Common": 1900,
        "Uncommon": 2350,
        "Rare": 2900,
        "Epic": 3600,
        "Legendary": 4500,
        "Mythic": 6000,
    },
}

DROP_RATES_BY_ROD_TIER = {
    1: {
        "Poor": 0.52,
        "Common": 0.45,
        "Uncommon": 0.03,
        "Rare": 0,
        "Epic": 0,
        "Legendary": 0,
        "Mythic": 0,
    },
    2: {
        "Poor": 0.12,
        "Common": 0.45,
        "Uncommon": 0.395,
        "Rare": 0.03,
        "Epic": 0.005,
        "Legendary": 0,
        "Mythic": 0,
    },
    3: {
        "Poor": 0.04,
        "Common": 0.10,
        "Uncommon": 0.50,
        "Rare": 0.325,
        "Epic": 0.035,
        "Legendary": 0,
        "Mythic": 0,
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

XP_MULTIPLIER_BY_TIER = {1: 1.0, 2: 1.1, 3: 1.2, 4: 1.3, 5: 1.4}

ZONE_LEVEL_REQUIREMENTS = {
    "Pirate": 1,
    "Greek": 25,
    "Japanese": 50,
    "Viking": 85,
}

ZONE_ORDER = ["Pirate", "Greek", "Japanese", "Viking"]


# Current level curve in config: xp(n) = floor(10 * n^1.6)
def current_xp_curve(n: int) -> int:
    return int(10 * (n**1.6))


# =====================================================================
# SIMULATION PARAMETERS
# =====================================================================

CASTS_PER_MIN = 8

# Fraction of zone time the player spends on each rod tier (T1..T5).
# Front-loaded: players grind early-zone XP on T1-T2 (cheaper), then upgrade
# faster as fish sell prices ramp.
TIME_PER_ROD_TIER = [0.10, 0.15, 0.25, 0.25, 0.25]

# Target playtimes (minutes from game start) to hit each zone unlock level.
TARGETS = {
    "Greek_unlock": (25, 40),  # level 25 by minute 40
    "Japanese_unlock": (50, 150),  # level 50 by minute 150
    "Viking_unlock": (85, 600),  # level 85 by minute 600
}

# =====================================================================
# EV COMPUTATION
# =====================================================================


def ev_per_cast(zone: str, tier: int) -> float:
    """Expected XP per cast for a given rod (zone + tier)."""
    xp_table = XP_PER_RARITY_BY_ZONE[zone]
    drops = DROP_RATES_BY_ROD_TIER[tier]
    mult = XP_MULTIPLIER_BY_TIER[tier]
    ev = sum(drops[r] * xp_table[r] for r in drops)
    return ev * mult


def avg_ev_for_zone(zone: str) -> float:
    """Time-weighted average EV per cast across the zone, given the time-per-tier model."""
    return sum(TIME_PER_ROD_TIER[t - 1] * ev_per_cast(zone, t) for t in range(1, 6))


# =====================================================================
# LEVEL CURVE TOOLS
# =====================================================================


def cumulative_xp(curve_fn, level: int) -> int:
    """Total XP needed to *reach* `level` from level 1."""
    return sum(curve_fn(n) for n in range(1, level))


def simulate_time_to_level(curve_fn, target_level: int) -> float:
    """
    Walks levels 1..target_level. Determines which zone player is currently
    leveling in (based on the curve's zone-unlock alignment from ZONE_LEVEL_REQUIREMENTS).
    Uses that zone's avg EV/cast to compute minutes per level.

    Returns total minutes.
    """
    total_minutes = 0.0
    for n in range(1, target_level):
        # Which zone is the player playing in at level n?
        zone = current_zone_for_level(n)
        ev = avg_ev_for_zone(zone)
        xp_for_this_level = curve_fn(n)
        minutes = xp_for_this_level / (ev * CASTS_PER_MIN)
        total_minutes += minutes
    return total_minutes


def current_zone_for_level(level: int) -> str:
    """Player is in the highest-tier zone they've unlocked."""
    current = "Pirate"
    for z in ZONE_ORDER:
        if level >= ZONE_LEVEL_REQUIREMENTS[z]:
            current = z
    return current


# =====================================================================
# CURVE FITTING
# =====================================================================


def fit_curve_to_targets():
    """
    Fit a piecewise-linear-within-zone level curve that hits the time targets.

    For each zone band:
      - Levels in band: [zone_start_level, next_zone_start_level - 1]
      - Total XP for the band = avg_EV * casts_per_min * target_minutes_in_zone
      - Distribute across levels with linear growth: xp(n) = base + slope*(n - band_start)
      - Choose base & slope so first level in band ≈ avg_EV*casts_per_min*1.5 min (gentle ramp)
        and the sum across the band equals target total XP.

    Returns a list of XP-per-level values (1-indexed).
    """
    minutes_so_far = 0.0
    curve = [0] * 201  # 1..200

    # Add a synthetic end-cap so Viking has bounds
    zone_starts = [(z, ZONE_LEVEL_REQUIREMENTS[z]) for z in ZONE_ORDER]
    zone_starts.append(("END", 121))  # cap Viking band at level 120

    target_minutes_total = {
        "Pirate": TARGETS["Greek_unlock"][1],  # 40
        "Greek": TARGETS["Japanese_unlock"][1] - TARGETS["Greek_unlock"][1],  # 110
        "Japanese": TARGETS["Viking_unlock"][1] - TARGETS["Japanese_unlock"][1],  # 450
        "Viking": 1800,  # speculative — ~30hr through level 120
    }

    for i, (zone, start_lvl) in enumerate(zone_starts[:-1]):
        end_lvl = zone_starts[i + 1][1] - 1
        num_levels = end_lvl - start_lvl + 1
        ev = avg_ev_for_zone(zone)
        target_minutes = target_minutes_total[zone]
        total_xp = ev * CASTS_PER_MIN * target_minutes
        avg_xp_per_level = total_xp / num_levels

        # Slight growth within band: first level = 60% of avg, last level = 140% of avg
        # Linear interpolation. Sum stays = total_xp.
        for j in range(num_levels):
            n = start_lvl + j
            frac = (j / max(1, num_levels - 1)) if num_levels > 1 else 0.5
            multiplier = 0.6 + 0.8 * frac  # 0.6 -> 1.4
            curve[n] = int(avg_xp_per_level * multiplier)

    # Past level 120: continue growing roughly geometrically for endgame
    last_xp = curve[120]
    for n in range(121, 201):
        curve[n] = int(last_xp * (1.04 ** (n - 120)))

    return curve


def fitted_curve(n: int, curve_list) -> int:
    if 1 <= n < len(curve_list):
        return curve_list[n]
    return curve_list[-1]


# =====================================================================
# REPORTING
# =====================================================================


def print_ev_summary():
    print("=" * 70)
    print("EV PER CAST BY ROD")
    print("=" * 70)
    print(f"{'Rod':<5} {'Zone':<10} {'Tier':<5} {'Mult':<6} {'EV/cast':>12}")
    print("-" * 70)
    rod_id = 101
    for zone in ZONE_ORDER:
        for tier in range(1, 6):
            ev = ev_per_cast(zone, tier)
            mult = XP_MULTIPLIER_BY_TIER[tier]
            print(f"{rod_id:<5} {zone:<10} {tier:<5} {mult:<6} {ev:>12,.2f}")
            rod_id += 1
    print()
    print(f"{'Zone':<10} {'Avg EV/cast (time-weighted)':>30}  {'XP/min':>10}")
    print("-" * 60)
    for zone in ZONE_ORDER:
        avg = avg_ev_for_zone(zone)
        xpm = avg * CASTS_PER_MIN
        print(f"{zone:<10} {avg:>30,.2f}  {xpm:>10,.0f}")
    print()


def print_curve_table(curve_fn, label):
    print("=" * 70)
    print(f"LEVEL CURVE: {label}")
    print("=" * 70)
    print(f"casts/min = {CASTS_PER_MIN}, time-per-rod-tier = {TIME_PER_ROD_TIER}")
    print()
    print(
        f"{'Level':<7} {'Zone':<10} {'XP for level':>14} {'Cumul XP':>14} {'Min to level':>14} {'Cumul min':>12}"
    )
    print("-" * 90)

    cumul_xp = 0
    cumul_min = 0
    milestones = {
        25: "GREEK UNLOCK",
        50: "JAPANESE UNLOCK",
        85: "VIKING UNLOCK",
        120: "ENDGAME",
    }
    show_levels = list(range(1, 11)) + list(range(15, 121, 5))
    for n in range(1, 121):
        xp_for_this = curve_fn(n)
        zone = current_zone_for_level(n)
        ev = avg_ev_for_zone(zone)
        min_for_this = xp_for_this / (ev * CASTS_PER_MIN)

        cumul_xp += xp_for_this
        cumul_min += min_for_this

        marker = ""
        if (n + 1) in milestones:
            marker = " <-- " + milestones[n + 1]
        if n in show_levels or (n + 1) in milestones:
            print(
                f"{n:<7} {zone:<10} {xp_for_this:>14,} {cumul_xp:>14,} {min_for_this:>14.2f} {cumul_min:>12.1f}{marker}"
            )
    print()


def print_milestone_check(curve_fn, label):
    print(f"Milestone check ({label}):")
    for name, (target_lvl, target_min) in TARGETS.items():
        actual_min = simulate_time_to_level(curve_fn, target_lvl)
        delta = actual_min - target_min
        status = "OK" if abs(delta) / max(target_min, 1) < 0.15 else "OFF"
        print(
            f"  {name:<20} target lvl {target_lvl:>3} @ {target_min:>4}min  actual: {actual_min:>6.1f}min  delta {delta:+.1f}min  [{status}]"
        )
    print()


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print_ev_summary()

    print("=" * 70)
    print("CURRENT CONFIG: xp(n) = floor(10 * n^1.6)")
    print("=" * 70)
    print_milestone_check(current_xp_curve, "current curve")

    fitted = fit_curve_to_targets()
    print("=" * 70)
    print("FITTED CURVE (piecewise per zone, linear growth within zone)")
    print("=" * 70)
    print_milestone_check(lambda n: fitted_curve(n, fitted), "fitted curve")

    print_curve_table(lambda n: fitted_curve(n, fitted), "FITTED")

    # Print as Lua-ready table
    print("=" * 70)
    print("FITTED CURVE AS LUA TABLE")
    print("=" * 70)
    print("FishingConfig.XP_TO_NEXT_LEVEL = {")
    for n in range(1, 121):
        print(f"    [{n}] = {fitted[n]},")
    print("    -- endgame (levels 121-200) extend geometrically")
    for n in range(121, 201, 10):
        print(f"    [{n}] = {fitted[n]},")
    print("}")
