"""
World Map — All Player Paths

Reads `data->'PlayerMetrics'->'locationPath'` from Postgres for every player,
subsamples client-side, and renders all paths as a single Plotly figure.

Lazily loaded so the main dashboard stays fast.
"""

import os
import streamlit as st
import plotly.graph_objects as go

import psycopg
from psycopg.rows import dict_row

st.set_page_config(page_title="World Map", layout="wide", page_icon="🗺️")

# ── Constants (mirrored from dashboard.py) ────────────────────────────────────
STAFF_USER_IDS = {
    7796282,
    56651650,
    1068573686,
    10782708602,
    10784724184,
    10811787773,
}

MAP_ISLANDS = [
    {"name": "Pirate", "x": -420, "z": -200, "color": "#D4763B"},
    {"name": "Greek", "x": 2555, "z": 5110, "color": "#4A90D9"},
    {"name": "Japanese", "x": -160, "z": 10200, "color": "#D94A6B"},
    {"name": "Viking", "x": 7292, "z": 15790, "color": "#6B4AD9"},
]


# ── Postgres connection (cached resource) ─────────────────────────────────────
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
        st.error("DATABASE_URL is not configured.")
        st.stop()
    return url


@st.cache_resource(show_spinner=False)
def get_conn() -> psycopg.Connection:
    return psycopg.connect(_database_url(), row_factory=dict_row)


# ── Cached path loader ────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading player paths…")
def _load_paths(staff_ids: frozenset, subsample: int) -> list:
    """Pulls the locationPath JSON array for every non-staff `status='ok'` player.

    Postgres returns parsed JSON, so we subsample and reshape in Python.
    """
    sql = (
        "SELECT user_id, "
        "       COALESCE(username, display_name, user_id::text) AS pname, "
        "       data->'PlayerMetrics'->'locationPath' AS lp "
        "FROM players "
        "WHERE status = 'ok' "
        "  AND data->'PlayerMetrics' ? 'locationPath' "
        "  AND user_id NOT IN (" + ",".join(["%s"] * len(staff_ids)) + ")"
    )
    with get_conn().cursor() as cur:
        cur.execute(sql, tuple(int(s) for s in staff_ids))
        rows = cur.fetchall()

    result = []
    for row in rows:
        lp = row.get("lp")
        if not isinstance(lp, list) or len(lp) < 2:
            continue
        result.append({"player": row["pname"], "path": lp[::subsample]})
    return result


# ── Colour helper ─────────────────────────────────────────────────────────────
def _path_color(t: float) -> str:
    r = int(40 + t * 195)
    g = int(190 - t * 190)
    return f"rgb({r},{g},40)"


_GREEN = _path_color(0.0)
_GRAD_SECS = 120
_N_SEGS = 6  # fewer gradient segments — still looks good


# ── Build figure (cached on path data) ───────────────────────────────────────
@st.cache_data(show_spinner="Building map…")
def _build_figure(paths: tuple) -> go.Figure:
    """
    All paths rendered as a single Scatter trace per colour band using
    None separators — orders of magnitude fewer trace objects.
    """
    # Separate green (old) from gradient (recent 120 s) segments
    green_xs, green_ys = [], []

    # For gradient we build N_SEGS colour buckets across all players
    seg_data: list[tuple[list, list, str]] = []  # (xs, ys, colour)

    for pp in paths:
        pts = pp["path"]
        if len(pts) < 2:
            continue

        t_start = pts[0].get("t", 0)
        t_end = pts[-1].get("t", 0)

        if (t_end - t_start) <= _GRAD_SECS:
            grad_idx = 0
        else:
            grad_time = t_end - _GRAD_SECS
            grad_idx = next(
                (i for i, p in enumerate(pts) if p.get("t", 0) >= grad_time),
                len(pts) - 1,
            )

        # Solid-green portion
        if grad_idx > 0:
            for p in pts[: grad_idx + 1]:
                green_xs.append(p["x"])
                green_ys.append(p["z"])
            green_xs.append(None)
            green_ys.append(None)

        # Gradient portion
        gpts = pts[grad_idx:]
        ng = len(gpts)
        if ng < 2:
            continue
        gt0 = gpts[0].get("t", 0)
        gt1 = gpts[-1].get("t", 0)
        grange = max(1, gt1 - gt0)
        n_segs = min(_N_SEGS, ng - 1)
        for s in range(n_segs):
            i0 = int(s * ng / n_segs)
            i1 = min(int((s + 1) * ng / n_segs) + 1, ng)
            color_t = (gpts[i0].get("t", 0) - gt0) / grange
            col = _path_color(color_t)
            seg_xs = [p["x"] for p in gpts[i0:i1]] + [None]
            seg_ys = [p["z"] for p in gpts[i0:i1]] + [None]
            seg_data.append((seg_xs, seg_ys, col))

    # Collapse same-colour segments into one trace each
    from collections import defaultdict

    color_buckets: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for xs, ys, col in seg_data:
        color_buckets[col][0].extend(xs)
        color_buckets[col][1].extend(ys)

    fig = go.Figure()

    if green_xs:
        fig.add_trace(
            go.Scatter(
                x=green_xs,
                y=green_ys,
                mode="lines",
                line=dict(color=_GREEN, width=1.2),
                opacity=0.6,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for col, (xs, ys) in color_buckets.items():
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=col, width=1.5),
                opacity=0.75,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Island markers
    for isl in MAP_ISLANDS:
        fig.add_trace(
            go.Scatter(
                x=[isl["x"]],
                y=[isl["z"]],
                mode="markers+text",
                marker=dict(
                    size=18,
                    color=isl["color"],
                    symbol="diamond",
                    line=dict(width=2, color="white"),
                ),
                text=[isl["name"]],
                textposition="top center",
                textfont=dict(size=12, color=isl["color"]),
                showlegend=False,
                hovertemplate=f"<b>{isl['name']}</b><extra></extra>",
            )
        )

    fig.update_layout(
        margin=dict(t=10, b=10),
        height=900,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,8,24,0.95)",
        font_color="#ccd6f6",
        xaxis=dict(
            title="X",
            gridcolor="#1a1a3e",
            zerolinecolor="#2a2a5e",
            autorange="reversed",
        ),
        yaxis=dict(
            title="Z", gridcolor="#1a1a3e", zerolinecolor="#2a2a5e", scaleanchor="x"
        ),
    )
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🗺️ World Map — All Player Paths")

st.sidebar.title("Map Options")
subsample = st.sidebar.slider(
    "Path density (1 = full, higher = faster)",
    min_value=1,
    max_value=10,
    value=3,
    help="Keep every Nth location sample. Higher = fewer points, faster render.",
)

paths = _load_paths(frozenset(STAFF_USER_IDS), subsample)

if not paths:
    st.warning("No location path data found.")
    st.stop()

total_pts = sum(len(p["path"]) for p in paths)
st.sidebar.markdown(f"**Players with paths:** `{len(paths)}`")
st.sidebar.markdown(f"**Total points:** `{total_pts:,}`")
st.sidebar.markdown(f"**Traces in figure:** `~{len(paths) * _N_SEGS}`")

# Cache key includes subsample so changing the slider re-builds
fig = _build_figure(
    tuple(
        (p["player"], tuple((pt["x"], pt["z"], pt.get("t", 0)) for pt in p["path"]))
        for p in paths
    )
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"{len(paths)} players · {total_pts:,} position samples · every {subsample}{'st' if subsample==1 else 'nd' if subsample==2 else 'rd' if subsample==3 else 'th'} point shown"
)
