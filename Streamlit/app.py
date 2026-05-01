"""
app.py — xT Scouting Application
Run:  streamlit run Streamlit/app.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from constants import GROUP_TO_FULL_POSITIONS, GROUPED_POSITION_MAP
from viz import (
    BG_CARD, MINT, GREEN, AMBER, RED, DIM, WHITE,
    plot_xt_heatmap,
    plot_vs_average_pair,
    plot_top_plays,
    plot_comparison_trio,
    plot_rank_bar,
    plot_team_rank_bar,
    _lookup_xt,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
STATS_PATH  = Path(__file__).parent / "player_xt_stats.parquet"
EVENTS_PATH = Path(__file__).parent / "player_events.parquet"
XT_CSV      = ROOT / "Models" / "xT" / "xt.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aidea",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* Main backgrounds */
  .stApp, .main, [data-testid="stAppViewContainer"] {{
      background-color: #06000f;
      color: {WHITE};
  }}
  [data-testid="stSidebar"] {{ background-color: #0a001f; }}
  [data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

  /* Headers */
  h1, h2, h3 {{ color: {MINT} !important; }}
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: {MINT} !important; }}

  /* Metric cards — smaller values so long names fit, with wrapping */
  [data-testid="stMetricLabel"] {{
      color: {DIM} !important;
      font-size: 0.75rem;
  }}
  [data-testid="stMetricValue"] {{
      color: {WHITE} !important;
      font-weight: 700;
      font-size: 1rem !important;
      white-space: normal !important;
      word-break: break-word;
      line-height: 1.3;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ background-color: {BG_CARD}; border-radius: 8px; }}
  .stTabs [data-baseweb="tab"] {{ color: {DIM}; padding: 8px 20px; }}
  .stTabs [aria-selected="true"] {{
      color: {MINT} !important;
      border-bottom: 3px solid {MINT};
      background-color: transparent;
  }}

  /* Selectbox / multiselect */
  .stSelectbox label, .stMultiSelect label, .stSlider label {{ color: {DIM} !important; }}
  [data-baseweb="select"] {{ background-color: {BG_CARD} !important; }}

  /* Multiselect tags */
  [data-baseweb="tag"] {{
      white-space: normal !important;
      word-break: break-word;
      height: auto !important;
      padding: 2px 8px !important;
      line-height: 1.4;
      background-color: #002d28 !important;
  }}
  [data-baseweb="tag"] span {{
      white-space: normal !important;
      word-break: break-word;
      color: {MINT} !important;
  }}
  /* Tag close (×) button — match MINT accent */
  [data-baseweb="tag"] [role="presentation"] {{
      color: {MINT} !important;
      fill: {MINT} !important;
  }}
  [data-baseweb="tag"] svg {{
      fill: {MINT} !important;
      color: {MINT} !important;
  }}

  /* Dataframe */
  [data-testid="stDataFrame"] {{ background-color: {BG_CARD}; border-radius: 8px; }}

  /* Toggle */
  .stToggle label {{ color: {WHITE} !important; }}

  /* Divider */
  hr {{ border-color: #3a1a5c; margin: 1rem 0; }}

  /* Caption */
  .stCaption {{ color: {DIM} !important; }}
</style>
""", unsafe_allow_html=True)


# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def load_stats() -> pd.DataFrame:
    if not STATS_PATH.exists():
        st.error("❌ player_xt_stats.parquet not found. Run `python Streamlit/data_prep.py` first.")
        st.stop()
    return pd.read_parquet(STATS_PATH)


@st.cache_data(ttl=86400)
def load_xt_grid() -> np.ndarray:
    return np.loadtxt(XT_CSV, delimiter=",")


@st.cache_data(ttl=86400)
def load_all_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(EVENTS_PATH)


def get_player_events(all_events: pd.DataFrame, player_id: int) -> pd.DataFrame:
    return all_events[all_events["player_id"] == player_id].copy()


def enrich_events(df: pd.DataFrame, xt_grid: np.ndarray) -> pd.DataFrame:
    """Ensure start_xt / end_xt / added_xt are present (in case parquet is old)."""
    df = df.copy()
    for col in ["x", "y", "end_x", "end_y", "start_xt", "end_xt", "added_xt"]:
        if col not in df.columns:
            df[col] = np.nan
    # Recompute start_xt if missing
    if df["start_xt"].isna().all():
        df["start_xt"], _ = _lookup_xt(df["x"].values, df["y"].values, xt_grid)
    return df


# ── Cached per-player + per-peer event slices ─────────────────────────────────
# Defined after load_all_events() so closures capture the loaded data.
# show_spinner=False keeps the UI clean; figures are cached for 24 h.

@st.cache_data(ttl=86400, show_spinner=False)
def _player_ev(player_id: int) -> pd.DataFrame:
    """Enriched events for one player, cached by player_id."""
    return enrich_events(
        all_events[all_events["player_id"] == player_id].copy(), xt_grid
    )


@st.cache_data(ttl=86400, show_spinner=False)
def _peer_ev(peer_ids: tuple) -> pd.DataFrame:
    """Enriched events for a peer group, cached by the sorted player-id tuple."""
    return enrich_events(
        all_events[all_events["player_id"].isin(peer_ids)].copy(), xt_grid
    )


@st.cache_data(ttl=86400, show_spinner=False)
def _peer_avg_xt(peer_ids: tuple) -> float:
    ev = all_events[all_events["player_id"].isin(peer_ids)]
    return float(ev["added_xt"].clip(lower=0).mean()) if len(ev) else 0.0


# ── Cached figure factories ────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _fig_heatmap(player_id: int, event_type: str, use_end: bool, title: str):
    return plot_xt_heatmap(
        _player_ev(player_id), xt_grid,
        event_type=event_type, use_end=use_end, title=title,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def _fig_vs_avg(player_id: int, peer_ids: tuple,
                event_type: str, use_end: bool,
                row_title: str, player_name: str):
    return plot_vs_average_pair(
        _player_ev(player_id), _peer_ev(peer_ids), xt_grid,
        event_type=event_type, use_end=use_end,
        row_title=row_title, player_name=player_name,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def _fig_top_plays(player_id: int, title: str, position_avg_xt=None):
    return plot_top_plays(
        _player_ev(player_id), xt_grid,
        position_avg_xt=position_avg_xt, title=title,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def _fig_comparison(p1_id: int, p2_id: int,
                    p1_name: str, p2_name: str,
                    event_type: str, use_end: bool, title: str):
    return plot_comparison_trio(
        _player_ev(p1_id), _player_ev(p2_id),
        p1_name, p2_name,
        event_type=event_type, use_end=use_end, title=title,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def _fig_rank_bar(player_dict: dict, peers: pd.DataFrame, metric: str, label: str):
    return plot_rank_bar(pd.Series(player_dict), peers, metric, label)


@st.cache_data(ttl=86400, show_spinner=False)
def _fig_team_rank_bar(agg_df: pd.DataFrame, team_name: str, metric: str, label: str):
    return plot_team_rank_bar(agg_df, team_name, metric, label)


# ── Percentile helpers ────────────────────────────────────────────────────────

def pct_rank(value: float, series: pd.Series) -> float:
    s = series.dropna()
    return round((s < value).mean() * 100, 1) if len(s) > 0 else 50.0


def pct_label(pct: float) -> str:
    top = 100 - pct
    color = GREEN if top <= 25 else (AMBER if top <= 50 else RED)
    return f'<span style="color:{color};font-weight:700">Top {top:.0f}%</span> ({pct:.0f}th pct.)'


# ══════════════════════════════════════════════════════════════════════════════
#  Load base data
# ══════════════════════════════════════════════════════════════════════════════

stats      = load_stats()
xt_grid    = load_xt_grid()
all_events = load_all_events()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(f"## ⚽  Aidea")
st.sidebar.markdown("---")

# ── Position list helpers ──────────────────────────────────────────────────────

def _pos_options(df, list_col, fallback_col):
    """All unique individual positions across every player's position list."""
    if list_col in df.columns:
        opts = set()
        for p in df[list_col].dropna():
            opts.update(p.split("|"))
        return sorted(opts)
    return sorted(df[fallback_col].dropna().unique())

def _any_in(series, selected):
    """Boolean mask: True when any item in a pipe-separated cell is in `selected`."""
    sel_set = set(selected)
    return series.apply(
        lambda p: bool(sel_set.intersection(
            p.split("|") if isinstance(p, str) else [p]
        ))
    )


# Season — mandatory; hide future/incomplete seasons and non-top5 competitions
_EXCLUDED_SEASONS = {"2025/2026", "2025/2026 ", "2025", "2025 "}
_EXCLUDED_COMPS   = {"Champions League", "Europa League", "UEFA Super Cup",
                     "UEFA Women's EURO", "World Cup Qualification UEFA"}

all_seasons = [
    s for s in sorted(stats["season"].dropna().unique(), reverse=True)
    if str(s).strip() not in _EXCLUDED_SEASONS
]
sel_season  = st.sidebar.selectbox("Season", all_seasons)
stats_season = stats[
    (stats["season"] == sel_season) &
    (~stats["competition"].isin(_EXCLUDED_COMPS))
].copy()

st.sidebar.markdown("---")

# Position Group
grp_list_col = ("grouped_position_list"
                if "grouped_position_list" in stats_season.columns
                else "grouped_position")
grp_col = "grouped_position" if "grouped_position" in stats_season.columns else "position"

with st.sidebar.expander("Position Group", expanded=True):
    all_groups = _pos_options(stats_season, grp_list_col, grp_col)
    sel_groups = st.multiselect("Position Group", all_groups, default=all_groups, label_visibility="collapsed")

if sel_groups:
    stats_season = stats_season[_any_in(stats_season[grp_list_col], sel_groups)]

# Specific position — limited to positions that belong to the selected groups
pos_list_col = ("position_full_list"
                if "position_full_list" in stats_season.columns
                else "position_full")
pos_col = "position_full" if "position_full" in stats_season.columns else "position"

with st.sidebar.expander("Position", expanded=True):
    all_positions_pool = _pos_options(stats_season, pos_list_col, pos_col)
    if sel_groups:
        allowed = set()
        for g in sel_groups:
            allowed |= GROUP_TO_FULL_POSITIONS.get(g, set())
        all_positions = [p for p in all_positions_pool if p in allowed]
    else:
        all_positions = all_positions_pool
    # Key includes sel_groups so the widget resets when groups change,
    # avoiding a crash from stale session-state values not in the new options.
    pos_key = "pos_ms_" + "_".join(sorted(sel_groups))
    sel_positions = st.multiselect("Position", all_positions, default=all_positions, key=pos_key, label_visibility="collapsed")

# Competition
with st.sidebar.expander("League / Competition", expanded=True):
    all_comps = sorted(stats_season["competition"].dropna().unique())
    sel_comps = st.multiselect("League / Competition", all_comps, default=all_comps, label_visibility="collapsed")

# Min events
with st.sidebar.expander("Min. xT Actions", expanded=False):
    min_events = st.slider("Min. xT Actions", min_value=0, max_value=500, value=100, step=10, label_visibility="collapsed")

# Build filtered set
filtered = stats_season.copy()
if sel_positions:
    filtered = filtered[_any_in(filtered[pos_list_col], sel_positions)]
if sel_comps:
    filtered = filtered[filtered["competition"].isin(sel_comps)]
filtered = filtered[
    (filtered["pass_xt_count"] + filtered["dribble_xt_count"]) >= min_events
].reset_index(drop=True)

st.sidebar.markdown(f"**{len(filtered):,}** players match filters")
st.sidebar.markdown("---")

if filtered.empty:
    st.warning("No players match the current filters. Try relaxing the criteria.")
    st.stop()

# ── Player selector ────────────────────────────────────────────────────────────
display_col = "name_clean" if "name_clean" in filtered.columns else "name"

# Players who played in multiple leagues get a "(League)" suffix so each row is selectable
name_league_counts = filtered.groupby(display_col)["competition"].nunique()
filtered["_display_label"] = filtered.apply(
    lambda r: (f"{r[display_col]} ({r['competition']})"
               if name_league_counts.get(r[display_col], 1) > 1
               else r[display_col]),
    axis=1,
)

# Team filter — narrows the player list before final player selection
with st.sidebar.expander("Team", expanded=False):
    all_teams = sorted(filtered["team_name"].dropna().unique())
    sel_teams = st.multiselect("Team", all_teams, default=[], label_visibility="collapsed")

# Apply team filter to narrow player list
if sel_teams:
    filtered_for_player = filtered[filtered["team_name"].isin(sel_teams)]
    if filtered_for_player.empty:
        filtered_for_player = filtered  # fallback if filter empties results
else:
    filtered_for_player = filtered

player_labels = sorted(filtered_for_player["_display_label"].tolist())
sel_label = st.sidebar.selectbox("🔍 Select Player", player_labels)
player_row = filtered_for_player[filtered_for_player["_display_label"] == sel_label].iloc[0]
sel_player_name = player_row[display_col]


# ── Position-level peers (season-scoped) ──────────────────────────────────────
# Use the most-played non-sub position as the ranking anchor.
_primary_pos = (player_row.get("primary_position") or player_row["position"])
_primary_group = (player_row.get("primary_grouped_position")
                  or GROUPED_POSITION_MAP.get(_primary_pos, _primary_pos))

_pos_list_col = ("position_list"
                 if "position_list" in stats_season.columns
                 else "position")
_grp_list_col2 = ("grouped_position_list"
                  if "grouped_position_list" in stats_season.columns
                  else "grouped_position")

def _pos_peers(base: pd.DataFrame, competition: str = None, min_ev: int = min_events):
    """Peers whose MAIN position matches the selected player's primary position."""
    pri_col = ("primary_position"
               if "primary_position" in base.columns
               else _pos_list_col)
    df = base[base[pri_col] == _primary_pos]
    if competition is not None:
        df = df[df["competition"] == competition]
    return df[(df["pass_xt_count"] + df["dribble_xt_count"]) >= min_ev]

def _grp_peers(base: pd.DataFrame, competition: str = None, min_ev: int = min_events):
    """Peers whose MAIN position group matches the selected player's main group.
    Uses primary_grouped_position (single value = group with most games played),
    so a player who mostly plays Right Defender is never mixed into Left Defender peers."""
    grp_col = ("primary_grouped_position"
               if "primary_grouped_position" in base.columns
               else _grp_list_col2)
    df = base[base[grp_col] == _primary_group]
    if competition is not None:
        df = df[df["competition"] == competition]
    return df[(df["pass_xt_count"] + df["dribble_xt_count"]) >= min_ev]

pos_league_peers  = _pos_peers(stats_season, competition=player_row["competition"])
pos_global_peers  = _pos_peers(stats_season)
viz_league_peers  = _grp_peers(stats_season, competition=player_row["competition"])
age = player_row.get("age")
age_peers = (
    stats_season[
        stats_season["age"].between(age - 2, age + 2) &
        ((stats_season["pass_xt_count"] + stats_season["dribble_xt_count"]) >= min_events)
    ]
    if pd.notna(age) else pd.DataFrame()
)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Player",   player_row.get("name_clean") or player_row["name"])
_pos_display = (
    str(player_row.get("position_full_list", "")).replace("|", " / ")
    or str(player_row.get("position_full") or player_row.get("position", "–"))
)
c2.metric("Position(s)", _pos_display)
c3.metric("Team",   str(player_row.get("team_name",   "–")))
c4.metric("League", str(player_row.get("competition", "–")))
c5.metric("Season", str(player_row.get("season",      "–")))
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_readme, tab_team, tab_arb, tab_rank, tab_vis, tab_cmp = st.tabs([
    "📖 Guide",
    "Step 1 · 🏟️ Team Analysis",
    "Step 2 · 🔀 Arbitrage",
    "Step 3 · 📊 Player ID",
    "Step 4 · 🗺️ Playstyle",
    "Step 5 · ⚖️ Comparison",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 0 — Guide / README
# ─────────────────────────────────────────────────────────────────────────────
with tab_readme:
    st.markdown(f"# Aidea — Scouting Intelligence Platform")
    st.markdown(
        f'<p style="color:{DIM};font-size:1rem;margin-top:-10px">'
        f"An xT-based player and team evaluation tool for the top 5 European leagues.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── What is xT? ──────────────────────────────────────────────────────────
    st.markdown("## What is Expected Threat (xT)?")
    st.markdown(f"""
Expected Threat (**xT**) is a pitch-control model that assigns a **probability of scoring** to
every location on the pitch. When a player moves the ball from one zone to another — via a pass
or a dribble — the difference in xT between the destination and the origin is the **xT added**
by that action.

> A pass from deep midfield to the edge of the box might move the ball from a zone worth **0.03 xT**
> to one worth **0.12 xT**, generating **+0.09 xT** for that action.

Unlike xG (which only counts shots), xT captures **all ball-progressing actions**, making it ideal
for evaluating **midfielders, wingers, and fullbacks** who build attacks without necessarily shooting.

**Key properties:**
- Values are derived from historical scoring probabilities, zone by zone (32 × 24 grid)
- Normalised **per 90 minutes** so players with different workloads are comparable
- Split into **Pass xT** and **Dribble xT** to distinguish passing range from carry ability
""")

    # ── Pitch zone plots ──────────────────────────────────────────────────────
    st.markdown("### xT values across the pitch")
    st.caption(
        "Each cell shows the probability of scoring from that zone. "
        "Passes and dribbles that move the ball toward high-value zones generate more xT."
    )

    @st.cache_data(ttl=86400)
    def _xt_grid_plots():
        from mplsoccer import Pitch
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        grid = load_xt_grid()   # shape (24, 32)
        pitch = Pitch(pitch_type="opta", pitch_color="#06000f", line_color="#3a1a5c")

        CMAP_XT = plt.colormaps["YlOrRd"]

        figs = []
        for title in ("Passes — xT Grid", "Dribbles — xT Grid"):
            fig, ax = pitch.draw(figsize=(7, 4.5))
            fig.patch.set_facecolor("#06000f")

            bin_w = 100 / 32
            bin_h = 100 / 24
            for row in range(24):
                for col in range(32):
                    val = grid[row, col]
                    x   = col * bin_w
                    y   = row * bin_h
                    color = CMAP_XT(val / (grid.max() + 1e-9))
                    rect = plt.Rectangle((x, y), bin_w, bin_h,
                                         color=color, alpha=0.85, zorder=2)
                    ax.add_patch(rect)
                    if val >= 0.05:
                        ax.text(x + bin_w / 2, y + bin_h / 2, f"{val:.2f}",
                                ha="center", va="center", fontsize=5,
                                color="white", fontweight="bold", zorder=3)

            sm = plt.cm.ScalarMappable(cmap=CMAP_XT,
                                        norm=mcolors.Normalize(0, grid.max()))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                                fraction=0.025, pad=0.02)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
            cbar.set_label("xT value", color="white", fontsize=8)

            ax.set_title(title, color="#00e5b0", fontsize=13,
                         fontweight="bold", pad=10)
            figs.append(fig)
        return figs

    _figs = _xt_grid_plots()
    _gc1, _gc2 = st.columns(2)
    with _gc1:
        st.pyplot(_figs[0]); plt.close(_figs[0])
    with _gc2:
        st.pyplot(_figs[1]); plt.close(_figs[1])

    st.markdown("---")

    # ── Scouting workflow ─────────────────────────────────────────────────────
    st.markdown("## Scouting Workflow")
    st.markdown("Aidea is designed as a **5-step scouting pipeline**. Each tab corresponds to one step.")

    steps = [
        ("Step 1 · 🏟️ Team Analysis",
         "Identify weak spots in a team's squad. Compare pass and dribble xT/90 against "
         "league and global benchmarks. The **xT Efficiency** section reveals which teams "
         "generate the most threat per euro of market value — a proxy for squad efficiency."),
        ("Step 2 · 🔀 Arbitrage",
         "Once a weak spot is identified, find replacement options for the selected player. "
         "Three categories are suggested: the **Ideal** (best performer globally), the "
         "**Perfect Arbitrage** (best xT at the same or lower price), and the "
         "**Money Generator** (comparable output, lowest cost)."),
        ("Step 3 · 📊 Player Identification",
         "Deep-dive into a specific candidate. Compare their Pass xT/90 and Dribble xT/90 "
         "against position peers within the same league, globally, and within their age bracket. "
         "The leaderboard table shows where they rank among the top 10 in their league."),
        ("Step 4 · 🗺️ Playstyle Understanding",
         "Visualise *where* a player operates and *how* they move the ball. Heatmaps of pass "
         "start/end and dribble start/end positions reveal their spatial tendencies. "
         "The Top 5 Most Common Plays shows their recurring patterns as pitch arrows. "
         "Toggle the comparison panel to see how their zones differ from the positional average."),
        ("Step 5 · ⚖️ Player Comparison",
         "Side-by-side analysis of two players. Select any player from any position group and "
         "league as the comparison target. The stats table shows key per-90 metrics and global "
         "percentiles. Heatmap trios (Player 1 | Player 2 | Difference) allow visual "
         "identification of complementary or overlapping profiles."),
    ]

    for title, desc in steps:
        st.markdown(
            f'<div style="background:{BG_CARD};border-left:3px solid {MINT};'
            f'border-radius:6px;padding:14px 18px;margin-bottom:12px">'
            f'<div style="color:{MINT};font-weight:700;font-size:0.95rem;margin-bottom:6px">'
            f'{title}</div>'
            f'<div style="color:#ccc;font-size:0.88rem;line-height:1.6">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB — Team Arbitrage
# ─────────────────────────────────────────────────────────────────────────────
with tab_arb:
    pname_arb = player_row.get("name_clean") or player_row["name"]
    st.subheader(f"Team Arbitrage — {pname_arb}")

    has_mv_col = "market_value_eur_m" in stats_season.columns

    # ── Player profile card ───────────────────────────────────────────────────
    arb_mv = player_row.get("market_value_eur_m")
    arb_mv_str = f"€{arb_mv:.1f}M" if pd.notna(arb_mv) and arb_mv is not None else "N/A"
    arb_nat = player_row.get("nationality", "–") or "–"
    arb_age = int(player_row["age"]) if pd.notna(player_row.get("age")) else "–"

    # Percentile of this player's total_xt_per90 vs position-group peers in same league
    _arb_grp_peers = _grp_peers(stats_season, competition=player_row["competition"])
    _arb_pct = pct_rank(player_row["total_xt_per90"], _arb_grp_peers["total_xt_per90"])

    st.markdown("#### Selected Player")
    p1a, p1b, p1c, p1d, p1e, p1f, p1g = st.columns(7)
    p1a.metric("League",        player_row.get("competition", "–"))
    p1b.metric("Team",          player_row.get("team_name",   "–"))
    p1c.metric("Position Grp",  _primary_group)
    p1d.metric("xT/90",         f"{player_row['total_xt_per90']:.3f}")
    p1e.metric("xT/90 %ile",    f"Top {100 - _arb_pct:.0f}%")
    p1f.metric("Age",           arb_age)
    p1g.metric("Market Value",  arb_mv_str)
    if arb_nat and arb_nat != "–":
        st.markdown(
            f'<span style="background:#0d2e1a;color:#00e5b0;border:1px solid #00e5b0;'
            f'border-radius:4px;padding:2px 10px;font-size:13px;">🌍 {arb_nat}</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Candidate pool: same position group, same season, min events ──────────
    _arb_pool = _grp_peers(stats_season).copy()
    # Exclude the selected player
    _arb_pool = _arb_pool[_arb_pool["player_id"] != player_row["player_id"]].copy()

    if _arb_pool.empty:
        st.info("Not enough position-group peers to generate suggestions.")
    else:
        # Pre-compute percentile for all candidates
        _pool_xt_series = _arb_pool["total_xt_per90"]

        def _cand_pct(val):
            return pct_rank(val, _pool_xt_series)

        _arb_pool["_xt_pct"] = _arb_pool["total_xt_per90"].apply(_cand_pct)
        _name_col = "name_clean" if "name_clean" in _arb_pool.columns else "name"

        def _arb_card(row, rank, color):
            mv_val = row.get("market_value_eur_m")
            mv_s   = f"€{mv_val:.1f}M" if pd.notna(mv_val) and mv_val is not None else "N/A"
            nat    = row.get("nationality", "–") or "–"
            age_v  = int(row["age"]) if pd.notna(row.get("age")) else "–"
            top_v  = 100 - row["_xt_pct"]
            team_s = row.get("team_name", "–")
            comp_s = row.get("competition", "–")

            st.markdown(
                f'<div style="background:{BG_CARD};border:1px solid {color};border-radius:10px;'
                f'padding:14px 18px;margin-bottom:6px">'
                f'<div style="color:{color};font-weight:700;font-size:0.75rem;'
                f'letter-spacing:0.08em;margin-bottom:6px">#{rank}</div>'
                f'<div style="color:{WHITE};font-size:1.05rem;font-weight:700;margin-bottom:2px">'
                f'{row[_name_col]}</div>'
                f'<div style="color:{DIM};font-size:0.82rem">'
                f'{team_s} · {comp_s}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            ca, cb, cc, cd, ce = st.columns(5)
            ca.metric("Market Value", mv_s)
            cb.metric("Age",          age_v)
            cc.metric("Nationality",  nat)
            cd.metric("xT/90",        f"{row['total_xt_per90']:.3f}")
            ce.metric("xT/90 %ile",   f"Top {top_v:.0f}%")

        def _render_candidates(df, color, empty_msg):
            if df.empty:
                st.info(empty_msg)
            else:
                for rank, (_, row) in enumerate(df.iterrows(), 1):
                    _arb_card(row, rank, color)

        # ── Option 1: Ideal — best xT/90 in the world ──────────────────────
        st.markdown("### 🌟 Ideal Option — Best performers in the world")
        st.caption("Top 3 players by xT/90 in this position group globally.")
        _render_candidates(
            _arb_pool.nlargest(3, "total_xt_per90"),
            MINT, "No candidates found."
        )

        st.markdown("---")

        # ── Option 2: Perfect Arbitrage — best xT at same or lower market value ──
        st.markdown("### ⚖️ Perfect Arbitrage — Best xT at same or lower market value")
        st.caption(
            f"Top 3 players by xT/90 whose market value ≤ {arb_mv_str}."
            if pd.notna(arb_mv) else
            "Market value for selected player is unknown — showing best available with known value."
        )
        if has_mv_col and pd.notna(arb_mv):
            _arb_candidates = _arb_pool[
                _arb_pool["market_value_eur_m"].notna() &
                (_arb_pool["market_value_eur_m"] <= arb_mv)
            ]
        elif has_mv_col:
            _arb_candidates = _arb_pool[_arb_pool["market_value_eur_m"].notna()]
        else:
            _arb_candidates = pd.DataFrame()

        _render_candidates(
            _arb_candidates.nlargest(3, "total_xt_per90") if not _arb_candidates.empty else pd.DataFrame(),
            AMBER, "No candidates found with known market value ≤ selected player's value."
        )

        st.markdown("---")

        # ── Option 3: Money Generator — same xT, cheapest price ───────────
        st.markdown("### 💰 Money Generator — Same xT output, lowest market value")
        player_xt = player_row["total_xt_per90"]
        _xt_tol = max(player_xt * 0.15, 0.005)
        _tol_note = "±15%"
        if has_mv_col:
            _money_candidates = _arb_pool[
                _arb_pool["market_value_eur_m"].notna() &
                (_arb_pool["total_xt_per90"] >= player_xt - _xt_tol) &
                (_arb_pool["total_xt_per90"] <= player_xt + _xt_tol)
            ]
            if len(_money_candidates) < 3:
                _xt_tol = player_xt * 0.30
                _tol_note = "±30%"
                _money_candidates = _arb_pool[
                    _arb_pool["market_value_eur_m"].notna() &
                    (_arb_pool["total_xt_per90"] >= player_xt - _xt_tol) &
                    (_arb_pool["total_xt_per90"] <= player_xt + _xt_tol)
                ]
        else:
            _money_candidates = pd.DataFrame()
        st.caption(
            f"Top 3 cheapest players producing similar xT/90 "
            f"({player_xt:.3f} {_tol_note}: "
            f"{player_xt - _xt_tol:.3f}–{player_xt + _xt_tol:.3f})."
        )
        _render_candidates(
            _money_candidates.nsmallest(3, "market_value_eur_m") if not _money_candidates.empty else pd.DataFrame(),
            GREEN, "No candidates found with known market value and similar xT output."
        )

        if not has_mv_col:
            st.warning("Market value data not available. Run `merge_market_values.py` to enable arbitrage options 2 and 3.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — xT Rankings
# ─────────────────────────────────────────────────────────────────────────────
with tab_rank:
    pname = player_row.get("name_clean") or player_row["name"]
    st.subheader(f"xT Profile — {pname}")

    # Multi-league notice
    player_leagues = filtered[filtered[display_col] == sel_player_name]["competition"].tolist()
    if len(player_leagues) > 1:
        other = [l for l in player_leagues if l != player_row["competition"]]
        st.info(
            f"This player has data in **{len(player_leagues)} competitions** this season. "
            f"Showing: **{player_row['competition']}**. "
            f"Other: {', '.join(other)}. Use the player selector to switch leagues."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pass xT / 90",    f"{player_row['pass_xt_per90']:.2f}")
    m2.metric("Dribble xT / 90", f"{player_row['dribble_xt_per90']:.2f}")
    m3.metric("Total xT / 90",   f"{player_row['total_xt_per90']:.2f}")
    m4.metric("Matches (est.)",   int(player_row["matches_played"]))

    st.markdown("---")

    pos_label = (
        str(player_row.get("primary_position_full") or "")
        or str(player_row.get("position_full") or player_row.get("position", ""))
    )
    age_label = f"Age {int(age)}±2 · All Positions" if pd.notna(age) else "Age"

    # ── Pass xT rankings ──────────────────────────────────────────────────────
    st.subheader("Pass xT / 90")
    _pdict = player_row.to_dict()
    rc1, rc2, rc3 = st.columns(3)
    for col, peers, ctx in [
        (rc1, pos_league_peers, f"{pos_label}s · {player_row['competition']}"),
        (rc2, pos_global_peers, f"{pos_label}s · All Leagues"),
        (rc3, age_peers,        age_label),
    ]:
        with col:
            if peers.empty:
                st.info("Not enough peers.")
            else:
                pct = pct_rank(player_row["pass_xt_per90"], peers["pass_xt_per90"])
                st.markdown(f"**{ctx}**", unsafe_allow_html=False)
                st.markdown(pct_label(pct), unsafe_allow_html=True)
                st.pyplot(_fig_rank_bar(_pdict, peers, "pass_xt_per90", "Pass xT / 90"))
                plt.close("all")

    st.markdown("---")

    # ── Dribble xT rankings ───────────────────────────────────────────────────
    st.subheader("Dribble xT / 90")
    st.caption("Dribble xT = threat level of zones where the player successfully dribbles.")
    rd1, rd2, rd3 = st.columns(3)
    for col, peers, ctx in [
        (rd1, pos_league_peers, f"{pos_label}s · {player_row['competition']}"),
        (rd2, pos_global_peers, f"{pos_label}s · All Leagues"),
        (rd3, age_peers,        age_label),
    ]:
        with col:
            if peers.empty:
                st.info("Not enough peers.")
            else:
                pct = pct_rank(player_row["dribble_xt_per90"], peers["dribble_xt_per90"])
                st.markdown(f"**{ctx}**", unsafe_allow_html=False)
                st.markdown(pct_label(pct), unsafe_allow_html=True)
                st.pyplot(_fig_rank_bar(_pdict, peers, "dribble_xt_per90", "Dribble xT / 90"))
                plt.close("all")

    st.markdown("---")

    # ── Top 10 leaderboard ────────────────────────────────────────────────────
    st.subheader(
        f"Top 10 {pos_label}s — {player_row['competition']}"
        + (f" · {sel_season}" if sel_season != "All" else "")
    )
    name_col = "name_clean" if "name_clean" in pos_league_peers.columns else "name"
    top10 = pos_league_peers.nlargest(10, "total_xt_per90")[
        [name_col, "team_name", "age", "pass_xt_per90",
         "dribble_xt_per90", "total_xt_per90", "matches_played"]
    ].rename(columns={
        name_col:           "Player",
        "team_name":        "Team",
        "age":              "Age",
        "pass_xt_per90":    "Pass xT/90",
        "dribble_xt_per90": "Dribble xT/90",
        "total_xt_per90":   "Total xT/90",
        "matches_played":   "Matches",
    })
    top10.insert(0, "#", range(1, len(top10) + 1))
    top10 = top10.reset_index(drop=True)

    def _highlight(row):
        style = f"background-color: #1a0535; color: {MINT}; font-weight: bold"
        return [style if row["Player"] == pname else "" for _ in row]

    st.dataframe(
        top10.style.apply(_highlight, axis=1).format({
            "Pass xT/90":    "{:.2f}",
            "Dribble xT/90": "{:.2f}",
            "Total xT/90":   "{:.2f}",
            "Age":           "{:.0f}",
        }),
        use_container_width=True,
        column_config={
            "#":            st.column_config.NumberColumn("#",            width="small"),
            "Matches":      st.column_config.NumberColumn("Matches",     width="small"),
            "Pass xT/90":   st.column_config.NumberColumn("Pass xT/90",  width="small"),
            "Dribble xT/90":st.column_config.NumberColumn("Drib xT/90",  width="small"),
            "Total xT/90":  st.column_config.NumberColumn("Total xT/90", width="small"),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — xT Visualizer
# ─────────────────────────────────────────────────────────────────────────────
with tab_vis:
    pname_vis = player_row.get("name_clean") or player_row["name"]
    st.subheader(f"xT Visualizer — {pname_vis}")

    if all_events.empty:
        st.warning("No event data found. Run `python Streamlit/data_prep.py` first.")
    else:
        _pid_vis = int(player_row["player_id"])
        _has_ev  = not _player_ev(_pid_vis).empty

        if not _has_ev:
            st.warning("No events found for this player in the events parquet.")
        else:
            _tog_col, _pos_col = st.columns([2, 3])
            with _tog_col:
                show_vs_avg = st.toggle("Compare vs Position Average", value=False)
            with _pos_col:
                _main_full   = player_row.get("primary_position") or player_row.get("position", "")
                _pos_full_raw = player_row.get("position_full_list") or player_row.get("position_full", "")
                _all_pos = [_main_full] if _main_full else []
                for _p in str(_pos_full_raw).split("|"):
                    _p = _p.strip()
                    if _p and _p not in _all_pos:
                        _all_pos.append(_p)
                _tags_html = (
                    '<span style="color:#888;font-size:12px;margin-right:6px;">Positions:</span>'
                    + " ".join(
                        f'<span style="background:#0d2e1a;color:#00e5b0;border:1px solid #00e5b0;'
                        f'border-radius:4px;padding:2px 8px;font-size:12px;white-space:nowrap;">'
                        f'{"★ " if i == 0 else ""}{p}</span>'
                        for i, p in enumerate(_all_pos)
                    )
                )
                st.markdown(_tags_html, unsafe_allow_html=True)
            st.markdown("---")

            _peer_ids_vis  = tuple(sorted(viz_league_peers["player_id"].tolist()))
            _pos_avg_xt    = _peer_avg_xt(_peer_ids_vis) if _peer_ids_vis else None

            # ── Passes row ────────────────────────────────────────────────────
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown(f"**{pname_vis} — Pass Start xT / 90**")
                if show_vs_avg and _peer_ids_vis:
                    fig = _fig_vs_avg(_pid_vis, _peer_ids_vis, "pass", False,
                                      f"{pname_vis} · Pass xT/90 — Start Position", pname_vis)
                else:
                    fig = _fig_heatmap(_pid_vis, "pass", False,
                                       f"{pname_vis} · Pass xT/90 — Start Position")
                st.pyplot(fig); plt.close("all")

            with pc2:
                st.markdown(f"**{pname_vis} — Pass End xT / 90**")
                if show_vs_avg and _peer_ids_vis:
                    fig = _fig_vs_avg(_pid_vis, _peer_ids_vis, "pass", True,
                                      f"{pname_vis} · Pass xT/90 — End Position", pname_vis)
                else:
                    fig = _fig_heatmap(_pid_vis, "pass", True,
                                       f"{pname_vis} · Pass xT/90 — End Position")
                st.pyplot(fig); plt.close("all")

            st.markdown("---")

            # ── Dribbles row ──────────────────────────────────────────────────
            st.caption("End position estimated from next action in the same match.")
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"**{pname_vis} — Dribble Start xT / 90**")
                if show_vs_avg and _peer_ids_vis:
                    fig = _fig_vs_avg(_pid_vis, _peer_ids_vis, "dribble", False,
                                      f"{pname_vis} · Dribble xT/90 — Start Position", pname_vis)
                else:
                    fig = _fig_heatmap(_pid_vis, "dribble", False,
                                       f"{pname_vis} · Dribble xT/90 — Start Position")
                st.pyplot(fig); plt.close("all")

            with dc2:
                st.markdown(f"**{pname_vis} — Dribble End xT / 90**")
                if show_vs_avg and _peer_ids_vis:
                    fig = _fig_vs_avg(_pid_vis, _peer_ids_vis, "dribble", True,
                                      f"{pname_vis} · Dribble xT/90 — End Position", pname_vis)
                else:
                    fig = _fig_heatmap(_pid_vis, "dribble", True,
                                       f"{pname_vis} · Dribble xT/90 — End Position")
                st.pyplot(fig); plt.close("all")

            st.markdown("---")

            # ── Top plays ─────────────────────────────────────────────────────
            st.markdown("#### Top 5 Most Common xT Plays")
            st.caption("Grouped into 4×4 bin zones · width ∝ frequency · "
                       "green = above avg, red = below avg")
            fig = _fig_top_plays(_pid_vis, f"{pname_vis} — Top Plays", _pos_avg_xt)
            st.pyplot(fig); plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Player Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    st.subheader("Player Comparison")

    cmp_name_col  = "name_clean" if "name_clean" in stats_season.columns else "name"
    grp_col_name  = ("primary_grouped_position"
                     if "primary_grouped_position" in stats_season.columns
                     else _grp_list_col2)

    p1_row  = player_row
    p1_name = sel_player_name
    p1_grp  = _primary_group
    p1_lgue = str(player_row.get("competition", ""))

    # ── Selectors ──────────────────────────────────────────────────────────────
    sel_col, cmp_col = st.columns(2)

    with sel_col:
        st.markdown("#### Selected Player")
        st.selectbox("Position Group", [p1_grp],  disabled=True, key="p1_grp_lock")
        st.selectbox("League",         [p1_lgue], disabled=True, key="p1_lgue_lock")
        st.markdown(f"**{p1_name}**")

    with cmp_col:
        st.markdown("#### Comparison Player")

        # Position group — default to selected player's group
        all_cmp_grps = sorted(
            g for g in stats_season[grp_col_name].dropna().unique()
            if g != "Substitute"
        )
        default_grp_idx = all_cmp_grps.index(p1_grp) if p1_grp in all_cmp_grps else 0
        cmp_grp = st.selectbox("Position Group", all_cmp_grps,
                               index=default_grp_idx, key="p2_grp")

        # League — default to selected player's league
        all_cmp_lgues   = sorted(stats_season["competition"].dropna().unique())
        default_lgue_idx = (all_cmp_lgues.index(p1_lgue)
                            if p1_lgue in all_cmp_lgues else 0)
        cmp_lgue = st.selectbox("League", all_cmp_lgues,
                                index=default_lgue_idx, key="p2_lgue")

        # Player pool filtered by group + league (using main position group)
        cmp_pool = stats_season[
            (stats_season[grp_col_name] == cmp_grp) &
            (stats_season["competition"] == cmp_lgue) &
            ((stats_season["pass_xt_count"] + stats_season["dribble_xt_count"]) >= min_events)
        ].copy()

        if cmp_pool.empty:
            st.warning("No players found for these filters.")
            p2_row, p2_name = None, None
        else:
            cmp_player_names = sorted(cmp_pool[cmp_name_col].dropna().tolist())
            p2_name = st.selectbox("Player", cmp_player_names, key="cmp_p2_name")
            p2_row  = cmp_pool[cmp_pool[cmp_name_col] == p2_name].iloc[0]

    if p2_row is not None:
        st.markdown("---")

        # ── Stats + percentile table ───────────────────────────────────────────
        # Percentiles relative to selected player's global peers (same primary position)
        stat_metrics = [
            ("Pass xT/90",    "pass_xt_per90"),
            ("Dribble xT/90", "dribble_xt_per90"),
            ("Total xT/90",   "total_xt_per90"),
        ]
        cmp_rows = []
        for metric_label, col in stat_metrics:
            p1_val = p1_row[col]
            p2_val = p2_row[col]
            p1_pct = pct_rank(p1_val, pos_global_peers[col]) if not pos_global_peers.empty else None
            p2_pct = pct_rank(p2_val, pos_global_peers[col]) if not pos_global_peers.empty else None
            cmp_rows.append({
                "Metric":          metric_label,
                p1_name:           f"{p1_val:.2f}",
                f"{p1_name} %ile": f"Top {100 - p1_pct:.0f}%" if p1_pct is not None else "–",
                p2_name:           f"{p2_val:.2f}",
                f"{p2_name} %ile": f"Top {100 - p2_pct:.0f}%" if p2_pct is not None else "–",
            })

        # Matches
        cmp_rows.append({
            "Metric":          "Matches",
            p1_name:           str(int(p1_row["matches_played"])),
            f"{p1_name} %ile": "–",
            p2_name:           str(int(p2_row["matches_played"])),
            f"{p2_name} %ile": "–",
        })

        # Age
        p1_age = int(p1_row["age"]) if pd.notna(p1_row.get("age")) else "–"
        p2_age = int(p2_row["age"]) if pd.notna(p2_row.get("age")) else "–"
        cmp_rows.append({
            "Metric":          "Age",
            p1_name:           str(p1_age),
            f"{p1_name} %ile": "–",
            p2_name:           str(p2_age),
            f"{p2_name} %ile": "–",
        })

        # Market Value
        def _fmt_mv(row):
            mv = row.get("market_value_eur_m")
            return f"€{mv:.1f}M" if pd.notna(mv) and mv is not None else "N/A"

        cmp_rows.append({
            "Metric":          "Market Value",
            p1_name:           _fmt_mv(p1_row),
            f"{p1_name} %ile": "–",
            p2_name:           _fmt_mv(p2_row),
            f"{p2_name} %ile": "–",
        })

        st.dataframe(pd.DataFrame(cmp_rows).set_index("Metric"), use_container_width=True)
        st.markdown("---")

        # ── Heatmap sections ───────────────────────────────────────────────────
        if all_events.empty:
            st.warning("Event data missing. Run data_prep.py.")
        else:
            _p1_id = int(p1_row["player_id"])
            _p2_id = int(p2_row["player_id"])
            for section_title, event_type, use_end in [
                ("Pass Start xT / 90",    "pass",    False),
                ("Pass End xT / 90",      "pass",    True),
                ("Dribble Start xT / 90", "dribble", False),
                ("Dribble End xT / 90",   "dribble", True),
            ]:
                st.markdown(f"#### {section_title}")
                fig = _fig_comparison(
                    _p1_id, _p2_id, p1_name, p2_name,
                    event_type, use_end,
                    f"{section_title}  ·  {p1_name} vs {p2_name}",
                )
                st.pyplot(fig)
                plt.close("all")
                st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Team Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_team:
    st.subheader("Team Analysis")

    # ── Selectors ──────────────────────────────────────────────────────────────
    tl_col, tt_col = st.columns(2)
    with tl_col:
        team_leagues    = sorted(stats_season["competition"].dropna().unique())
        sel_team_league = st.selectbox("League", team_leagues, key="team_league")
    with tt_col:
        league_pool  = stats_season[stats_season["competition"] == sel_team_league]
        team_options = sorted(league_pool["team_name"].dropna().unique())
        sel_team_name = st.selectbox("Team", team_options, key="team_select")

    st.markdown("---")

    # ── Aggregate per team (mean xT/90 across qualifying players) ─────────────
    def _team_agg(df, metric):
        return (
            df[(df["pass_xt_count"] + df["dribble_xt_count"]) >= min_events]
            .groupby("team_name")[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: "value"})
        )

    lge_pass = _team_agg(league_pool,  "pass_xt_per90")
    lge_drib = _team_agg(league_pool,  "dribble_xt_per90")
    glb_pass = _team_agg(stats_season, "pass_xt_per90")
    glb_drib = _team_agg(stats_season, "dribble_xt_per90")

    def _team_val(agg_df, team):
        row = agg_df[agg_df["team_name"] == team]
        return float(row["value"].iloc[0]) if not row.empty else None

    tv_pass = _team_val(lge_pass, sel_team_name)
    tv_drib = _team_val(lge_drib, sel_team_name)

    # Ensure selected team appears in global chart (top 20 + selected team)
    def _top_with_team(agg_df, team, n=20):
        top = agg_df.nlargest(n, "value")
        if team not in top["team_name"].values:
            team_row = agg_df[agg_df["team_name"] == team]
            top = pd.concat([top, team_row])
        return top.copy()

    glb_pass_chart = _top_with_team(glb_pass, sel_team_name)
    glb_drib_chart = _top_with_team(glb_drib, sel_team_name)

    # ── Section 1: Same league ─────────────────────────────────────────────────
    st.subheader(f"League Comparison — {sel_team_league}")
    lc1, lc2 = st.columns(2)
    with lc1:
        lge_pct_pass = pct_rank(tv_pass, lge_pass["value"]) if tv_pass is not None else None
        if lge_pct_pass is not None:
            st.markdown("**Pass xT/90**")
            st.markdown(pct_label(lge_pct_pass), unsafe_allow_html=True)
        fig = _fig_team_rank_bar(
            lge_pass.rename(columns={"value": "pass_xt_per90"}),
            sel_team_name, "pass_xt_per90", "Avg Pass xT/90",
        )
        st.pyplot(fig); plt.close("all")

    with lc2:
        lge_pct_drib = pct_rank(tv_drib, lge_drib["value"]) if tv_drib is not None else None
        if lge_pct_drib is not None:
            st.markdown("**Dribble xT/90**")
            st.markdown(pct_label(lge_pct_drib), unsafe_allow_html=True)
        fig = _fig_team_rank_bar(
            lge_drib.rename(columns={"value": "dribble_xt_per90"}),
            sel_team_name, "dribble_xt_per90", "Avg Dribble xT/90",
        )
        st.pyplot(fig); plt.close("all")

    st.markdown("---")

    # ── Section 2: Global comparison ──────────────────────────────────────────
    st.subheader("Global Comparison — Top 20 teams across all leagues")
    gc1, gc2 = st.columns(2)
    with gc1:
        gp_pass = pct_rank(tv_pass, glb_pass["value"]) if tv_pass is not None else None
        if gp_pass is not None:
            st.markdown("**Pass xT/90**")
            st.markdown(pct_label(gp_pass), unsafe_allow_html=True)
        fig = _fig_team_rank_bar(
            glb_pass_chart.rename(columns={"value": "pass_xt_per90"}),
            sel_team_name, "pass_xt_per90", "Avg Pass xT/90",
        )
        st.pyplot(fig); plt.close("all")

    with gc2:
        gp_drib = pct_rank(tv_drib, glb_drib["value"]) if tv_drib is not None else None
        if gp_drib is not None:
            st.markdown("**Dribble xT/90**")
            st.markdown(pct_label(gp_drib), unsafe_allow_html=True)
        fig = _fig_team_rank_bar(
            glb_drib_chart.rename(columns={"value": "dribble_xt_per90"}),
            sel_team_name, "dribble_xt_per90", "Avg Dribble xT/90",
        )
        st.pyplot(fig); plt.close("all")

    st.markdown("---")

    # ── Section 3: xT per Market Value ────────────────────────────────────────
    st.subheader("xT Efficiency — Total xT per €M of Market Value")
    st.caption("Mean total_xt_per90 ÷ mean market_value across qualifying players with a known market value.")

    def _team_xt_per_mv(df):
        """Per-team: mean(total_xt_per90) / mean(market_value_eur_m) for players with known MV."""
        has_mv = df[
            (df["pass_xt_count"] + df["dribble_xt_count"] >= min_events) &
            df["market_value_eur_m"].notna() &
            (df["market_value_eur_m"] > 0)
        ]
        if has_mv.empty:
            return pd.DataFrame(columns=["team_name", "value"])
        agg = has_mv.groupby("team_name").apply(
            lambda g: g["total_xt_per90"].mean() / g["market_value_eur_m"].mean()
        ).reset_index()
        agg.columns = ["team_name", "value"]
        return agg.dropna(subset=["value"])

    if "market_value_eur_m" not in stats_season.columns:
        st.info("Market value data not found in parquet. Run `merge_market_values.py` first.")
    else:
        lge_xtmv = _team_xt_per_mv(league_pool)
        glb_xtmv = _team_xt_per_mv(stats_season)

        tv_xtmv = _team_val(lge_xtmv, sel_team_name) if not lge_xtmv.empty else None
        glb_xtmv_chart = _top_with_team(glb_xtmv, sel_team_name) if not glb_xtmv.empty else pd.DataFrame()

        mv1, mv2 = st.columns(2)
        with mv1:
            st.markdown(f"**League — {sel_team_league}**")
            if tv_xtmv is not None and not lge_xtmv.empty:
                lge_pct_mv = pct_rank(tv_xtmv, lge_xtmv["value"])
                st.markdown(pct_label(lge_pct_mv), unsafe_allow_html=True)
                fig = _fig_team_rank_bar(
                    lge_xtmv.rename(columns={"value": "xt_per_mv"}),
                    sel_team_name, "xt_per_mv", "xT/90 per €M"
                )
                st.pyplot(fig); plt.close("all")
            else:
                st.info("No market value data for this league.")

        with mv2:
            st.markdown("**Global — Top 20 teams**")
            if not glb_xtmv_chart.empty and tv_xtmv is not None:
                glb_pct_mv = pct_rank(tv_xtmv, glb_xtmv["value"])
                st.markdown(pct_label(glb_pct_mv), unsafe_allow_html=True)
                fig = _fig_team_rank_bar(
                    glb_xtmv_chart.rename(columns={"value": "xt_per_mv"}),
                    sel_team_name, "xt_per_mv", "xT/90 per €M"
                )
                st.pyplot(fig); plt.close("all")
            else:
                st.info("No market value data available.")

    st.markdown("---")

    # ── Best player per position group ─────────────────────────────────────────
    st.subheader(f"Key Players — {sel_team_name}")
    st.caption("Best player per position group (by Total xT/90). "
               "Percentile ranked vs all players in the same position group in the league.")

    grp_col_t  = ("primary_grouped_position"
                  if "primary_grouped_position" in league_pool.columns
                  else _grp_list_col2)
    name_col_t = "name_clean" if "name_clean" in league_pool.columns else "name"

    team_valid = league_pool[
        (league_pool["team_name"] == sel_team_name) &
        ((league_pool["pass_xt_count"] + league_pool["dribble_xt_count"]) >= min_events)
    ].copy()

    if team_valid.empty:
        st.info("No players meet the minimum action threshold for this team.")
    else:
        best_per_grp = (
            team_valid
            .sort_values("total_xt_per90", ascending=False)
            .drop_duplicates(grp_col_t)
            [[name_col_t, grp_col_t,
              "pass_xt_per90", "dribble_xt_per90", "total_xt_per90"]]
            .copy()
        )

        # Compute league percentile per player vs their position group peers
        def _grp_pct(row):
            peers_vals = league_pool[
                league_pool[grp_col_t] == row[grp_col_t]
            ]["total_xt_per90"]
            pct = pct_rank(row["total_xt_per90"], peers_vals)
            top = 100 - pct
            color = GREEN if top <= 25 else (AMBER if top <= 50 else RED)
            return f'<span style="color:{color};font-weight:700">Top {top:.0f}%</span>'

        best_per_grp["League %ile"] = best_per_grp.apply(_grp_pct, axis=1)

        best_per_grp = best_per_grp.rename(columns={
            name_col_t:          "Player",
            grp_col_t:           "Position Group",
            "pass_xt_per90":     "Pass xT/90",
            "dribble_xt_per90":  "Dribble xT/90",
            "total_xt_per90":    "Total xT/90",
        }).sort_values("Position Group")

        # Render as HTML so the coloured span shows
        html_rows = ""
        for _, r in best_per_grp.iterrows():
            html_rows += (
                f"<tr>"
                f"<td>{r['Position Group']}</td>"
                f"<td>{r['Player']}</td>"
                f"<td>{r['Pass xT/90']:.2f}</td>"
                f"<td>{r['Dribble xT/90']:.2f}</td>"
                f"<td>{r['Total xT/90']:.2f}</td>"
                f"<td>{r['League %ile']}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;color:{WHITE};font-size:0.9rem">
          <thead>
            <tr style="color:{MINT};border-bottom:1px solid #3a1a5c">
              <th style="text-align:left;padding:6px">Position Group</th>
              <th style="text-align:left;padding:6px">Player</th>
              <th style="text-align:right;padding:6px">Pass xT/90</th>
              <th style="text-align:right;padding:6px">Drib xT/90</th>
              <th style="text-align:right;padding:6px">Total xT/90</th>
              <th style="text-align:center;padding:6px">League %ile</th>
            </tr>
          </thead>
          <tbody>{html_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
