"""
app.py — xT Scouting Application
Run:  streamlit run Streamlit/app.py
"""

import sys
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


# Season — mandatory, no "All" option
all_seasons = sorted(stats["season"].dropna().unique(), reverse=True)
sel_season  = st.sidebar.selectbox("Season", all_seasons)
stats_season = stats[stats["season"] == sel_season].copy()

st.sidebar.markdown("---")

# Position Group
grp_list_col = ("grouped_position_list"
                if "grouped_position_list" in stats_season.columns
                else "grouped_position")
grp_col = "grouped_position" if "grouped_position" in stats_season.columns else "position"

with st.sidebar.expander("Position Group", expanded=True):
    all_groups = _pos_options(stats_season, grp_list_col, grp_col)
    sel_groups = st.multiselect("", all_groups, default=all_groups, label_visibility="collapsed")

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
    sel_positions = st.multiselect("", all_positions, default=all_positions, label_visibility="collapsed")

# Competition
with st.sidebar.expander("League / Competition", expanded=True):
    all_comps = sorted(stats_season["competition"].dropna().unique())
    sel_comps = st.multiselect("", all_comps, default=all_comps, label_visibility="collapsed")

# Min events
with st.sidebar.expander("Min. xT Actions", expanded=False):
    min_events = st.slider("", min_value=0, max_value=500, value=20, step=10, label_visibility="collapsed")

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

player_labels = sorted(filtered["_display_label"].tolist())
sel_label = st.sidebar.selectbox("🔍 Select Player", player_labels)
player_row = filtered[filtered["_display_label"] == sel_label].iloc[0]
sel_player_name = player_row[display_col]

# Team filter — narrows the player list after initial selection
all_teams = sorted(filtered["team_name"].dropna().unique())
sel_teams = st.sidebar.multiselect("Team", all_teams, default=[])
if sel_teams:
    # Re-filter and rebuild the player selector if a team is chosen
    filtered_by_team = filtered[filtered["team_name"].isin(sel_teams)]
    if not filtered_by_team.empty:
        player_labels = sorted(filtered_by_team["_display_label"].tolist())
        sel_label = st.sidebar.selectbox("🔍 Select Player (filtered)", player_labels, key="player_team")
        player_row = filtered_by_team[filtered_by_team["_display_label"] == sel_label].iloc[0]
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 xT Rankings", "🗺️ xT Visualizer", "⚖️ Player Comparison", "🏟️ Team Analysis"])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — xT Rankings
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
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
                st.pyplot(plot_rank_bar(player_row, peers, "pass_xt_per90", "Pass xT / 90"))
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
                st.pyplot(plot_rank_bar(player_row, peers, "dribble_xt_per90", "Dribble xT / 90"))
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
with tab2:
    pname_vis = player_row.get("name_clean") or player_row["name"]
    st.subheader(f"xT Visualizer — {pname_vis}")

    if all_events.empty:
        st.warning("No event data found. Run `python Streamlit/data_prep.py` first.")
    else:
        player_events = enrich_events(
            get_player_events(all_events, int(player_row["player_id"])), xt_grid
        )

        if player_events.empty:
            st.warning("No events found for this player in the events parquet.")
        else:
            show_vs_avg = st.toggle("Compare vs Position Average", value=False)
            st.markdown("---")

            # Peer events — use position-GROUP peers (same league) for the visualizer
            viz_peer_ids = viz_league_peers["player_id"].tolist()
            avg_ev = None
            if show_vs_avg and viz_peer_ids:
                avg_ev = enrich_events(
                    all_events[all_events["player_id"].isin(viz_peer_ids)], xt_grid
                )

            position_avg_xt = None
            if viz_peer_ids:
                peer_ev = all_events[all_events["player_id"].isin(viz_peer_ids)]
                if len(peer_ev):
                    position_avg_xt = float(peer_ev["added_xt"].clip(lower=0).mean())

            # ── Passes row ────────────────────────────────────────────────────
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown(f"**{pname_vis} — Pass Start xT / 90**")
                if show_vs_avg and avg_ev is not None:
                    fig = plot_vs_average_pair(
                        player_events, avg_ev, xt_grid,
                        event_type="pass", use_end=False,
                        row_title=f"{pname_vis} · Pass xT/90 — Start Position",
                        player_name=pname_vis)
                else:
                    fig = plot_xt_heatmap(
                        player_events, xt_grid,
                        event_type="pass", use_end=False,
                        title=f"{pname_vis} · Pass xT/90 — Start Position")
                st.pyplot(fig); plt.close("all")

            with pc2:
                st.markdown(f"**{pname_vis} — Pass End xT / 90**")
                if show_vs_avg and avg_ev is not None:
                    fig = plot_vs_average_pair(
                        player_events, avg_ev, xt_grid,
                        event_type="pass", use_end=True,
                        row_title=f"{pname_vis} · Pass xT/90 — End Position",
                        player_name=pname_vis)
                else:
                    fig = plot_xt_heatmap(
                        player_events, xt_grid,
                        event_type="pass", use_end=True,
                        title=f"{pname_vis} · Pass xT/90 — End Position")
                st.pyplot(fig); plt.close("all")

            st.markdown("---")

            # ── Dribbles row ──────────────────────────────────────────────────
            st.caption("End position estimated from next action in the same match.")
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"**{pname_vis} — Dribble Start xT / 90**")
                if show_vs_avg and avg_ev is not None:
                    fig = plot_vs_average_pair(
                        player_events, avg_ev, xt_grid,
                        event_type="dribble", use_end=False,
                        row_title=f"{pname_vis} · Dribble xT/90 — Start Position",
                        player_name=pname_vis)
                else:
                    fig = plot_xt_heatmap(
                        player_events, xt_grid,
                        event_type="dribble", use_end=False,
                        title=f"{pname_vis} · Dribble xT/90 — Start Position")
                st.pyplot(fig); plt.close("all")

            with dc2:
                st.markdown(f"**{pname_vis} — Dribble End xT / 90**")
                if show_vs_avg and avg_ev is not None:
                    fig = plot_vs_average_pair(
                        player_events, avg_ev, xt_grid,
                        event_type="dribble", use_end=True,
                        row_title=f"{pname_vis} · Dribble xT/90 — End Position",
                        player_name=pname_vis)
                else:
                    fig = plot_xt_heatmap(
                        player_events, xt_grid,
                        event_type="dribble", use_end=True,
                        title=f"{pname_vis} · Dribble xT/90 — End Position")
                st.pyplot(fig); plt.close("all")

            st.markdown("---")

            # ── Top plays ─────────────────────────────────────────────────────
            st.markdown("#### Top 5 Most Common xT Plays")
            st.caption("Grouped into 4×4 bin zones · width ∝ frequency · "
                       "green = above avg, red = below avg")
            fig = plot_top_plays(player_events, xt_grid,
                                 position_avg_xt=position_avg_xt,
                                 title=f"{pname_vis} — Top Plays")
            st.pyplot(fig); plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Player Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
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
                "Metric":         metric_label,
                p1_name:          f"{p1_val:.2f}",
                f"{p1_name} %ile": f"Top {100 - p1_pct:.0f}%" if p1_pct is not None else "–",
                p2_name:          f"{p2_val:.2f}",
                f"{p2_name} %ile": f"Top {100 - p2_pct:.0f}%" if p2_pct is not None else "–",
            })
        cmp_rows.append({
            "Metric":          "Matches",
            p1_name:           str(int(p1_row["matches_played"])),
            f"{p1_name} %ile": "–",
            p2_name:           str(int(p2_row["matches_played"])),
            f"{p2_name} %ile": "–",
        })
        st.dataframe(pd.DataFrame(cmp_rows).set_index("Metric"), use_container_width=True)
        st.markdown("---")

        # ── Heatmap sections ───────────────────────────────────────────────────
        if all_events.empty:
            st.warning("Event data missing. Run data_prep.py.")
        else:
            p1_ev = enrich_events(
                get_player_events(all_events, int(p1_row["player_id"])), xt_grid)
            p2_ev = enrich_events(
                get_player_events(all_events, int(p2_row["player_id"])), xt_grid)

            for section_title, event_type, use_end in [
                ("Pass Start xT / 90",    "pass",    False),
                ("Pass End xT / 90",      "pass",    True),
                ("Dribble Start xT / 90", "dribble", False),
                ("Dribble End xT / 90",   "dribble", True),
            ]:
                st.markdown(f"#### {section_title}")
                fig = plot_comparison_trio(
                    p1_ev, p2_ev, p1_name, p2_name,
                    event_type=event_type, use_end=use_end,
                    title=f"{section_title}  ·  {p1_name} vs {p2_name}",
                )
                st.pyplot(fig)
                plt.close("all")
                st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Team Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
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

    # League-level aggregates (for the bar chart)
    lge_pass   = _team_agg(league_pool,  "pass_xt_per90")
    lge_drib   = _team_agg(league_pool,  "dribble_xt_per90")

    # Global aggregates (for cross-league percentile annotation)
    glb_pass   = _team_agg(stats_season, "pass_xt_per90")
    glb_drib   = _team_agg(stats_season, "dribble_xt_per90")

    def _team_val(agg_df, team):
        row = agg_df[agg_df["team_name"] == team]
        return float(row["value"].iloc[0]) if not row.empty else None

    tv_pass = _team_val(lge_pass,  sel_team_name)
    tv_drib = _team_val(lge_drib,  sel_team_name)

    gp_pass = pct_rank(tv_pass, glb_pass["value"]) if tv_pass is not None else None
    gp_drib = pct_rank(tv_drib, glb_drib["value"]) if tv_drib is not None else None

    # ── Percentile charts ──────────────────────────────────────────────────────
    rc1, rc2 = st.columns(2)
    with rc1:
        lge_pct_pass = pct_rank(tv_pass, lge_pass["value"]) if tv_pass is not None else None
        if lge_pct_pass is not None:
            st.markdown(f"**Pass xT/90 — {sel_team_league}**")
            st.markdown(pct_label(lge_pct_pass), unsafe_allow_html=True)
        fig = plot_team_rank_bar(
            lge_pass.rename(columns={"value": "pass_xt_per90"}),
            sel_team_name, "pass_xt_per90", "Avg Pass xT/90",
            global_pct=gp_pass,
        )
        st.pyplot(fig); plt.close("all")

    with rc2:
        lge_pct_drib = pct_rank(tv_drib, lge_drib["value"]) if tv_drib is not None else None
        if lge_pct_drib is not None:
            st.markdown(f"**Dribble xT/90 — {sel_team_league}**")
            st.markdown(pct_label(lge_pct_drib), unsafe_allow_html=True)
        fig = plot_team_rank_bar(
            lge_drib.rename(columns={"value": "dribble_xt_per90"}),
            sel_team_name, "dribble_xt_per90", "Avg Dribble xT/90",
            global_pct=gp_drib,
        )
        st.pyplot(fig); plt.close("all")

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
