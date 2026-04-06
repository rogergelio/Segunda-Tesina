"""
viz.py — Reusable pitch plot functions for the xT scouting app.

Color palette — dark purple bg, mint / aqua / green accents:
  BG_DARK  = #0d0020   pitch background
  BG_CARD  = #1a0535   card / figure background
  MINT     = #00e5b0   primary accent
  AQUA     = #00bcd4   secondary accent
  GREEN    = #00ff9f   positive / above average
  AMBER    = #ffb300   neutral / selected player
  RED      = #ff4d6d   negative / below average
  LINE     = #3a1a5c   pitch lines / grid
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from mplsoccer import Pitch

# ── Palette ───────────────────────────────────────────────────────────────────
BG_DARK = "#0d0020"
BG_CARD = "#1a0535"
MINT    = "#00e5b0"
AQUA    = "#00bcd4"
GREEN   = "#00ff9f"
AMBER   = "#ffb300"
RED     = "#ff4d6d"
LINE    = "#3a1a5c"
WHITE   = "#ffffff"
DIM     = "#8080a0"

BINS   = (32, 24)
ZONE_W = 4          # bins per zone in x  → 8 zone columns
ZONE_H = 4          # bins per zone in y  → 6 zone rows

pitch = Pitch(
    pitch_type="opta",
    line_zorder=2,
    line_color=LINE,
    pitch_color=BG_DARK,
)

CMAP_MINT = matplotlib.colors.LinearSegmentedColormap.from_list(
    "mint", [BG_DARK, "#004d3d", MINT])
CMAP_AQUA = matplotlib.colors.LinearSegmentedColormap.from_list(
    "aqua", [BG_DARK, "#003340", AQUA])
CMAP_DIFF = matplotlib.colors.LinearSegmentedColormap.from_list(
    "diff", [RED, BG_DARK, GREEN])

FIG_W, FIG_H = 10, 6.5
STROKE = [pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _base_fig(nrows=1, ncols=1, **kwargs):
    fig, axes = pitch.draw(
        nrows=nrows, ncols=ncols,
        figsize=(FIG_W * ncols, FIG_H * nrows), **kwargs
    )
    fig.patch.set_facecolor(BG_CARD)
    return fig, axes


def _subtitle(ax, text):
    ax.set_title(text, color=WHITE, fontsize=12, fontweight="bold", pad=6)


def _lookup_xt(x_arr, y_arr, xt_grid):
    stat   = pitch.bin_statistic(x_arr, y_arr, bins=BINS)
    bx, by = stat["binnumber"][0], stat["binnumber"][1]
    inside = stat["inside"]
    vals   = np.zeros(len(x_arr))
    vals[inside] = xt_grid[by[inside], bx[inside]]
    return vals, stat["inside"]


def _draw_xt_grid(ax):
    """Overlay xT bin grid lines on an already-drawn pitch axes."""
    x_step = pitch.dim.length / BINS[0]
    y_step = pitch.dim.width  / BINS[1]
    for i in range(1, BINS[0]):
        ax.axvline(pitch.dim.left   + i * x_step,
                   color=LINE, lw=0.35, alpha=0.6, zorder=1)
    for j in range(1, BINS[1]):
        ax.axhline(pitch.dim.bottom + j * y_step,
                   color=LINE, lw=0.35, alpha=0.6, zorder=1)


def _colorbar(hm, ax):
    cb = plt.colorbar(hm, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_tick_params(color=WHITE)
    cb.ax.tick_params(colors=WHITE)


def _bin_stat(df, x_col, y_col, values=None):
    if values is not None:
        return pitch.bin_statistic(df[x_col], df[y_col],
                                   statistic="sum", values=values, bins=BINS)
    return pitch.bin_statistic(df[x_col], df[y_col],
                               statistic="sum",
                               values=df["start_xt"].values, bins=BINS)


# ─── Tab 2: xT Visualizer ─────────────────────────────────────────────────────

def plot_xt_heatmap(
    events_df: pd.DataFrame,
    xt_grid: np.ndarray,
    event_type: str = "pass",   # "pass" | "dribble"
    use_end: bool = False,       # True = end position (passes only)
    title: str = "",
    vmax: float = None,          # shared colour scale for side-by-side comparisons
):
    """
    Single heatmap weighted by xT.

    event_type="pass",  use_end=False → pass start positions,  weighted by start_xt
    event_type="pass",  use_end=True  → pass end positions,    weighted by end_xt
    event_type="dribble"              → dribble start positions, weighted by start_xt
    """
    flag = "is_pass" if event_type == "pass" else "is_dribble"
    df = events_df[events_df[flag]].copy()

    # Per-90 denominator: number of unique matches the player appeared in
    n90 = max(
        events_df["match_id"].nunique() if "match_id" in events_df.columns else 1, 1
    )

    fig, ax = _base_fig()
    if df.empty:
        ax.text(50, 50, "No data", color=WHITE, ha="center", va="center", fontsize=12)
        if title:
            _subtitle(ax, title)
        return fig

    if use_end:
        mask = df["end_x"].notna() & df["end_y"].notna()
        df   = df[mask]
        x, y = df["end_x"].values, df["end_y"].values
        vals = df["end_xt"].values
        cmap = CMAP_AQUA
    else:
        x, y = df["x"].values, df["y"].values
        vals = df["start_xt"].values
        cmap = CMAP_MINT

    stat = dict(pitch.bin_statistic(x, y, statistic="sum", values=vals, bins=BINS))
    stat["statistic"] = stat["statistic"] / n90
    effective_vmax = vmax if vmax else stat["statistic"].max() or 1
    hm = pitch.heatmap(stat, ax=ax, cmap=cmap,
                       vmin=0, vmax=effective_vmax, edgecolors="none")
    _colorbar(hm, ax)
    if title:
        _subtitle(ax, title)
    return fig


def plot_vs_average_pair(
    player_df: pd.DataFrame,
    avg_df: pd.DataFrame,
    xt_grid: np.ndarray,
    event_type: str = "pass",
    use_end: bool = False,
    row_title: str = "",
    player_name: str = "Player",
):
    """
    Side-by-side: [Player | Position Average] for one heatmap type.
    Both plots use mean xT per action per bin so volume differences between
    one player and the entire peer group don't break the colour scale.
    """
    flag = "is_pass" if event_type == "pass" else "is_dribble"

    def _coords_vals(df):
        sub = df[df[flag]].copy()
        if sub.empty:
            return None, None, None
        if use_end:
            sub = sub[sub["end_x"].notna() & sub["end_y"].notna()]
            if sub.empty:
                return None, None, None
            return sub["end_x"].values, sub["end_y"].values, sub["end_xt"].values
        return sub["x"].values, sub["y"].values, sub["start_xt"].values

    px, py, pv = _coords_vals(player_df)
    ax_, ay, av = _coords_vals(avg_df)

    # Per-90 denominators:
    #   player  → unique matches for that player
    #   peers   → sum of unique matches per peer player (total peer-90s)
    player_n90 = max(
        player_df["match_id"].nunique() if "match_id" in player_df.columns else 1, 1
    )
    if "player_id" in avg_df.columns and "match_id" in avg_df.columns:
        peer_n90 = max(
            avg_df.groupby("player_id")["match_id"].nunique().sum(), 1
        )
    else:
        peer_n90 = max(avg_df["match_id"].nunique() if "match_id" in avg_df.columns else 1, 1)

    def _stat_sum(x, y, v, divisor=1):
        if x is None or len(x) == 0:
            return None
        st = pitch.bin_statistic(x, y, statistic="sum", values=v, bins=BINS)
        st = dict(st)
        st["statistic"] = st["statistic"] / divisor
        return st

    s_p = _stat_sum(px, py, pv, divisor=player_n90)
    s_a = _stat_sum(ax_, ay, av, divisor=peer_n90)
    vals_p = s_p["statistic"] if s_p is not None else np.array([0])
    vals_a = s_a["statistic"] if s_a is not None else np.array([0])
    vmax = max(np.nanmax(np.nan_to_num(vals_p)), np.nanmax(np.nan_to_num(vals_a))) or 1.0

    # Difference grid (player − normalised avg per zone)
    zero_grid = np.zeros((BINS[1], BINS[0]))
    diff = (s_p["statistic"] if s_p is not None else zero_grid) \
         - (s_a["statistic"] if s_a is not None else zero_grid)
    abs_max = np.nanmax(np.abs(diff)) or 1.0

    cmap = CMAP_AQUA if use_end else CMAP_MINT
    fig, axes = _base_fig(nrows=1, ncols=3)
    ax_p, ax_a, ax_d = axes[0], axes[1], axes[2]

    for stat, ax, lbl in [(s_p, ax_p, f"{player_name} xT/90"), (s_a, ax_a, "Position Avg xT/90")]:
        if stat is None:
            ax.text(50, 50, "No data", color=WHITE, ha="center", va="center")
        else:
            hm = pitch.heatmap(stat, ax=ax, cmap=cmap,
                               vmin=0, vmax=vmax, edgecolors="none")
            _colorbar(hm, ax)
        _subtitle(ax, lbl)

    # Difference panel — green = above avg, red = below avg, BG_DARK ≈ zero
    diff_stat = dict(s_p if s_p is not None else s_a)
    diff_stat["statistic"] = diff
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    hm_d = pitch.heatmap(diff_stat, ax=ax_d, cmap=CMAP_DIFF,
                         norm=norm, edgecolors="none")
    _colorbar(hm_d, ax_d)
    _subtitle(ax_d, "Difference (Player − Avg)")

    if row_title:
        fig.suptitle(row_title, color=MINT, fontsize=13, fontweight="bold", y=0.98)
    return fig


def plot_top_plays(
    events_df: pd.DataFrame,
    xt_grid: np.ndarray,
    position_avg_xt: float = None,
    n: int = 5,
    title: str = "Top 5 xT Plays",
):
    """
    Arrow plot for the N most frequent positive-xT plays.

    • Pitch grid matching the 32×24 xT bins is drawn behind the arrows.
    • Actions are grouped into 4×4-bin zones (8×6 = 48 zones total).
    • Each arrow is labelled: rank, pass/dribble, share of all actions, avg xT vs avg.
    • Arrow width ∝ frequency; colour = GREEN / AMBER / RED vs position average.
    """
    df = events_df[events_df["added_xt"] > 0].copy()
    total_actions = len(events_df)   # denominator for share % (all player actions)

    fig, ax = _base_fig()

    # ── xT grid overlay ──────────────────────────────────────────────────────
    _draw_xt_grid(ax)

    if df.empty or len(df) < 3:
        ax.text(50, 50, "Not enough data", color=WHITE,
                ha="center", va="center", fontsize=11)
        _subtitle(ax, title)
        return fig

    # ── Bin → zone assignment ─────────────────────────────────────────────────
    s_stat = pitch.bin_statistic(df["x"].values, df["y"].values, bins=BINS)
    # For passes use end coords; for dribbles fall back to start coords
    plot_ex = np.where(df["is_pass"].values & df["end_x"].notna().values,
                       df["end_x"].values, df["x"].values)
    plot_ey = np.where(df["is_pass"].values & df["end_y"].notna().values,
                       df["end_y"].values, df["y"].values)
    e_stat = pitch.bin_statistic(plot_ex, plot_ey, bins=BINS)

    df = df.copy()
    df["s_bx"]    = s_stat["binnumber"][0]
    df["s_by"]    = s_stat["binnumber"][1]
    df["e_bx"]    = e_stat["binnumber"][0]
    df["e_by"]    = e_stat["binnumber"][1]
    df["plot_ex"] = plot_ex
    df["plot_ey"] = plot_ey
    df = df[s_stat["inside"] & e_stat["inside"]]

    # Map bins to zones
    df["s_zx"] = df["s_bx"] // ZONE_W
    df["s_zy"] = df["s_by"] // ZONE_H
    df["e_zx"] = df["e_bx"] // ZONE_W
    df["e_zy"] = df["e_by"] // ZONE_H

    # Drop same-zone events (no visible movement)
    df = df[(df["s_zx"] != df["e_zx"]) | (df["s_zy"] != df["e_zy"])]

    def _agg(sub, label):
        if sub.empty:
            return pd.DataFrame()
        return (sub.groupby(["s_zx", "s_zy", "e_zx", "e_zy"])
                   .agg(
                       freq    = ("added_xt", "count"),
                       avg_xt  = ("added_xt", "mean"),
                       # mean actual coords → avoids any coordinate-system inversion
                       xs = ("x",       "mean"),
                       ys = ("y",       "mean"),
                       xe = ("plot_ex", "mean"),
                       ye = ("plot_ey", "mean"),
                   )
                   .reset_index()
                   .assign(event_type=label))

    combined = pd.concat([
        _agg(df[df["is_pass"]],    "Pass"),
        _agg(df[df["is_dribble"]], "Dribble"),
    ], ignore_index=True)

    if combined.empty:
        ax.text(50, 50, "Not enough data", color=WHITE,
                ha="center", va="center", fontsize=11)
        _subtitle(ax, title)
        return fig

    top = combined.nlargest(n, "freq").reset_index(drop=True)

    global_avg = position_avg_xt if position_avg_xt is not None else top["avg_xt"].mean()
    max_freq   = top["freq"].max()

    for i, row in top.iterrows():
        ratio = row["avg_xt"] / global_avg if global_avg > 0 else 1.0
        color = GREEN if ratio >= 1.10 else (RED if ratio <= 0.90 else AMBER)
        lw    = 1.5 + 5.0 * (row["freq"] / max_freq)

        pitch.arrows(row["xs"], row["ys"], row["xe"], row["ye"],
                     ax=ax, width=lw, color=color, alpha=0.9,
                     zorder=4, headwidth=5, headlength=4)

        share  = row["freq"] / total_actions * 100 if total_actions > 0 else 0
        pct_vs = (ratio - 1) * 100
        sign   = "+" if pct_vs >= 0 else ""
        label  = (f"#{i + 1} {row['event_type']}\n"
                  f"{share:.1f}% of actions\n"
                  f"xT {row['avg_xt']:.2f}  {sign}{pct_vs:.0f}% vs avg")

        ax.text(row["xs"], row["ys"] + 2.0, label,
                color=color, fontsize=6, ha="center", va="bottom",
                zorder=5, path_effects=STROKE)

    for lbl, clr in [("Above avg (+10%)", GREEN),
                     ("Near avg",         AMBER),
                     ("Below avg (-10%)", RED)]:
        ax.plot([], [], color=clr, lw=3, label=lbl)
    ax.legend(loc="lower left", fontsize=7, facecolor=BG_CARD,
              labelcolor=WHITE, framealpha=0.8)

    _subtitle(ax, f"{title}  ·  zones = 4×4 xT bins  ·  width ∝ frequency")
    return fig


# ─── Tab 3: Player Comparison ─────────────────────────────────────────────────

def plot_comparison_trio(
    p1_events: pd.DataFrame,
    p2_events: pd.DataFrame,
    p1_name: str,
    p2_name: str,
    event_type: str = "pass",
    use_end: bool = False,
    title: str = "",
):
    """
    Three-panel comparison: [Player 1 xT/90 | Player 2 xT/90 | Difference (P1 − P2)].
    Each player normalised by their own match count for a fair per-90 comparison.
    Difference panel: green = P1 above P2, red = P1 below P2.
    """
    flag = "is_pass" if event_type == "pass" else "is_dribble"
    cmap = CMAP_AQUA if use_end else CMAP_MINT

    def _prep(ev):
        sub = ev[ev[flag]].copy()
        if use_end:
            sub = sub[sub["end_x"].notna() & sub["end_y"].notna()]
            return sub["end_x"].values, sub["end_y"].values, sub["end_xt"].values
        return sub["x"].values, sub["y"].values, sub["start_xt"].values

    def _stat(x, y, v, n90):
        if len(x) == 0:
            return None
        s = dict(pitch.bin_statistic(x, y, statistic="sum", values=v, bins=BINS))
        s["statistic"] = s["statistic"] / max(n90, 1)
        return s

    n90_1 = max(p1_events["match_id"].nunique() if "match_id" in p1_events.columns else 1, 1)
    n90_2 = max(p2_events["match_id"].nunique() if "match_id" in p2_events.columns else 1, 1)

    x1, y1, v1 = _prep(p1_events)
    x2, y2, v2 = _prep(p2_events)
    s1 = _stat(x1, y1, v1, n90_1)
    s2 = _stat(x2, y2, v2, n90_2)

    zero = np.zeros((BINS[1], BINS[0]))
    g1   = s1["statistic"] if s1 is not None else zero
    g2   = s2["statistic"] if s2 is not None else zero

    vmax    = max(np.nanmax(g1), np.nanmax(g2)) or 1.0
    diff    = g1 - g2
    abs_max = np.nanmax(np.abs(diff)) or 1.0

    fig, axes = _base_fig(nrows=1, ncols=3)

    for stat, ax, name in [(s1, axes[0], p1_name), (s2, axes[1], p2_name)]:
        if stat is None:
            ax.text(50, 50, "No data", color=WHITE, ha="center", va="center")
        else:
            hm = pitch.heatmap(stat, ax=ax, cmap=cmap,
                               vmin=0, vmax=vmax, edgecolors="none")
            _colorbar(hm, ax)
        _subtitle(ax, name)

    diff_stat = dict(s1 if s1 is not None else s2 if s2 is not None
                     else {"statistic": zero, "binnumber": None, "inside": None})
    diff_stat["statistic"] = diff
    norm  = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    hm_d  = pitch.heatmap(diff_stat, ax=axes[2], cmap=CMAP_DIFF,
                          norm=norm, edgecolors="none")
    _colorbar(hm_d, axes[2])
    _subtitle(axes[2], f"Difference  ({p1_name} − {p2_name})")

    if title:
        fig.suptitle(title, color=MINT, fontsize=13, fontweight="bold", y=0.98)
    return fig


# ─── Tab 1: Rankings ──────────────────────────────────────────────────────────

def plot_rank_bar(
    player_row: pd.Series,
    peers_df: pd.DataFrame,
    metric: str,
    label: str,
):
    """
    Horizontal bar chart — selected player highlighted in mint.
    Title / context text is rendered by the caller as st.markdown so it is
    outside the figure and easier to read.
    """
    name_col = "name_clean" if "name_clean" in peers_df.columns else "name"

    top_n = peers_df.nlargest(15, metric).copy()
    pid   = player_row["player_id"]
    if pid not in top_n["player_id"].values:
        top_n = pd.concat([top_n, peers_df[peers_df["player_id"] == pid]]).head(16)

    top_n = top_n.sort_values(metric).reset_index(drop=True)

    # Add "(League)" suffix where the same name appears more than once
    dup_names = top_n[name_col][top_n[name_col].duplicated(keep=False)]
    if "competition" in top_n.columns and not dup_names.empty:
        top_n["_bar_label"] = np.where(
            top_n[name_col].isin(dup_names),
            top_n[name_col] + " (" + top_n["competition"].fillna("?") + ")",
            top_n[name_col],
        )
    else:
        top_n["_bar_label"] = top_n[name_col]

    colors = [MINT if p == pid else "#2a4060" for p in top_n["player_id"]]

    fig, ax = plt.subplots(figsize=(7, max(4, len(top_n) * 0.48)))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD)

    bars = ax.barh(top_n["_bar_label"], top_n[metric], color=colors, height=0.65)

    ax.set_xlabel(label, color=WHITE, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors=DIM)
    ax.tick_params(axis="y", colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)

    max_val = top_n[metric].max()
    for bar, val in zip(bars, top_n[metric]):
        ax.text(bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", color=WHITE, fontsize=8)

    fig.tight_layout()
    return fig


# ─── Tab 4: Team Analysis ─────────────────────────────────────────────────────

def plot_team_rank_bar(
    team_agg: pd.DataFrame,
    sel_team: str,
    metric: str,
    label: str,
    global_pct: float = None,
):
    """
    Horizontal bar chart of teams sorted by `metric`.
    `team_agg` must have columns: 'team_name' and `metric`.
    Selected team highlighted in mint; all others in navy.
    `global_pct` (0–100), when provided, is annotated as a subtitle.
    """
    df = team_agg.sort_values(metric).reset_index(drop=True)
    colors = [MINT if t == sel_team else "#2a4060" for t in df["team_name"]]

    fig, ax = plt.subplots(figsize=(7, max(4, len(df) * 0.48)))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD)

    bars = ax.barh(df["team_name"], df[metric], color=colors, height=0.65)

    ax.set_xlabel(label, color=WHITE, fontsize=10)
    ax.xaxis.label.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors=DIM)
    ax.tick_params(axis="y", colors=WHITE, labelsize=9)

    max_val = df[metric].max() or 1
    for bar, val in zip(bars, df[metric]):
        ax.text(bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", color=WHITE, fontsize=8)

    if global_pct is not None:
        top_global = 100 - global_pct
        ax.set_title(f"Global ranking: Top {top_global:.0f}%",
                     color=DIM, fontsize=9, pad=4)

    fig.tight_layout()
    return fig
