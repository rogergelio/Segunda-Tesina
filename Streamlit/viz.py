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
    ax.set_title(text, color=DIM, fontsize=9, pad=4)


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

    fig, ax = _base_fig()
    if df.empty:
        ax.text(50, 50, "No data", color=WHITE, ha="center", va="center", fontsize=12)
        if title:
            _subtitle(ax, title)
        return fig

    if use_end and event_type == "pass":
        mask = df["end_x"].notna() & df["end_y"].notna()
        df   = df[mask]
        x, y = df["end_x"].values, df["end_y"].values
        vals = df["end_xt"].values
        cmap = CMAP_AQUA
    else:
        x, y = df["x"].values, df["y"].values
        vals = df["start_xt"].values
        cmap = CMAP_MINT

    stat = pitch.bin_statistic(x, y, statistic="sum", values=vals, bins=BINS)
    hm   = pitch.heatmap(stat, ax=ax, cmap=cmap,
                         vmin=0, vmax=vmax if vmax else stat["statistic"].max() or 1,
                         edgecolors="none")
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
        if use_end and event_type == "pass":
            sub = sub[sub["end_x"].notna() & sub["end_y"].notna()]
            return sub["end_x"].values, sub["end_y"].values, sub["end_xt"].values
        return sub["x"].values, sub["y"].values, sub["start_xt"].values

    px, py, pv = _coords_vals(player_df)
    ax_, ay, av = _coords_vals(avg_df)

    # Normalise peer group by number of unique players so both sides show
    # "total xT accumulated in each zone" on a per-player basis.
    n_avg_players = (
        avg_df["player_id"].nunique()
        if avg_df is not None and "player_id" in avg_df.columns
        else 1
    ) or 1

    def _stat_sum(x, y, v, divisor=1):
        if x is None or len(x) == 0:
            return None
        st = pitch.bin_statistic(x, y, statistic="sum", values=v, bins=BINS)
        st = dict(st)                          # shallow copy so we can mutate
        st["statistic"] = st["statistic"] / divisor
        return st

    s_p = _stat_sum(px, py, pv, divisor=1)
    s_a = _stat_sum(ax_, ay, av, divisor=n_avg_players)
    vals_p = s_p["statistic"] if s_p is not None else np.array([0])
    vals_a = s_a["statistic"] if s_a is not None else np.array([0])
    vmax = max(np.nanmax(np.nan_to_num(vals_p)), np.nanmax(np.nan_to_num(vals_a))) or 1.0

    cmap = CMAP_AQUA if use_end else CMAP_MINT
    fig, axes = _base_fig(nrows=1, ncols=2)
    ax_p, ax_a = axes[0], axes[1]

    for stat, ax, lbl in [(s_p, ax_p, "Selected Player"), (s_a, ax_a, "Position Avg (per player)")]:
        if stat is None:
            ax.text(50, 50, "No data", color=WHITE, ha="center", va="center")
        else:
            hm = pitch.heatmap(stat, ax=ax, cmap=cmap,
                               vmin=0, vmax=vmax, edgecolors="none")
            _colorbar(hm, ax)
        _subtitle(ax, lbl)

    if row_title:
        fig.suptitle(row_title, color=MINT, fontsize=12, fontweight="bold", y=1.02)
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

def plot_comparison(
    p1_events: pd.DataFrame,
    p2_events: pd.DataFrame,
    p1_name: str,
    p2_name: str,
    event_flag: str = "is_pass",
    weight_by: str = "added_xt",
    xt_grid: np.ndarray = None,
):
    """Side-by-side heatmaps for two players."""
    label = "Passing" if event_flag == "is_pass" else "Dribbling"
    use_end = (event_flag == "is_pass")
    cmap = CMAP_AQUA if use_end else CMAP_MINT

    def _prep(ev):
        sub = ev[ev[event_flag]].copy()
        if use_end:
            sub = sub[sub["end_x"].notna() & sub["end_y"].notna()]
            return sub["end_x"].values, sub["end_y"].values, sub["end_xt"].values
        return sub["x"].values, sub["y"].values, sub["start_xt"].values

    x1, y1, v1 = _prep(p1_events)
    x2, y2, v2 = _prep(p2_events)

    def _stat(x, y, v):
        if len(x) == 0:
            return None
        return pitch.bin_statistic(x, y, statistic="sum", values=v, bins=BINS)

    s1, s2 = _stat(x1, y1, v1), _stat(x2, y2, v2)
    vmax = max(
        np.nanmax(s1["statistic"]) if s1 else 0,
        np.nanmax(s2["statistic"]) if s2 else 0,
    ) or 1.0

    fig, axes = _base_fig(nrows=1, ncols=2)
    for stat, ax, name in [(s1, axes[0], p1_name), (s2, axes[1], p2_name)]:
        if stat is None:
            ax.text(50, 50, "No data", color=WHITE, ha="center", va="center")
        else:
            hm = pitch.heatmap(stat, ax=ax, cmap=cmap,
                               vmin=0, vmax=vmax, edgecolors="none")
            _colorbar(hm, ax)
        _subtitle(ax, name)

    fig.suptitle(f"{label} xT Comparison", color=MINT,
                 fontsize=13, fontweight="bold", y=1.01)
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
