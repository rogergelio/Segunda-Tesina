"""
data_prep.py — Precompute player-level xT stats.

Strategy:
  1. Pull match_events, matches, and players tables in full.
  2. Join and filter entirely in Python/pandas.
  3. Compute xT, aggregate per player, and save parquet outputs.

Key design:
  • Passes  (Pass / OffsidePass, Successful, has end_x/end_y) -> added_xt = end_xt - start_xt
  • Dribbles (TakeOn, Successful) -> NO end coordinates in DB -> added_xt = start_xt

Outputs:
  player_xt_stats.parquet   — one row per player
  player_events.parquet     — one row per pass/dribble event

Usage:
    cd "Segunda Tesina"
    python Streamlit/data_prep.py
"""

import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mplsoccer import Pitch

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PARAMS  = dict(dbname="postgres", user="postgres", password="admin",
                  host="localhost", port=5432)
XT_CSV     = ROOT / "Models" / "xT" / "xt.csv"
BINS       = (32, 24)
OUT_DIR      = Path(__file__).parent
STATS_OUT    = OUT_DIR / "player_xt_stats.parquet"
STATS_CSV    = OUT_DIR / "player_xt_stats.csv"
EVENTS_OUT   = OUT_DIR / "player_events.parquet"
COVERAGE_CSV = OUT_DIR / "league_season_coverage.csv"

# Filters applied at SQL level — change here to rebuild for a different scope.
TARGET_SEASON = "2024/2025"
TARGET_COMPS  = (
    "Premier League",
    "LaLiga",
    "Bundesliga",
    "Serie A",
    "Ligue 1",
    "Champions League",
    "Europa League",
)

# ── Timing helper ──────────────────────────────────────────────────────────────

@contextmanager
def _step(label: str):
    print(f"\n[data_prep] ▶  {label}", flush=True)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"[data_prep] ✓  {label}: {elapsed:.1f}s", flush=True)

pitch   = Pitch(pitch_type="opta")
xt_grid = np.loadtxt(XT_CSV, delimiter=",")   # shape (24, 32)

from constants import POSITION_MAP, GROUPED_POSITION_MAP


def fix_name(s: str) -> str:
    """
    Fix double-CP1252 encoded player names (e.g. 'MÃ¼ller' -> 'Müller').

    Encoding history in the DB:
      char -> UTF-8 bytes -> misread as CP1252 chars -> stored as UTF-8 (twice).
    Reverse: encode CP1252 -> decode UTF-8, applied up to twice.

    Names with 'ï¿½' contain U+FFFD (replacement char) — the original byte was
    lost at scrape time.  The SQL query now prefers clean records, so this path
    is a last-resort guard.
    """
    if not isinstance(s, str):
        return s
    result = s
    for _ in range(2):
        try:
            result = result.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
    # If a replacement char survived, the original is unrecoverable — strip the
    # broken prefix char (typically a stray 'Ã') that precedes it.
    if "\ufffd" in result:
        result = result.replace("Ã\ufffd", "").replace("\ufffd", "")
    return result


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(**DB_PARAMS)


def _read(conn, query: str, label: str) -> pd.DataFrame:
    print(f"[data_prep] Pulling {label} …", flush=True)
    df = pd.read_sql(query, conn)
    print(f"  -> {len(df):,} rows", flush=True)
    return df


# ── xT helpers ─────────────────────────────────────────────────────────────────

def _lookup_xt(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    stat   = pitch.bin_statistic(x_arr, y_arr, bins=BINS)
    bx, by = stat["binnumber"][0], stat["binnumber"][1]
    inside = stat["inside"]
    vals   = np.zeros(len(x_arr))
    vals[inside] = xt_grid[by[inside], bx[inside]]
    return vals


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    _T0 = time.perf_counter()
    print(f"\n{'='*60}", flush=True)
    print(f"[data_prep] START  {TARGET_SEASON}  |  {len(TARGET_COMPS)} competitions", flush=True)
    print(f"{'='*60}", flush=True)

    _comps_sql = ", ".join(f"'{c}'" for c in TARGET_COMPS)

    # ── 1. DB pulls ────────────────────────────────────────────────────────────
    with _step("Connect + pull matches"):
        conn = get_conn()
        matches = _read(conn,
            f"""
            SELECT match_id, competition, season, home_id, away_id, home_team, away_team
            FROM matches
            WHERE season = '{TARGET_SEASON}'
              AND competition IN ({_comps_sql})
            """,
            f"matches ({TARGET_SEASON})"
        )

    with _step("Pull match_events (filtered by match_id)"):
        events = _read(conn,
            f"""
            SELECT me.player_id, me.match_id, me.team_id,
                   me.x, me.y, me.end_x, me.end_y,
                   me.type_display_name, me.outcome_type_display_name,
                   me.qualifiers::text AS qualifiers,
                   me.period_display_name, me.minute, me.second
            FROM match_events me
            WHERE me.match_id IN (
                SELECT match_id FROM matches
                WHERE season = '{TARGET_SEASON}'
                  AND competition IN ({_comps_sql})
            )
            """,
            "match_events"
        )

    with _step("Pull players + position counts"):
        players = _read(conn,
            """
            SELECT DISTINCT ON (player_id)
                player_id, name, age, position, team_id
            FROM players
            ORDER BY player_id,
                (name NOT LIKE '%Ã%' AND name NOT LIKE '%ï¿½%') DESC,
                ds DESC
            """,
            "players"
        )
        MIN_POS_GAMES = 5
        players_pos_counts = _read(conn,
            """
            SELECT player_id, position,
                   COUNT(DISTINCT ds) AS games_at_pos
            FROM players
            WHERE position IS NOT NULL AND position != ''
            GROUP BY player_id, position
            """,
            "players pos counts"
        )
        conn.close()

    # ── 2. Build position lists ────────────────────────────────────────────────
    with _step("Build multi-position lists + enrich players"):
        valid_pos = players_pos_counts[
            (players_pos_counts["games_at_pos"] >= MIN_POS_GAMES) &
            (players_pos_counts["position"] != "Sub")
        ]

        def _map_pos_list(pos_pipe: str, mapping: dict) -> str:
            codes  = pos_pipe.split("|") if isinstance(pos_pipe, str) else [pos_pipe]
            mapped = []
            seen   = set()
            for c in codes:
                label = mapping.get(c, c)
                if label not in seen:
                    seen.add(label)
                    mapped.append(label)
            return "|".join(mapped)

        primary_pos_df = (
            valid_pos
            .sort_values("games_at_pos", ascending=False)
            .drop_duplicates("player_id")[["player_id", "position"]]
            .rename(columns={"position": "primary_position"})
        )
        pos_list_df = (
            valid_pos
            .groupby("player_id")["position"]
            .apply(lambda x: "|".join(sorted(x.unique())))
            .reset_index(name="position_list")
        )

        players["name_clean"]       = players["name"].apply(fix_name)
        players["position_full"]    = players["position"].map(POSITION_MAP).fillna(players["position"])
        players["grouped_position"] = players["position"].map(GROUPED_POSITION_MAP).fillna(players["position"])
        players = players.merge(pos_list_df,    on="player_id", how="left")
        players = players.merge(primary_pos_df, on="player_id", how="left")
        players["primary_position"] = players["primary_position"].fillna(players["position"])
        players["position_list"]    = players["position_list"].fillna(players["primary_position"])
        players["position_full_list"]       = players["position_list"].apply(lambda p: _map_pos_list(p, POSITION_MAP))
        players["grouped_position_list"]    = players["position_list"].apply(lambda p: _map_pos_list(p, GROUPED_POSITION_MAP))
        players["primary_position_full"]    = players["primary_position"].map(POSITION_MAP).fillna(players["primary_position"])
        players["primary_grouped_position"] = players["primary_position"].map(GROUPED_POSITION_MAP).fillna(players["primary_position"])

    # ── 3. Coerce + join + sort ────────────────────────────────────────────────
    with _step("Coerce numeric columns"):
        for col in ["x", "y", "end_x", "end_y"]:
            events[col] = pd.to_numeric(events[col], errors="coerce")
        _PERIOD_ORDER = {
            "FirstHalf": 1, "SecondHalf": 2,
            "ExtraFirstHalf": 3, "ExtraSecondHalf": 4,
            "PenaltyShootout": 5,
        }
        events["_period_num"] = events["period_display_name"].map(_PERIOD_ORDER).fillna(9)
        for col in ["minute", "second"]:
            events[col] = pd.to_numeric(events[col], errors="coerce").fillna(0)
        matches["home_id"] = pd.to_numeric(matches["home_id"], errors="coerce")
        matches["away_id"] = pd.to_numeric(matches["away_id"], errors="coerce")

    with _step("Join events → matches + derive team_name"):
        events = events.merge(
            matches[["match_id", "competition", "season", "home_id", "away_id",
                     "home_team", "away_team"]],
            on="match_id", how="left"
        )
        events["team_name"] = np.where(
            events["team_id"] == events["home_id"],
            events["home_team"],
            events["away_team"]
        )
        events.drop(columns=["home_id", "away_id", "home_team", "away_team"], inplace=True)

    with _step("Sort by match/period/minute + compute next-action coords"):
        events = events.sort_values(
            ["match_id", "_period_num", "minute", "second"], na_position="last"
        ).reset_index(drop=True)
        events["_next_x"] = events.groupby("match_id")["x"].shift(-1)
        events["_next_y"] = events.groupby("match_id")["y"].shift(-1)

    # ── 4. Match counts + set-piece filter ────────────────────────────────────
    with _step("Match counts per player × competition × season"):
        match_counts = (
            events.groupby(["player_id", "competition", "season"])["match_id"]
            .nunique()
            .reset_index(name="matches_played")
        )

    with _step("Exclude set-piece events"):
        SET_PIECE_QUALIFIERS = {
            "CornerTaken", "DirectFreekick", "IndirectFreekick",
            "GoalKick", "ThrowinSetPiece", "Penalty",
        }
        def _is_setpiece(q) -> bool:
            if not isinstance(q, str):
                return False
            return any(kw in q for kw in SET_PIECE_QUALIFIERS)

        before = len(events)
        events = events[~events["qualifiers"].apply(_is_setpiece)].copy()
        print(f"  removed {before - len(events):,} set-piece rows ({len(events):,} remaining)", flush=True)

    # ── 5. Filter event types ──────────────────────────────────────────────────
    with _step("Filter to Pass / OffsidePass / TakeOn / GoodSkill"):
        is_pass = (
            events["type_display_name"].isin(["Pass", "OffsidePass"]) &
            (events["outcome_type_display_name"] == "Successful") &
            events["end_x"].notna() & events["end_y"].notna()
        )
        is_dribble = (
            events["type_display_name"].isin(["TakeOn", "GoodSkill"]) &
            (events["outcome_type_display_name"] == "Successful")
        )
        passes   = events[is_pass].copy()
        dribbles = events[is_dribble].copy()
        print(f"  {len(passes):,} pass events  |  {len(dribbles):,} dribble events", flush=True)

    # ── 6. Compute xT ─────────────────────────────────────────────────────────
    with _step("Compute xT — passes"):
        passes["start_xt"] = _lookup_xt(passes["x"].values, passes["y"].values)
        passes["end_xt"]   = _lookup_xt(passes["end_x"].values, passes["end_y"].values)
        passes["added_xt"] = passes["end_xt"] - passes["start_xt"]
        passes["is_pass"]    = True
        passes["is_dribble"] = False

    with _step("Compute xT — dribbles"):
        dribbles["end_x"] = events.loc[dribbles.index, "_next_x"].values
        dribbles["end_y"] = events.loc[dribbles.index, "_next_y"].values
        dribbles["start_xt"] = _lookup_xt(dribbles["x"].values, dribbles["y"].values)
        has_end = dribbles["end_x"].notna() & dribbles["end_y"].notna()
        dribbles["end_xt"] = np.nan
        dribbles.loc[has_end, "end_xt"] = _lookup_xt(
            dribbles.loc[has_end, "end_x"].values,
            dribbles.loc[has_end, "end_y"].values,
        )
        dribbles["added_xt"] = dribbles["start_xt"].copy()
        dribbles.loc[has_end, "added_xt"] = (
            dribbles.loc[has_end, "end_xt"] - dribbles.loc[has_end, "start_xt"]
        )
        dribbles["is_pass"]    = False
        dribbles["is_dribble"] = True

    # ── 7. Combine + save events ───────────────────────────────────────────────
    with _step("Combine events + save player_events.parquet"):
        keep_cols = [
            "player_id", "match_id", "team_id", "team_name",
            "competition", "season",
            "x", "y", "end_x", "end_y",
            "type_display_name",
            "start_xt", "end_xt", "added_xt",
            "is_pass", "is_dribble",
        ]
        all_events = pd.concat([passes[keep_cols], dribbles[keep_cols]], ignore_index=True)
        print(f"  {len(all_events):,} total events", flush=True)
        all_events.to_parquet(EVENTS_OUT, index=False)
        print(f"  → {EVENTS_OUT}", flush=True)

    # ── 8. Aggregate xT ───────────────────────────────────────────────────────
    with _step("Aggregate xT per player × competition × season"):
        KEY = ["player_id", "competition", "season"]
        pass_agg = (
            all_events[all_events["is_pass"]]
            .groupby(KEY)
            .agg(
                pass_xt_total = ("added_xt", lambda x: x.clip(lower=0).sum()),
                pass_xt_count = ("added_xt", "count"),
            )
            .reset_index()
        )
        dribble_agg = (
            all_events[all_events["is_dribble"]]
            .groupby(KEY)
            .agg(
                dribble_xt_total = ("added_xt", "sum"),
                dribble_xt_count = ("added_xt", "count"),
            )
            .reset_index()
        )
        team_per_stint = (
            all_events
            .sort_values("match_id", ascending=False)
            .drop_duplicates(KEY)[KEY + ["team_name"]]
        )

    # ── 9. Build + save stats ──────────────────────────────────────────────────
    with _step("Build player stats table + save player_xt_stats.parquet"):
        stats = match_counts.merge(
            players[[
                "player_id", "name", "name_clean", "age",
                "position", "position_full", "grouped_position",
                "position_list", "position_full_list", "grouped_position_list",
                "primary_position", "primary_position_full", "primary_grouped_position",
            ]],
            on="player_id", how="left"
        )
        stats = stats.merge(team_per_stint, on=KEY, how="left")
        stats = stats.merge(pass_agg,       on=KEY, how="left")
        stats = stats.merge(dribble_agg,    on=KEY, how="left")

        for col in ["pass_xt_total", "pass_xt_count", "dribble_xt_total", "dribble_xt_count"]:
            stats[col] = stats[col].fillna(0.0)
        stats["matches_played"] = stats["matches_played"].fillna(1)

        stats["pass_xt_per90"]    = stats["pass_xt_total"]    / stats["matches_played"]
        stats["dribble_xt_per90"] = stats["dribble_xt_total"] / stats["matches_played"]
        stats["total_xt"]         = stats["pass_xt_total"] + stats["dribble_xt_total"]
        stats["total_xt_per90"]   = stats["total_xt"]         / stats["matches_played"]

        print(f"  {len(stats):,} player rows", flush=True)
        stats.to_parquet(STATS_OUT, index=False)
        stats.to_csv(STATS_CSV, index=False)
        print(f"  → {STATS_OUT}", flush=True)

    # ── 10. Coverage table ─────────────────────────────────────────────────────
    with _step("Build league/season coverage table"):
        events_with_names = events.merge(
            players[["player_id", "name", "name_clean"]], on="player_id", how="left"
        )
        coverage = (
            events_with_names
            .dropna(subset=["competition", "season"])
            .groupby(["competition", "season"])
            .agg(
                unique_team_ids     = ("team_id",   "nunique"),
                unique_team_names   = ("team_name", "nunique"),
                unique_player_ids   = ("player_id", "nunique"),
                unique_player_names = ("name",       "nunique"),
                unique_clean_names  = ("name_clean", "nunique"),
            )
            .reset_index()
            .sort_values(["competition", "season"])
        )
        coverage.to_csv(COVERAGE_CSV, index=False)
        print(f"  → {COVERAGE_CSV}", flush=True)
        print(coverage.to_string(index=False), flush=True)

    total = time.perf_counter() - _T0
    print(f"\n{'='*60}", flush=True)
    print(f"[data_prep] DONE  total: {total:.1f}s", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
