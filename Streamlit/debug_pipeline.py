"""
debug_pipeline.py — Step-by-step integrity audit of the xT pipeline.

At every step prints a pivot:  season × league  →  teams | players
The team count per (league, season) should stay constant across all steps.
Player count will shrink as filters are applied — that is expected.

Bundesliga is used as the reference example throughout.

Run:
    cd "Segunda Tesina"
    python Streamlit/debug_pipeline.py 2>&1 | tee Streamlit/debug_pipeline.log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DB_PARAMS = dict(dbname="postgres", user="postgres", password="admin",
                 host="localhost", port=5432)

EXAMPLE_LEAGUE = "Bundesliga"
W = 80

# ── Display helpers ────────────────────────────────────────────────────────────

def banner(step: str, subtitle: str = ""):
    print(f"\n{'=' * W}")
    print(f"  {step}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'=' * W}")


def pivot_teams_players(
    df: pd.DataFrame,
    step_label: str,
    comp_col: str = "competition",
    season_col: str = "season",
    player_col: str = "player_id",
    team_col: str = "team_name",
):
    """
    Print a compact pivot:  season  ×  league  →  teams | players
    Then print just the Bundesliga rows as the reference example.
    """
    if comp_col not in df.columns or season_col not in df.columns:
        print(f"  [SKIP — no {comp_col}/{season_col} columns]")
        return

    pivot = (
        df.dropna(subset=[comp_col, season_col])
        .groupby([season_col, comp_col], sort=True)
        .agg(
            teams   = (team_col,   "nunique") if team_col   in df.columns else (player_col, lambda x: None),
            players = (player_col, "nunique"),
        )
        .reset_index()
    )

    # ── Full pivot (all leagues) ───────────────────────────────────────────────
    print(f"\n  ── {step_label}: ALL LEAGUES ──")
    full = pivot.pivot_table(
        index=comp_col, columns=season_col,
        values=["teams", "players"],
        aggfunc="sum",
    ).fillna(0).astype(int)
    # Flatten column names: (teams, 2024/2025) → "teams 2024/2025"
    full.columns = [f"{m} {s}" for m, s in full.columns]
    full = full.sort_index()
    print(full.to_string())

    # ── Bundesliga focus ───────────────────────────────────────────────────────
    bl = pivot[pivot[comp_col] == EXAMPLE_LEAGUE].sort_values(season_col)
    print(f"\n  ── {step_label}: {EXAMPLE_LEAGUE} only ──")
    if bl.empty:
        print(f"  (no {EXAMPLE_LEAGUE} rows at this step)")
    else:
        print(bl[[season_col, "teams", "players"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
conn = psycopg2.connect(**DB_PARAMS)

# ── STEP 0a: match_events ──────────────────────────────────────────────────────
banner("STEP 0a — Raw pull: match_events",
       "No league info yet — just row / player / team totals")
print("  Pulling all match_events …", flush=True)
events_raw = pd.read_sql(
    """
    SELECT player_id, match_id, team_id,
           x, y, end_x, end_y,
           type_display_name, outcome_type_display_name
    FROM match_events
    """,
    conn,
)
print(f"  Rows: {len(events_raw):>10,}")
print(f"  Unique players:  {events_raw['player_id'].nunique():>7,}")
print(f"  Unique team_ids: {events_raw['team_id'].nunique():>7,}")
print(f"  Unique matches:  {events_raw['match_id'].nunique():>7,}")
print(f"\n  Event-type breakdown (top 15):")
print(events_raw["type_display_name"].value_counts().head(15).to_string())


# ── STEP 0b: matches ──────────────────────────────────────────────────────────
banner("STEP 0b — Raw pull: matches",
       "Reference: how many teams exist per (league, season) in the source")
matches = pd.read_sql(
    "SELECT match_id, competition, season, home_id, away_id, home_team, away_team FROM matches",
    conn,
)
for col in ["home_id", "away_id"]:
    matches[col] = pd.to_numeric(matches[col], errors="coerce")

# Build one row per team per match so we can count distinct teams per (comp, season)
home = matches[["competition", "season", "home_id", "home_team"]].rename(
    columns={"home_id": "team_id", "home_team": "team_name"})
away = matches[["competition", "season", "away_id", "away_team"]].rename(
    columns={"away_id": "team_id", "away_team": "team_name"})
teams_ref = pd.concat([home, away]).drop_duplicates()
teams_ref["player_id"] = teams_ref["team_id"]   # placeholder so pivot works

pivot_teams_players(teams_ref, "matches table",
                    team_col="team_name", player_col="team_id")


# ── STEP 0c: players ──────────────────────────────────────────────────────────
banner("STEP 0c — Raw pull: players", "Latest record per player_id")
players = pd.read_sql(
    """
    SELECT DISTINCT ON (player_id)
        player_id, name, age, position, team_id
    FROM players
    ORDER BY player_id, ds DESC
    """,
    conn,
)
conn.close()
print(f"  Unique players: {len(players):,}")
print(f"\n  Position breakdown:")
print(players["position"].value_counts().to_string())


# ── STEP 1: Join events → matches ─────────────────────────────────────────────
banner("STEP 1 — Join events → matches",
       "Attach competition / season / team_name to every event")
events = events_raw.merge(
    matches[["match_id", "competition", "season",
             "home_id", "away_id", "home_team", "away_team"]],
    on="match_id", how="left",
)
events["team_name"] = np.where(
    events["team_id"] == events["home_id"],
    events["home_team"],
    events["away_team"],
)
events.drop(columns=["home_id", "away_id", "home_team", "away_team"], inplace=True)

print(f"  Events with no competition (unmatched match_id): "
      f"{events['competition'].isna().sum():,}")
pivot_teams_players(events, "Step 1 — all events after join")


# ── STEP 2: Coerce numeric coords ─────────────────────────────────────────────
banner("STEP 2 — Coerce x/y/end_x/end_y to numeric")
for col in ["x", "y", "end_x", "end_y"]:
    events[col] = pd.to_numeric(events[col], errors="coerce")

# Match count per player (over all events, before filtering)
match_counts = (
    events.groupby("player_id")["match_id"]
    .nunique()
    .reset_index(name="matches_played")
)
print(f"  match_counts rows: {len(match_counts):,}")
print(f"  matches_played — min {match_counts['matches_played'].min()}, "
      f"median {match_counts['matches_played'].median():.0f}, "
      f"max {match_counts['matches_played'].max()}")
# No player/team count change here — same as Step 1


# ── STEP 3: Pass filter ────────────────────────────────────────────────────────
banner("STEP 3 — Filter: Pass / OffsidePass + Successful + end_x not null")
mask_pass = (
    events["type_display_name"].isin(["Pass", "OffsidePass"]) &
    (events["outcome_type_display_name"] == "Successful") &
    events["end_x"].notna() &
    events["end_y"].notna()
)
passes = events[mask_pass].copy()
print(f"  Pass events kept: {len(passes):,}")

all_pass_rows = events[events["type_display_name"].isin(["Pass", "OffsidePass"])]
print(f"  Total Pass/OffsidePass rows:            {len(all_pass_rows):>10,}")
print(f"  → non-Successful dropped:               "
      f"{(all_pass_rows['outcome_type_display_name'] != 'Successful').sum():>10,}")
succ = all_pass_rows[all_pass_rows["outcome_type_display_name"] == "Successful"]
print(f"  → Successful but end_x null dropped:   {succ['end_x'].isna().sum():>10,}")
print(f"  → Final kept:                           {len(passes):>10,}")

pivot_teams_players(passes, "Step 3 — passes")


# ── STEP 4: Dribble filter ────────────────────────────────────────────────────
banner("STEP 4 — Filter: TakeOn + Successful  (end_x is always null in DB)")
mask_drib = (
    (events["type_display_name"] == "TakeOn") &
    (events["outcome_type_display_name"] == "Successful")
)
dribbles = events[mask_drib].copy()
all_takeons = events[events["type_display_name"] == "TakeOn"]
print(f"  Total TakeOn rows:          {len(all_takeons):>10,}")
print(f"  → Unsuccessful dropped:     "
      f"{(all_takeons['outcome_type_display_name'] != 'Successful').sum():>10,}")
print(f"  → end_x available:          {all_takeons['end_x'].notna().sum():>10,}  (expected 0)")
print(f"  → Dribble events kept:      {len(dribbles):>10,}")

pivot_teams_players(dribbles, "Step 4 — dribbles")


# ── STEP 5: Player metadata from combined events ───────────────────────────────
banner("STEP 5 — Per-player metadata",
       "Most recent (competition, season, team_name) from their last xT event")
combined = pd.concat([passes, dribbles], ignore_index=True)
player_meta = (
    combined
    .sort_values("season", ascending=False)
    .drop_duplicates("player_id")[["player_id", "competition", "season", "team_name"]]
)
print(f"  Players assigned metadata: {len(player_meta):,}")
pivot_teams_players(player_meta, "Step 5 — player_meta")


# ── STEP 6: Final merge ───────────────────────────────────────────────────────
banner("STEP 6 — Final merge: players + match_counts + player_meta + aggs")
pass_agg    = passes.groupby("player_id").size().reset_index(name="pass_xt_count")
dribble_agg = dribbles.groupby("player_id").size().reset_index(name="dribble_xt_count")

stats = players.merge(match_counts, on="player_id", how="left")
stats = stats.merge(player_meta,   on="player_id", how="left")
stats = stats.merge(pass_agg,      on="player_id", how="left")
stats = stats.merge(dribble_agg,   on="player_id", how="left")
stats["pass_xt_count"]    = stats["pass_xt_count"].fillna(0)
stats["dribble_xt_count"] = stats["dribble_xt_count"].fillna(0)
stats["matches_played"]   = stats["matches_played"].fillna(1)

print(f"  Total rows (players): {len(stats):,}")
print(f"  No competition:       {stats['competition'].isna().sum():,}")
pivot_teams_players(stats, "Step 6 — final stats")


# ── STEP 7: Parquet output check ──────────────────────────────────────────────
banner("STEP 7 — Parquet output check (only if data_prep.py has been run)")
STATS_PATH  = Path(__file__).parent / "player_xt_stats.parquet"
EVENTS_PATH = Path(__file__).parent / "player_events.parquet"

if STATS_PATH.exists():
    s = pd.read_parquet(STATS_PATH)
    print(f"  player_xt_stats.parquet → {len(s):,} rows")
    pivot_teams_players(s, "Parquet — player_xt_stats")
else:
    print("  player_xt_stats.parquet not found — run data_prep.py first.")

if EVENTS_PATH.exists():
    e = pd.read_parquet(EVENTS_PATH)
    print(f"\n  player_events.parquet → {len(e):,} rows")
    print(f"  is_pass=True:    {e['is_pass'].sum():>10,}")
    print(f"  is_dribble=True: {e['is_dribble'].sum():>10,}")
    pivot_teams_players(e, "Parquet — player_events")
else:
    print("  player_events.parquet not found — run data_prep.py first.")


print(f"\n{'=' * W}")
print("  DEBUG COMPLETE")
print(f"{'=' * W}")
