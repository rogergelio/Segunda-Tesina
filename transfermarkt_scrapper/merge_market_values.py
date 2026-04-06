# merge_market_values.py
# Cross-references player_xt_stats.parquet (2024/25) with market_values_2024_25.csv
# Adds a market_value column (EUR millions) and saves the updated parquet.
# Prints the most valuable players that could not be matched.
#
# Matching strategy:
#   1. Normalize team names (strip accents, "FC"/"CF"/etc., lowercase)
#   2. Apply a manual mapping for TM full names -> XT short names
#   3. Within matched teams, normalize player names and exact-match first,
#      then fall back to token-sort fuzzy match (difflib).

import re
import unicodedata
import difflib
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────

MV_CSV   = "transfermarkt_scrapper/market_values_2024_25.csv"
XT_PAR   = "Streamlit/player_xt_stats.parquet"
OUT_PAR  = "Streamlit/player_xt_stats.parquet"   # overwrite in-place
OUT_CSV  = "Streamlit/player_xt_stats.csv"

TOP5_LEAGUES_XT = {"Premier League", "LaLiga", "Bundesliga", "Serie A", "Ligue 1"}
SEASON_XT       = "2024/2025"

FUZZY_THRESHOLD       = 0.72   # difflib ratio cutoff for within-team fuzzy match
FUZZY_THRESHOLD_XTEAM = 0.92   # stricter cutoff for cross-team (transfer) fallback

# ── TM full name (normalised) → XT short name (normalised) ───────────────────
# Covers both transfermarkt.us (English) and legacy .es (Spanish) names.
TEAM_MAP = {
    # ── English (Premier League) ──────────────────────────────────────────────
    "afc bournemouth":               "bournemouth",
    "brighton & hove albion":        "brighton",
    "newcastle united":              "newcastle",
    "tottenham hotspur":             "tottenham",
    "west ham united":               "west ham",
    "wolverhampton wanderers":       "wolves",
    "ipswich town":                  "ipswich",
    "leicester city":                "leicester",
    # ── Spanish (La Liga) ─────────────────────────────────────────────────────
    "atletico de madrid":            "atletico madrid",   # .es locale
    "atletico madrid":               "atletico madrid",   # already correct, safety
    "celta de vigo":                 "celta vigo",
    "ca osasuna":                    "osasuna",
    "cd leganes":                    "leganes",
    "real betis balompie":           "real betis",
    "espanyol barcelona":            "espanyol",          # RCD stripped → espanyol barcelona
    "athletic bilbao":               "athletic club",     # TM.us uses "Athletic Bilbao"
    # ── German (Bundesliga) ───────────────────────────────────────────────────
    "bayer 04 leverkusen":           "bayer leverkusen",
    "1899 hoffenheim":               "hoffenheim",        # TSG stripped → 1899 hoffenheim
    "borussia monchengladbach":      "borussia m.gladbach",
    "heidenheim 1846":               "heidenheim",
    "sv werder bremen":              "werder bremen",
    # legacy .es names (kept for old CSVs)
    "augsburgo":                     "augsburg",
    "friburgo":                      "freiburg",
    "wolfsburgo":                    "wolfsburg",
    # ── Italian (Serie A) ─────────────────────────────────────────────────────
    "acf fiorentina":                "fiorentina",        # ACF not stripped → acf fiorentina
    "atalanta bc":                   "atalanta",          # .us name
    "bologna 1909":                  "bologna",           # FC stripped → bologna 1909
    "cagliari calcio":               "cagliari",
    "genoa cfc":                     "genoa",             # CFC not stripped → genoa cfc
    "hellas verona":                 "verona",
    "inter milan":                   "inter",             # .us name
    "ssc napoli":                    "napoli",
    "udinese calcio":                "udinese",
    "us lecce":                      "lecce",
    "parma calcio 1909":             "parma calcio 1913", # TM may use 1909 or 1913
    # legacy .es names
    "atalanta de bergamo":           "atalanta",
    "bolonia":                       "bologna",
    "genova":                        "genoa",
    "inter de milan":                "inter",
    "juventus de turin":             "juventus",
    "ssc napoles":                   "napoli",
    # ── French (Ligue 1) ──────────────────────────────────────────────────────
    "aj auxerre":                    "auxerre",
    "angers sco":                    "angers",
    "losc lille":                    "lille",
    "montpellier hsc":               "montpellier",
    "ogc nice":                      "nice",
    "olympique lyon":                "lyon",
    "olympique marseille":           "marseille",
    "strasbourg alsace":             "strasbourg",        # RC stripped → strasbourg alsace
    "stade brestois 29":             "brest",
    "stade reims":                   "reims",
    "stade rennais":                 "rennes",
    "como 1907":                     "como",
    # legacy .es names
    "ogc niza":                      "nice",
    "olympique de lyon":             "lyon",
    "olympique de marsella":         "marseille",
    "racing club de estrasburgo":    "strasbourg",
    "stade de reims":                "reims",
    "eintracht francfort":           "eintracht frankfurt",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _strip_accents(s: str) -> str:
    return (
        unicodedata.normalize("NFD", s)
        .encode("ascii", "ignore")
        .decode()
    )


def norm_team(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _strip_accents(s)
    s = re.sub(
        r"\b(FC|CF|AC|SC|AS|SS|SL|RC|SD|UD|RCD|VfL|VfB|TSG|RB|1\.\s*FC|1\.\s*FSV)\b",
        "", s, flags=re.I,
    )
    return re.sub(r"\s+", " ", s).strip().lower()


def norm_player(s: str) -> str:
    """Token-sorted norm — handles name-order differences (Vinicius Junior = Junior Vinicius)."""
    if not isinstance(s, str):
        return ""
    s = _strip_accents(s)
    tokens = sorted(re.sub(r"[^a-zA-Z0-9 ]", "", s).lower().split())
    return " ".join(tokens)


def norm_player_plain(s: str) -> str:
    """Unsorted norm — preserves word order for prefix/containment checks."""
    if not isinstance(s, str):
        return ""
    s = _strip_accents(s)
    return re.sub(r"[^a-zA-Z0-9 ]", "", s).lower().strip()


# Position word sets used for tiebreaking ambiguous single-name matches.
# Both TM and XT use English position names so a word-overlap check works well.
_POS_STOPWORDS = {"the", "a", "an", "left", "right", "attacking", "defensive",
                  "second", "wing", "back"}

def pos_key(pos_str) -> set:
    """Core position keywords after stripping directional/generic words."""
    if not pos_str or (isinstance(pos_str, float)):
        return set()
    words = re.sub(r"[-/]", " ", str(pos_str)).lower().split()
    return {w for w in words if w not in _POS_STOPWORDS and len(w) > 2}


def pos_compatible(mv_pos, xt_pos) -> bool:
    """True when positions share at least one key word, or either is unknown."""
    mk, xk = pos_key(mv_pos), pos_key(xt_pos)
    if not mk or not xk:
        return True   # unknown → don't disqualify
    return bool(mk & xk)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

print("Loading files...")
mv  = pd.read_csv(MV_CSV, encoding="utf-8-sig")
xt  = pd.read_parquet(XT_PAR)

# Keep only 24/25 top-5 rows in the XT slice used for matching
xt25 = xt[
    (xt["season"] == SEASON_XT) &
    (xt["competition"].isin(TOP5_LEAGUES_XT))
].copy()

print(f"  MV rows:      {len(mv):,}")
print(f"  XT total:     {len(xt):,}")
print(f"  XT 24/25 top5:{len(xt25):,}")

# ── NORMALIZE TEAM NAMES ──────────────────────────────────────────────────────

mv["_team_norm"] = mv["team"].apply(norm_team).replace(TEAM_MAP)
xt25["_team_norm"] = xt25["team_name"].apply(norm_team)

# ── NORMALIZE PLAYER NAMES ────────────────────────────────────────────────────

name_col = "name_clean" if "name_clean" in xt25.columns else "name"
mv["_player_norm"]        = mv["player"].apply(norm_player)
mv["_player_plain"]       = mv["player"].apply(norm_player_plain)
xt25["_player_norm"]      = xt25[name_col].apply(norm_player)
xt25["_player_plain"]     = xt25[name_col].apply(norm_player_plain)

# ── BUILD LOOKUPS ─────────────────────────────────────────────────────────────

# One row per (team, player) — keep highest xT when a player appears in
# multiple competitions (e.g. both Bundesliga and Champions League).
xt25_dedup = (
    xt25.sort_values("total_xt", ascending=False)
    .drop_duplicates(subset=["_team_norm", "_player_norm"])
)

# Fast exact lookup: (team_norm, player_norm) → xt row index
lookup_exact = {
    (r["_team_norm"], r["_player_norm"]): r.name
    for _, r in xt25_dedup.iterrows()
}

# Per-team candidate list: (sorted_norm, plain_norm, position, xt_index)
team_players = {}
pos_col = "primary_position_full" if "primary_position_full" in xt25_dedup.columns else "position_full"
for _, r in xt25_dedup.iterrows():
    team_players.setdefault(r["_team_norm"], []).append(
        (r["_player_norm"], r["_player_plain"], r.get(pos_col, ""), r.name)
    )

# Global player list for cross-team transfer fallback
# Only include players with a unique normalised name to avoid ambiguous matches
all_player_norms = [r["_player_norm"] for _, r in xt25_dedup.iterrows()]
all_player_idxs  = [r.name           for _, r in xt25_dedup.iterrows()]
from collections import Counter
_norm_counts = Counter(all_player_norms)
# map norm → xt index, but only for norms that are globally unique
lookup_global_unique = {
    norm: idx
    for norm, idx in zip(all_player_norms, all_player_idxs)
    if _norm_counts[norm] == 1
}

# ── MATCH MV ROWS → XT INDICES ────────────────────────────────────────────────

print("\nMatching players...")

mv2 = mv.copy()
mv2["_match_idx"] = -1

exact_hits = fuzzy_hits = prefix_hits = xteam_hits = 0
unmatched_mv_indices = []

for row_i, mv_row in mv.iterrows():
    t_norm   = mv_row["_team_norm"]
    p_norm   = mv_row["_player_norm"]    # token-sorted
    p_plain  = mv_row["_player_plain"]   # unsorted (preserves word order)
    mv_pos   = mv_row.get("position", "")

    # 1) Exact match — same team, token-sorted norm
    if (t_norm, p_norm) in lookup_exact:
        mv2.at[row_i, "_match_idx"] = lookup_exact[(t_norm, p_norm)]
        exact_hits += 1
        continue

    candidates = team_players.get(t_norm, [])   # (sorted, plain, pos, idx)
    cand_sorted = [c[0] for c in candidates]
    cand_plain  = [c[1] for c in candidates]

    # 2) Fuzzy within same team.
    #    Try BOTH the token-sorted norm (handles order differences like
    #    "Vinicius Junior" vs "Junior Vinicius") AND the plain unsorted norm
    #    (handles spelling variants like "Emmanuel" vs "Emanuel" where
    #    token-sorting scrambles the comparison and collapses the ratio).
    if cand_sorted:
        matched_norm = None
        # 2a) sorted-norm fuzzy
        hits = difflib.get_close_matches(p_norm, cand_sorted, n=1, cutoff=FUZZY_THRESHOLD)
        if hits:
            matched_norm = hits[0]
            best_idx = next(idx for (sn, pl, pos, idx) in candidates if sn == matched_norm)
        # 2b) plain-norm fuzzy (fallback — catches spelling errors scrambled by sort)
        if matched_norm is None:
            hits = difflib.get_close_matches(p_plain, cand_plain, n=1, cutoff=FUZZY_THRESHOLD)
            if hits:
                best_idx = next(idx for (sn, pl, pos, idx) in candidates if pl == hits[0])
                matched_norm = hits[0]
        if matched_norm is not None:
            mv2.at[row_i, "_match_idx"] = best_idx
            fuzzy_hits += 1
            continue

    # 3) Single-name containment matching using UNSORTED (plain) norms.
    #    Handles all combinations where one side uses a partial name:
    #      - MV="Gabriel"       XT="Gabriel Magalhaes"  (MV prefix of XT)
    #      - MV="Beraldo"       XT="Lucas Beraldo"       (MV last-name in XT)
    #      - MV="Kepa Arrizabalaga"  XT="Kepa"           (XT prefix of MV)
    #      - MV="Lucas Beraldo"      XT="Beraldo"         (XT last-name in MV)
    #    When multiple candidates share the token, position breaks the tie.
    if len(p_plain) >= 4 and candidates:
        p_words    = set(p_plain.split())
        p_is_short = len(p_plain.split()) == 1   # MV is a single token

        # 3a) MV name appears in XT name (prefix OR any contained word)
        fwd = [(sn, pl, pos, idx) for (sn, pl, pos, idx) in candidates
               if pl.startswith(p_plain + " ")           # "gabriel" → "gabriel magalhaes"
               or pl == p_plain                           # exact plain
               or (p_is_short and p_plain in pl.split())]  # "beraldo" → "lucas beraldo"

        # 3b) XT name (single token) appears in MV name (prefix OR any word)
        rev = [(sn, pl, pos, idx) for (sn, pl, pos, idx) in candidates
               if len(pl.split()) == 1 and (
                   p_plain.startswith(pl + " ")   # "kepa arrizabalaga" starts with "kepa"
                   or pl in p_words)]              # "lucas beraldo" contains "beraldo"

        for pool in (fwd, rev):
            if not pool:
                continue
            if len(pool) == 1:
                mv2.at[row_i, "_match_idx"] = pool[0][3]
                prefix_hits += 1
                break
            # Ambiguous — use position to pick the best match
            pos_ok = [(sn, pl, pos, idx) for (sn, pl, pos, idx) in pool
                      if pos_compatible(mv_pos, pos)]
            if len(pos_ok) == 1:
                mv2.at[row_i, "_match_idx"] = pos_ok[0][3]
                prefix_hits += 1
                break
        else:
            # Neither pool resolved — fall through to cross-team
            pass

        if mv2.at[row_i, "_match_idx"] >= 0:
            continue

    # 4) Cross-team fallback for mid-season transfers:
    #    Player is globally unique in XT — accept regardless of team.
    if p_norm in lookup_global_unique:
        mv2.at[row_i, "_match_idx"] = lookup_global_unique[p_norm]
        xteam_hits += 1
        continue
    if len(p_norm) >= 6:
        close = difflib.get_close_matches(
            p_norm, list(lookup_global_unique.keys()), n=1, cutoff=FUZZY_THRESHOLD_XTEAM
        )
        if close:
            mv2.at[row_i, "_match_idx"] = lookup_global_unique[close[0]]
            xteam_hits += 1
            continue

    unmatched_mv_indices.append(row_i)

print(f"  Exact (same team):      {exact_hits:,}")
print(f"  Fuzzy (same team):      {fuzzy_hits:,}")
print(f"  Prefix (single-name):   {prefix_hits:,}")
print(f"  Cross-team (transfers): {xteam_hits:,}")
print(f"  Unmatched:              {len(unmatched_mv_indices):,}")

matched_mv = mv2[mv2["_match_idx"] >= 0].copy()

# Group by xt index → take max market value
idx_to_mv = (
    matched_mv.groupby("_match_idx")["market_value"]
    .max()
    .to_dict()
)

# Apply to the full xt dataframe (not just xt25 slice)
xt["market_value_eur_m"] = xt.index.map(idx_to_mv)

covered = xt["market_value_eur_m"].notna().sum()
print(f"\n  XT rows with market value attached: {covered:,} / {len(xt):,}")

# ── SAVE ──────────────────────────────────────────────────────────────────────

xt.to_parquet(OUT_PAR, index=True)
xt.to_csv(OUT_CSV, index=False)
print(f"  Saved -> {OUT_PAR}")
print(f"  Saved -> {OUT_CSV}")

# ── UNMATCHED HIGH-VALUE PLAYERS ──────────────────────────────────────────────

unmatched_mv = mv.loc[unmatched_mv_indices].copy()

# Only show players who have a market value and were in a team that exists in XT
# (i.e., not missing due to team mismatch — those are less interesting)
xt25_team_norms = set(xt25["_team_norm"].unique())
unmatched_known_team = unmatched_mv[
    unmatched_mv["_team_norm"].isin(xt25_team_norms) &
    unmatched_mv["market_value"].notna()
].copy()

unmatched_all_with_value = unmatched_mv[
    unmatched_mv["market_value"].notna()
].copy()

print("\n" + "=" * 70)
print("MOST VALUABLE PLAYERS WITHOUT AN XT MATCH")
print("=" * 70)

top_unmatched = (
    unmatched_all_with_value
    .sort_values("market_value", ascending=False)
    .drop_duplicates(subset=["player", "team"])
    [["player", "team", "league", "market_value", "market_value_raw", "_team_norm"]]
    .head(40)
)

# Flag whether the team was found in XT at all
top_unmatched["team_in_xt"] = top_unmatched["_team_norm"].isin(xt25_team_norms)

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_colwidth", 30)
pd.set_option("display.width", 120)
output = top_unmatched.drop(columns=["_team_norm"]).to_string(index=False)
print(output.encode("ascii", "replace").decode())

print(f"\nTotal unmatched with value: {len(unmatched_all_with_value)}")
print(f"Of which team WAS in XT:    {top_unmatched['team_in_xt'].sum()} (in top 40)")
