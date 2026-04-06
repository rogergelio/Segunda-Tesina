# market_values_scraper.py
# Scrapes player market values from Transfermarkt squad pages
# for the top 5 European leagues in the 2024/25 season.
#
# Flow:
#   1. For each league, fetch the league startseite to collect team URLs.
#   2. For each team, fetch its /kader page and parse the squad table.
#   3. Save everything to market_values_2024_25.csv
#
# Squad table column layout (9 tds per row, confirmed from live page):
#   td[0]  shirt #          (zentriert rueckennummer)
#   td[1]  photo cell       (posrela — contains an inline-table)
#   td[2]  photo rowspan    (extra td emitted by BeautifulSoup for the rowspan img)
#   td[3]  player name      (hauptlink) ← a[href]
#   td[4]  position         (plain td, no special class)
#   td[5]  age              (zentriert, plain integer text)
#   td[6]  nationality      (zentriert, one or more <img class="flaggenrahmen">)
#   td[7]  current club     (zentriert, club badge img/link)
#   td[8]  market value     (rechts hauptlink) ← a href
#
# requirements: pip install requests beautifulsoup4 pandas lxml tqdm

import re
import time
from typing import Optional
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE    = "https://www.transfermarkt.us"
SEASON  = 2024   # 2024 = 2024/25

LEAGUES = [
    {"name": "Premier League", "path": "premier-league", "code": "GB1"},
    {"name": "La Liga",        "path": "laliga",         "code": "ES1"},
    {"name": "Bundesliga",     "path": "bundesliga",     "code": "L1"},
    {"name": "Serie A",        "path": "serie-a",        "code": "IT1"},
    {"name": "Ligue 1",        "path": "ligue-1",        "code": "FR1"},
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

DELAY = 1.5   # seconds between requests — be polite to the server

# ── HELPERS ───────────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update(HEADERS)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def get_soup(url: str) -> BeautifulSoup:
    time.sleep(DELAY)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def parse_market_value(raw: str) -> Optional[float]:
    """Convert '€120.00m' / '€850k' / '-' to a float (millions EUR)."""
    raw = clean(raw).replace("\xa0", "").replace(",", ".")
    if not raw or raw in ("-", "?"):
        return None
    # Match patterns like: €20.00m  £850k  $1.20m  950 Th.  etc.
    m = re.search(r"([\d.]+)\s*(m|k|th\.?)?", raw, re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit  = (m.group(2) or "").lower()
    if unit.startswith("k") or unit.startswith("th"):
        value /= 1_000
    return round(value, 4)


# ── STEP 1 — collect team URLs from the league page ──────────────────────────

def get_team_links(league: dict) -> list:
    """Returns a list of {"team_name", "team_id", "team_path"} dicts."""
    url = (
        f"{BASE}/{league['path']}/startseite/wettbewerb/{league['code']}"
        f"/plus/?saison_id={SEASON}"
    )
    soup = get_soup(url)

    teams = []
    seen  = set()

    for a in soup.select("table.items td.hauptlink a[href*='/startseite/verein/']"):
        href  = a["href"]
        title = clean(a.get_text())
        if not title or href in seen:
            continue
        seen.add(href)

        m = re.search(r"/startseite/verein/(\d+)", href)
        if not m:
            continue
        team_id   = m.group(1)
        team_path = href.split("/startseite/")[0].lstrip("/")

        teams.append({"team_name": title, "team_id": team_id, "team_path": team_path})

    return teams


# ── STEP 2 — scrape squad page for one team ───────────────────────────────────

def scrape_squad(team: dict, league_name: str) -> list:
    """Fetches /kader page and returns a list of player dicts."""
    url = (
        f"{BASE}/{team['team_path']}/kader/verein/{team['team_id']}"
        f"/plus/0/galerie/0?saison_id={SEASON}"
    )

    try:
        soup = get_soup(url)
    except Exception as e:
        tqdm.write(f"  !! Could not load {team['team_name']}: {e}")
        return []

    table = soup.select_one("table.items")
    if not table:
        tqdm.write(f"  !! No squad table for {team['team_name']}")
        return []

    rows = []
    for tr in table.select("tbody tr.odd, tbody tr.even"):
        tds = tr.find_all("td")
        if len(tds) < 9:
            continue

        # td[3] — player name (hauptlink)
        name_a = tds[3].find("a")
        player_name = clean(name_a.get_text()) if name_a else clean(tds[3].get_text())
        if not player_name:
            continue

        # td[4] — position (plain text)
        position = clean(tds[4].get_text())

        # td[5] — age (plain integer)
        age_text = clean(tds[5].get_text())
        try:
            age_val = int(age_text)
        except ValueError:
            age_val = None

        # td[6] — nationality: first flaggenrahmen img title
        nat_img = tds[6].find("img", class_="flaggenrahmen")
        nationality = clean(nat_img["title"]) if nat_img and nat_img.get("title") else ""

        # td[8] — market value (rechts hauptlink)
        mv_raw = clean(tds[8].get_text())
        mv_eur = parse_market_value(mv_raw)

        rows.append({
            "league":            league_name,
            "team":              team["team_name"],
            "player":            player_name,
            "position":          position,
            "age":               age_val,
            "nationality":       nationality,
            "market_value":      mv_eur,
            "market_value_raw":  mv_raw,
            "season":            f"{SEASON}/{str(SEASON + 1)[-2:]}",
        })

    return rows


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    all_rows = []

    for league in tqdm(LEAGUES, desc="Leagues"):
        tqdm.write(f"\n>> {league['name']}")

        teams = get_team_links(league)
        tqdm.write(f"   Found {len(teams)} teams")

        for team in tqdm(teams, desc=f"  {league['name']}", leave=False):
            rows = scrape_squad(team, league["name"])
            tqdm.write(f"   {team['team_name']:30s} — {len(rows)} players")
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"\nTotal players scraped: {len(df)}")
    print(df.head(10).to_string(index=False))

    out_path = "transfermarkt_scrapper/market_values_2024_25.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
