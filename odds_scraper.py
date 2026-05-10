"""
odds_scraper.py
Scrapes historical UFC fight closing odds from bestfightodds.com.

Strategy:
- Walks the archive page backwards using ?before=<event_id> pagination
- Collects UFC event slugs (e.g. /events/ufc-perth-4079)
- For each event, parses fighter moneyline odds, averages across bookmakers,
  removes vig, stores implied win probabilities in the `odds` table
- Incremental by default: skips already-scraped slugs

Usage:
    python odds_scraper.py              # incremental (new events only)
    python odds_scraper.py --full       # rescrape everything
    python odds_scraper.py --test ufc-perth-4079   # debug single event
"""

import re
import time
import sqlite3
import logging
import argparse
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
DB_PATH  = DATA_DIR / "ufc.db"
BASE_URL = "https://www.bestfightodds.com"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
SLEEP = 1.5   # seconds between requests — be polite to the server


# ─────────────────────────────────────────────
# ODDS MATH
# ─────────────────────────────────────────────

def american_to_prob(odds: float) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

def remove_vig(p1: float, p2: float) -> tuple[float, float]:
    total = p1 + p2
    return (p1 / total, p2 / total) if total > 0 else (0.5, 0.5)

def parse_odds_value(text: str) -> float | None:
    text = re.sub(r"[▼▲,\s]", "", text)
    try:
        v = float(text)
        return v if -5000 < v < 5000 and v != 0 else None
    except ValueError:
        return None


# ─────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def fetch(url: str) -> BeautifulSoup | None:
    try:
        r = SESSION.get(url, timeout=20, allow_redirects=True)
        if r.status_code in (404, 410):
            return None
        # Detect redirect away from an event page back to homepage/archive
        # (only check this for event URLs, not the archive page itself)
        if "/events/" in url and r.url.rstrip("/") in (BASE_URL, BASE_URL + "/archive"):
            log.debug(f"Redirected away from {url} — skipping")
            return None
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except Exception as e:
        log.warning(f"Fetch failed {url}: {e}")
        return None


# ─────────────────────────────────────────────
# SLUG DISCOVERY
# ─────────────────────────────────────────────

_UFC_EVENT_PATTERN = re.compile(r"^/events/ufc[a-z0-9\-]+-\d+$")
_EVENT_PATTERN     = re.compile(r"^/events/[a-z0-9\-]+-\d+$")
_FIGHTER_PATTERN   = re.compile(r"^/fighters/[A-Za-z0-9\-]+-\d+$")


def _extract_ufc_slugs(soup: BeautifulSoup) -> set[str]:
    """Pull all UFC event slugs from a parsed page."""
    return {
        a["href"].removeprefix("/events/")
        for a in soup.find_all("a", href=_UFC_EVENT_PATTERN)
    }


def _extract_fighter_hrefs(soup: BeautifulSoup) -> set[str]:
    """Pull all fighter profile hrefs from a parsed page."""
    return {a["href"] for a in soup.find_all("a", href=_FIGHTER_PATTERN)}


def collect_all_ufc_slugs(already_scraped: set[str] | None = None) -> list[str]:
    """
    Discover every UFC event slug on bestfightodds.com.

    Strategy:
    1. Seed from the archive page (most-recent ~20 events, typically 2-4 UFC).
    2. For each UFC event page fetched, collect all fighter profile links.
    3. For each fighter profile not yet visited, fetch it and collect UFC event slugs.
    4. Repeat until the fighter queue is exhausted — at that point we'll have
       seen every UFC event that any scraped fighter appeared in.

    Returns slugs ordered newest-first (by BFO numeric ID, descending).
    """
    already_scraped = already_scraped or set()

    # -- Step 1: seed from archive ----------------------------------------
    archive_soup = fetch(f"{BASE_URL}/archive")
    seed_slugs: set[str] = set()
    if archive_soup:
        seed_slugs = _extract_ufc_slugs(archive_soup)
    log.info(f"Archive seed: {len(seed_slugs)} UFC event slugs")

    all_ufc_slugs: set[str] = set(seed_slugs)
    fighter_queue: set[str] = set()   # fighter hrefs to visit
    visited_fighters: set[str] = set()

    # -- Step 2 & 3: BFS through event pages → fighter profiles → events ---
    events_to_scrape_for_fighters = list(seed_slugs)

    while events_to_scrape_for_fighters:
        slug = events_to_scrape_for_fighters.pop()
        soup = fetch(f"{BASE_URL}/events/{slug}")
        time.sleep(SLEEP)
        if soup is None:
            continue
        fighter_queue.update(_extract_fighter_hrefs(soup))

    log.info(f"Collected {len(fighter_queue)} fighter profiles to visit")

    for fighter_href in fighter_queue:
        if fighter_href in visited_fighters:
            continue
        visited_fighters.add(fighter_href)

        fsoup = fetch(f"{BASE_URL}{fighter_href}")
        time.sleep(SLEEP)
        if fsoup is None:
            continue

        new_slugs = _extract_ufc_slugs(fsoup) - all_ufc_slugs
        if new_slugs:
            all_ufc_slugs.update(new_slugs)
            log.debug(f"  {fighter_href}: +{len(new_slugs)} new UFC events")

    log.info(f"Total UFC event slugs discovered: {len(all_ufc_slugs)}")

    # Sort newest-first by numeric ID
    def _event_id(slug: str) -> int:
        m = re.search(r"-(\d+)$", slug)
        return int(m.group(1)) if m else 0

    return sorted(all_ufc_slugs, key=_event_id, reverse=True)


# ─────────────────────────────────────────────
# EVENT PAGE PARSER
# ─────────────────────────────────────────────

def parse_event(slug: str) -> dict | None:
    """
    Scrape one event page. Returns dict or None on failure.
    """
    url = f"{BASE_URL}/events/{slug}"
    soup = fetch(url)
    if soup is None:
        return None

    # Event name from h1 or title
    h1 = soup.find("h1")
    event_name = h1.get_text(strip=True) if h1 else slug

    fights = _extract_fights(soup)
    if not fights:
        log.debug(f"No fights found for {slug}")
        return None

    return {"slug": slug, "event_name": event_name, "fights": fights}


def _excluded_col_indices(soup: BeautifulSoup) -> frozenset[int]:
    """
    Parse the table header row to find column indices for prediction markets
    (Kalshi, Polymarket) that stay open during and after the fight.
    These must be excluded from the sportsbook average to avoid data leakage.
    Returns a frozenset of 0-indexed column positions (0 = fighter name col).
    Falls back to empty set for old events where these books didn't exist.
    """
    EXCLUDED_BOOKS = frozenset(["kalshi", "polymarket"])
    KNOWN_SPORTSBOOKS = frozenset(["fanduel", "caesars", "betmgm", "draftkings",
                                    "betrivers", "betway", "unibet"])
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 5:
            continue
        texts = [c.get_text(strip=True).lower() for c in cells]
        # Confirm this is the book header row by spotting a known sportsbook
        if any(any(sb in t for sb in KNOWN_SPORTSBOOKS) for t in texts):
            return frozenset(
                i for i, t in enumerate(texts)
                if any(excl in t for excl in EXCLUDED_BOOKS)
            )
    return frozenset()


def _extract_fights(soup: BeautifulSoup) -> list[dict]:
    """
    Extract fight matchups and moneyline odds.

    bestfightodds layout (as of 2024-2026):
      The page contains two identical table sections. The first section
      (~rows 1-1635) is navigation-only: fighter name rows have exactly
      1 <td> cell and no odds. The second section contains the actual
      bookmaker odds — fighter rows there have 8+ cells with values like
      "+140 ?" or "-191 ?".

      We skip any fighter row with fewer than MIN_ODDS_CELLS cells, which
      filters out all navigation rows cleanly.

      Prediction markets (Kalshi, Polymarket) are excluded from the average
      because they remain open during and after the fight, which would
      introduce data leakage into pre-fight odds.
    """
    MIN_ODDS_CELLS = 5  # navigation rows have 1 cell; odds rows have 8+

    excluded_cols = _excluded_col_indices(soup)

    fights = []
    pending: dict | None = None

    for row in soup.find_all("tr"):
        fighter_link = row.find("a", href=re.compile(r"^/fighters/"))
        if not fighter_link:
            continue

        cells = row.find_all("td")
        # Skip navigation/header rows that have no odds columns
        if len(cells) < MIN_ODDS_CELLS:
            continue

        name = fighter_link.get_text(strip=True)
        # Collect numeric moneyline values from sportsbooks only.
        # Skip prediction market columns (Kalshi, Polymarket) and props.
        # |v| >= 100 filters out round totals, percentages, and prop values.
        odds_vals = []
        for i, td in enumerate(cells):
            if i == 0 or i in excluded_cols:
                continue
            v = parse_odds_value(td.get_text(strip=True))
            if v is not None and abs(v) >= 100:
                odds_vals.append(v)

        if pending is None:
            pending = {"name": name, "odds": odds_vals}
        else:
            fight = _build_fight(pending, {"name": name, "odds": odds_vals})
            if fight:
                fights.append(fight)
            pending = None

    # Deduplicate: keep the last entry for each fighter pair since moneyline
    # rows appear after any prop/navigation rows for the same matchup.
    deduped: dict[tuple, dict] = {}
    for f in fights:
        key = (f["f1_name"], f["f2_name"])
        # Always overwrite — last entry with real odds wins
        if f["f1_implied_prob"] is not None or key not in deduped:
            deduped[key] = f
    fights = list(deduped.values())

    # Fallback: name-pairs only (no odds columns parseable)
    if not fights:
        links = soup.find_all("a", href=re.compile(r"^/fighters/"))
        names = list(dict.fromkeys(a.get_text(strip=True) for a in links))
        for i in range(0, len(names) - 1, 2):
            fights.append({
                "f1_name": names[i], "f2_name": names[i + 1],
                "f1_avg_odds": None, "f2_avg_odds": None,
                "f1_implied_prob": None, "f2_implied_prob": None,
            })

    return fights


def _build_fight(f1: dict, f2: dict) -> dict | None:
    f1_odds = f1["odds"]
    f2_odds = f2["odds"]

    if f1_odds and f2_odds:
        f1_avg = sum(f1_odds) / len(f1_odds)
        f2_avg = sum(f2_odds) / len(f2_odds)
        # Average implied probabilities across books, not raw moneylines.
        # Averaging moneylines breaks when books straddle even money
        # (e.g., -115 and +110 average to -2.5 → 2.4% implied, not ~50%).
        p1_raw = sum(american_to_prob(o) for o in f1_odds) / len(f1_odds)
        p2_raw = sum(american_to_prob(o) for o in f2_odds) / len(f2_odds)
        p1, p2 = remove_vig(p1_raw, p2_raw)
    else:
        f1_avg = f2_avg = None
        p1 = p2 = None

    return {
        "f1_name": f1["name"], "f2_name": f2["name"],
        "f1_avg_odds": round(f1_avg, 1) if f1_avg is not None else None,
        "f2_avg_odds": round(f2_avg, 1) if f2_avg is not None else None,
        "f1_implied_prob": round(p1, 4) if p1 is not None else None,
        "f2_implied_prob": round(p2, 4) if p2 is not None else None,
    }


# ─────────────────────────────────────────────
# NAME MATCHING
# ─────────────────────────────────────────────

def load_db_names(db_path: Path = DB_PATH) -> dict[str, str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT first_name, last_name FROM fighters").fetchall()
    conn.close()
    result = {}
    for fn, ln in rows:
        full = f"{fn or ''} {ln or ''}".strip()
        result[_norm(full)] = full
    return result

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def match_name(bfo_name: str, db_names: dict[str, str], threshold: float = 0.82) -> str | None:
    n = _norm(bfo_name)
    if n in db_names:
        return db_names[n]
    best, best_name = 0.0, None
    for k, v in db_names.items():
        s = SequenceMatcher(None, n, k).ratio()
        if s > best:
            best, best_name = s, v
    return best_name if best >= threshold else None


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds (
            bfo_slug        TEXT,
            bfo_event_name  TEXT,
            bfo_f1_name     TEXT,
            bfo_f2_name     TEXT,
            db_f1_name      TEXT,
            db_f2_name      TEXT,
            f1_avg_odds     REAL,
            f2_avg_odds     REAL,
            f1_implied_prob REAL,
            f2_implied_prob REAL,
            scraped_at      TEXT,
            PRIMARY KEY (bfo_slug, bfo_f1_name, bfo_f2_name)
        )
    """)
    conn.commit()
    conn.close()

def get_scraped_slugs(db_path: Path = DB_PATH) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT bfo_slug FROM odds").fetchall()
    conn.close()
    return {r[0] for r in rows}

def save_event(event: dict, db_names: dict, db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    now = str(datetime.now())
    saved = 0
    unmatched = []
    for f in event["fights"]:
        db_f1 = match_name(f["f1_name"], db_names)
        db_f2 = match_name(f["f2_name"], db_names)
        for name, matched in [(f["f1_name"], db_f1), (f["f2_name"], db_f2)]:
            if not matched:
                unmatched.append(name)
        conn.execute("""
            INSERT OR REPLACE INTO odds
            (bfo_slug, bfo_event_name, bfo_f1_name, bfo_f2_name,
             db_f1_name, db_f2_name,
             f1_avg_odds, f2_avg_odds, f1_implied_prob, f2_implied_prob,
             scraped_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            event["slug"], event["event_name"],
            f["f1_name"], f["f2_name"],
            db_f1, db_f2,
            f["f1_avg_odds"], f["f2_avg_odds"],
            f["f1_implied_prob"], f["f2_implied_prob"],
            now,
        ))
        saved += 1
    conn.commit()
    conn.close()
    return saved, unmatched


# ─────────────────────────────────────────────
# ODDS → FEATURES MERGE
# ─────────────────────────────────────────────

def merge_odds_into_features(db_path: Path = DB_PATH):
    """
    Join closing-line implied probabilities from the `odds` table into the
    `features` table as three new columns:
        f1_closing_prob    — implied win prob for features.fighter_1
        f2_closing_prob    — implied win prob for features.fighter_2
        diff_closing_prob  — f1 - f2  (the diff-encoded signal for the model)

    Handles fighter-order mismatch: odds.db_f1_name may correspond to either
    fighter_1 or fighter_2 in the features table (because features applies a
    random 50% fighter-order swap for label balancing).

    Rows with no matching odds entry get NULL (NaN) — the model pipeline's
    SimpleImputer fills these with median (~0) which encodes "no information".
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # Ensure columns exist
    existing = {row[1] for row in conn.execute("PRAGMA table_info(features)").fetchall()}
    for col in ["f1_closing_prob", "f2_closing_prob", "diff_closing_prob"]:
        if col not in existing:
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} REAL")

    # Build a lookup: (name_a, name_b) → (prob_a, prob_b)
    # where name_a is the first fighter in the BFO row
    rows = conn.execute(
        "SELECT db_f1_name, db_f2_name, f1_implied_prob, f2_implied_prob FROM odds "
        "WHERE f1_implied_prob IS NOT NULL AND db_f1_name IS NOT NULL AND db_f2_name IS NOT NULL"
    ).fetchall()

    # Build two-way lookup keyed by frozenset so order doesn't matter
    odds_lookup: dict[frozenset, tuple[str, str, float, float]] = {}
    for db_f1, db_f2, p1, p2 in rows:
        key = frozenset([db_f1, db_f2])
        odds_lookup[key] = (db_f1, db_f2, p1, p2)

    # Read features
    features = conn.execute(
        "SELECT fight_url, fighter_1, fighter_2 FROM features"
    ).fetchall()

    updated = 0
    for fight_url, f1, f2 in features:
        key = frozenset([f1, f2])
        if key not in odds_lookup:
            continue
        bfo_f1, bfo_f2, p1, p2 = odds_lookup[key]
        # Align probabilities to features table order
        if bfo_f1 == f1:
            prob_f1, prob_f2 = p1, p2
        else:
            prob_f1, prob_f2 = p2, p1
        conn.execute(
            "UPDATE features SET f1_closing_prob=?, f2_closing_prob=?, diff_closing_prob=? "
            "WHERE fight_url=?",
            (prob_f1, prob_f2, round(prob_f1 - prob_f2, 4), fight_url)
        )
        updated += 1

    conn.commit()
    conn.close()
    log.info(f"Odds merged into features: {updated} fights updated out of {len(features)} total")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(full: bool = False, db_path: Path = DB_PATH):
    init_db(db_path)
    db_names = load_db_names(db_path)
    scraped  = set() if full else get_scraped_slugs(db_path)

    log.info(f"Discovering UFC event slugs... Already scraped: {len(scraped)} events.")

    # Phase 1: discover all UFC event slugs via archive + fighter profile BFS
    all_slugs = collect_all_ufc_slugs(already_scraped=scraped)
    new_slugs = [s for s in all_slugs if s not in scraped]
    log.info(f"Found {len(all_slugs)} total UFC events, {len(new_slugs)} new to scrape.")

    total_saved = 0

    # Phase 2: scrape each new event
    for slug in new_slugs:
        event = parse_event(slug)
        time.sleep(SLEEP)

        if event is None:
            log.warning(f"Could not parse {slug}")
            continue

        saved, unmatched = save_event(event, db_names, db_path)
        total_saved += saved
        log.info(
            f"  {event['event_name']} ({slug}): "
            f"{saved} fights, {len(unmatched)} unmatched"
        )
        if unmatched:
            log.debug(f"    Unmatched names: {unmatched}")
        scraped.add(slug)

    log.info(f"Pipeline complete. Total fight rows saved: {total_saved}")

    # Phase 3: merge closing-line probabilities into the features table
    merge_odds_into_features(db_path)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Rescrape all events")
    parser.add_argument("--test", type=str, metavar="SLUG",
                        help="Debug-parse a single event slug (e.g. ufc-perth-4079)")
    args = parser.parse_args()

    if args.test:
        logging.getLogger().setLevel(logging.DEBUG)
        slug = args.test
        print(f"\nFetching /events/{slug} ...")
        event = parse_event(slug)
        if event is None:
            print("Failed to parse event.")
            # Print raw page text for debugging
            soup = fetch(f"{BASE_URL}/events/{slug}")
            if soup:
                txt = soup.get_text(" ", strip=True)
                print(f"Page title: {soup.title.get_text() if soup.title else 'n/a'}")
                print(f"H1 tags: {[h.get_text() for h in soup.find_all('h1')]}")
                print(f"Fighter links: {[a['href'] for a in soup.find_all('a', href=re.compile(r'/fighters/'))][:10]}")
                print(f"\nFirst 1500 chars of text:\n{txt[:1500]}")
        else:
            print(f"\nEvent: {event['event_name']}")
            print(f"Fights ({len(event['fights'])}):")
            for f in event["fights"]:
                print(
                    f"  {f['f1_name']:<25} ({f.get('f1_avg_odds') or '?':>6})  "
                    f"vs  {f['f2_name']:<25} ({f.get('f2_avg_odds') or '?':>6})  "
                    f"implied: {f.get('f1_implied_prob') or '?'} / {f.get('f2_implied_prob') or '?'}"
                )
    else:
        run_pipeline(full=args.full)
