"""
ufc_predictor/scraper.py
Scrapes ufcstats.com for events, fights, and fighter stats.
Designed to be run incrementally - only scrapes new events each run.

Uses concurrent requests (10 threads, rate-limited to 10 req/s) to speed
up full scrapes from ~2h to ~20min.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "http://ufcstats.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UFC-Predictor-Bot/1.0)"}
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "ufc.db"

MAX_WORKERS = 10   # concurrent request threads
RATE_LIMIT  = 10   # max requests per second (shared across all threads)

# Map raw outcome text from ufcstats.com to clean values
OUTCOME_MAP = {"W": "win", "L": "loss", "D": "draw", "NC": "nc"}


# ─────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────

class _RateLimiter:
    """Thread-safe global rate limiter. Ensures no more than N requests/sec."""
    def __init__(self, max_per_second: float):
        self._min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            wait_time = self._last + self._min_interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self._last = time.time()

_limiter = _RateLimiter(RATE_LIMIT)


# ─────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────

@contextmanager
def get_db():
    """Context manager for SQLite connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read perf
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                event_url   TEXT PRIMARY KEY,
                event_name  TEXT NOT NULL,
                date        TEXT,
                location    TEXT
            );

            CREATE TABLE IF NOT EXISTS fighters (
                fighter_url TEXT PRIMARY KEY,
                first_name  TEXT,
                last_name   TEXT,
                nickname    TEXT,
                height      TEXT,
                weight      TEXT,
                reach       TEXT,
                stance      TEXT,
                wins        TEXT,
                losses      TEXT,
                draws       TEXT,
                dob         TEXT,
                slpm        REAL,
                str_acc     TEXT,
                sapm        REAL,
                str_def     TEXT,
                td_avg      REAL,
                td_acc      TEXT,
                td_def      TEXT,
                sub_avg     REAL,
                updated_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS fights (
                fight_url       TEXT PRIMARY KEY,
                event_url       TEXT REFERENCES events(event_url),
                event_name      TEXT,
                event_date      TEXT,
                event_location  TEXT,
                fighter_1       TEXT,
                fighter_2       TEXT,
                outcome         TEXT,
                weight_class    TEXT,
                method          TEXT,
                round           TEXT,
                time            TEXT,
                scheduled_rounds INTEGER,
                -- totals
                tot_kd_f1 TEXT, tot_kd_f2 TEXT,
                tot_sig_str_f1 TEXT, tot_sig_str_f2 TEXT,
                tot_sig_str_pct_f1 TEXT, tot_sig_str_pct_f2 TEXT,
                tot_total_str_f1 TEXT, tot_total_str_f2 TEXT,
                tot_td_f1 TEXT, tot_td_f2 TEXT,
                tot_td_pct_f1 TEXT, tot_td_pct_f2 TEXT,
                tot_sub_att_f1 TEXT, tot_sub_att_f2 TEXT,
                tot_rev_f1 TEXT, tot_rev_f2 TEXT,
                tot_ctrl_f1 TEXT, tot_ctrl_f2 TEXT,
                -- significant strike breakdown (target + position)
                sig_sig_str_f1 TEXT, sig_sig_str_f2 TEXT,
                sig_sig_str_pct_f1 TEXT, sig_sig_str_pct_f2 TEXT,
                sig_head_f1 TEXT, sig_head_f2 TEXT,
                sig_body_f1 TEXT, sig_body_f2 TEXT,
                sig_leg_f1 TEXT, sig_leg_f2 TEXT,
                sig_distance_f1 TEXT, sig_distance_f2 TEXT,
                sig_clinch_f1 TEXT, sig_clinch_f2 TEXT,
                sig_ground_f1 TEXT, sig_ground_f2 TEXT
            );

            CREATE TABLE IF NOT EXISTS features (
                fight_url       TEXT PRIMARY KEY REFERENCES fights(fight_url),
                event_date      TEXT,
                fighter_1       TEXT,
                fighter_2       TEXT,
                weight_class    TEXT,
                label           INTEGER,
                -- differential features (f1 - f2)
                diff_win_rate REAL, diff_finish_rate REAL,
                diff_avg_sig_str_landed REAL, diff_avg_sig_str_att REAL,
                diff_avg_td_landed REAL, diff_avg_td_att REAL,
                diff_avg_ctrl_secs REAL, diff_avg_kd REAL,
                diff_avg_sub_att REAL, diff_sig_str_acc REAL,
                diff_td_acc REAL, diff_avg_round_ended REAL,
                diff_fights_count REAL, diff_height REAL, diff_reach REAL,
                -- individual fighter stats
                f1_win_rate REAL, f2_win_rate REAL,
                f1_finish_rate REAL, f2_finish_rate REAL,
                f1_fights_count REAL, f2_fights_count REAL,
                f1_height REAL, f2_height REAL,
                f1_reach REAL, f2_reach REAL,
                f1_stance TEXT, f2_stance TEXT,
                f1_stance_Orthodox INTEGER, f1_stance_Southpaw INTEGER, f1_stance_Switch INTEGER,
                f2_stance_Orthodox INTEGER, f2_stance_Southpaw INTEGER, f2_stance_Switch INTEGER,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_fights_event_url  ON fights(event_url);
            CREATE INDEX IF NOT EXISTS idx_fights_fighter_1  ON fights(fighter_1);
            CREATE INDEX IF NOT EXISTS idx_fights_fighter_2  ON fights(fighter_2);
            CREATE INDEX IF NOT EXISTS idx_fights_event_date ON fights(event_date);
            CREATE INDEX IF NOT EXISTS idx_features_date     ON features(event_date);
        """)

        # Add new columns to fights table for existing databases
        fights_existing = {row[1] for row in conn.execute("PRAGMA table_info(fights)").fetchall()}
        fights_new_cols = [
            ("scheduled_rounds", "INTEGER"),
            ("sig_head_f1", "TEXT"), ("sig_head_f2", "TEXT"),
            ("sig_body_f1", "TEXT"), ("sig_body_f2", "TEXT"),
            ("sig_leg_f1", "TEXT"), ("sig_leg_f2", "TEXT"),
            ("sig_distance_f1", "TEXT"), ("sig_distance_f2", "TEXT"),
            ("sig_clinch_f1", "TEXT"), ("sig_clinch_f2", "TEXT"),
            ("sig_ground_f1", "TEXT"), ("sig_ground_f2", "TEXT"),
        ]
        for col, dtype in fights_new_cols:
            if col not in fights_existing:
                conn.execute(f"ALTER TABLE fights ADD COLUMN {col} {dtype}")
                log.info(f"Added column {col} to fights table")

        # Add new columns to fighters table for existing databases
        new_cols = [
            ("dob",     "TEXT"),
            ("slpm",    "REAL"),
            ("str_acc", "TEXT"),
            ("sapm",    "REAL"),
            ("str_def", "TEXT"),
            ("td_avg",  "REAL"),
            ("td_acc",  "TEXT"),
            ("td_def",  "TEXT"),
            ("sub_avg", "REAL"),
        ]
        existing = {row[1] for row in conn.execute("PRAGMA table_info(fighters)").fetchall()}
        for col, dtype in new_cols:
            if col not in existing:
                conn.execute(f"ALTER TABLE fighters ADD COLUMN {col} {dtype}")
                log.info(f"Added column {col} to fighters table")


def get_soup(url: str) -> BeautifulSoup:
    _limiter.wait()
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_event_date(date_str: str) -> str:
    """Parse 'March 07, 2026' → '2026-03-07'. Returns empty string on failure."""
    try:
        dt = datetime.strptime(date_str.strip(), "%B %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return ""


# ─────────────────────────────────────────────
# 1. EVENTS
# ─────────────────────────────────────────────

def scrape_all_events() -> pd.DataFrame:
    """Returns a DataFrame of all completed UFC events."""
    url = f"{BASE_URL}/statistics/events/completed?page=all"
    soup = get_soup(url)
    rows = soup.select("tr.b-statistics__table-row")

    events = []
    for row in rows:
        link_tag = row.select_one("a.b-link")
        if not link_tag:
            continue

        # Date is inside a <span class="b-statistics__date"> within column 0
        date_span = row.select_one("span.b-statistics__date")
        date_raw = date_span.get_text(strip=True) if date_span else ""
        date_iso = parse_event_date(date_raw)

        # Location is in the second column (index 1)
        tds = row.select("td.b-statistics__table-col")
        location_str = tds[1].get_text(strip=True) if len(tds) > 1 else ""

        events.append({
            "event_name": link_tag.get_text(strip=True),
            "event_url":  link_tag["href"],
            "date":       date_iso,
            "location":   location_str,
        })

    df = pd.DataFrame(events)
    log.info(f"Found {len(df)} events")
    return df


# ─────────────────────────────────────────────
# 2. FIGHTS PER EVENT
# ─────────────────────────────────────────────

def scrape_event_fights(event_url: str) -> list[dict]:
    """Returns list of fight dicts from a single event page."""
    soup = get_soup(event_url)
    rows = soup.select("tr.b-fight-details__table-row[data-link]")

    fights = []
    for row in rows:
        fight_url = row.get("data-link", "")
        cols = row.select("td.b-fight-details__table-col")
        if len(cols) < 10:
            continue

        def col_text(idx, li_idx=0):
            items = cols[idx].select("p.b-fight-details__table-text")
            return items[li_idx].get_text(strip=True) if len(items) > li_idx else ""

        raw_outcome = col_text(0)
        outcome = OUTCOME_MAP.get(raw_outcome, raw_outcome.lower())

        fights.append({
            "fight_url":       fight_url,
            "fighter_1":       col_text(1, 0),
            "fighter_2":       col_text(1, 1),
            "outcome":         outcome,
            "weight_class":    col_text(6),
            "method":          col_text(7),
            "round":           col_text(8),
            "time":            col_text(9),
        })

    return fights


# ─────────────────────────────────────────────
# 3. DETAILED FIGHT STATS
# ─────────────────────────────────────────────

def scrape_fight_stats(fight_url: str) -> dict:
    """Scrapes per-fight totals and per-round data for both fighters."""
    soup = get_soup(fight_url)
    result = {}

    # ── Scheduled rounds (from "Time format: 3 Rnd (5-5-5)") ──
    text_items = soup.select("i.b-fight-details__text-item")
    for item in text_items:
        text = item.get_text(strip=True)
        if text.startswith("Time format:"):
            # "Time format:3 Rnd (5-5-5)" → 3
            fmt = text.replace("Time format:", "").strip()
            try:
                result["scheduled_rounds"] = int(fmt.split()[0])
            except (ValueError, IndexError):
                pass
            break

    # ── Totals table (first stats table) ──
    tables = soup.select("table.b-fight-details__table")
    if not tables:
        return result

    def parse_stat_row(row, prefix, labels):
        cols = row.select("td.b-fight-details__table-col")
        stats = {}
        for i, label in enumerate(labels):
            if i + 1 >= len(cols):
                break
            items = cols[i + 1].select("p.b-fight-details__table-text")
            stats[f"{prefix}_{label}_f1"] = items[0].get_text(strip=True) if len(items) > 0 else ""
            stats[f"{prefix}_{label}_f2"] = items[1].get_text(strip=True) if len(items) > 1 else ""
        return stats

    TOTALS_LABELS = [
        "kd", "sig_str", "sig_str_pct", "total_str",
        "td", "td_pct", "sub_att", "rev", "ctrl"
    ]
    SIG_STRIKES_LABELS = [
        "sig_str", "sig_str_pct", "head", "body", "leg",
        "distance", "clinch", "ground"
    ]

    totals_rows = tables[0].select("tr.b-fight-details__table-row")
    for row in totals_rows[1:]:  # skip header
        result.update(parse_stat_row(row, "tot", TOTALS_LABELS))
        break  # only first data row for totals

    # ── Significant strikes breakdown (second table) ──
    if len(tables) > 1:
        sig_rows = tables[1].select("tr.b-fight-details__table-row")
        for row in sig_rows[1:]:
            result.update(parse_stat_row(row, "sig", SIG_STRIKES_LABELS))
            break

    return result


# ─────────────────────────────────────────────
# 4. FIGHTER PROFILES
# ─────────────────────────────────────────────

def scrape_fighter_profile(fighter_url: str) -> dict:
    """Scrape career stats from an individual fighter's profile page.

    Returns dict with keys: dob, slpm, str_acc, sapm, str_def,
    td_avg, td_acc, td_def, sub_avg. Values are strings/floats as appropriate.
    """
    try:
        soup = get_soup(fighter_url)
    except Exception as e:
        log.warning(f"Failed to fetch fighter profile {fighter_url}: {e}")
        return {}

    items = soup.select("li.b-list__box-list-item_type_block")
    if len(items) < 14:
        return {}

    def item_val(idx):
        """Extract the value text after the label from a list item."""
        text = items[idx].get_text(strip=True)
        # Format is "Label: Value" — split on the first colon
        parts = text.split(":", 1)
        return parts[1].strip() if len(parts) > 1 else ""

    def to_float(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    return {
        "dob":     item_val(4) or None,
        "slpm":    to_float(item_val(5)),
        "str_acc": item_val(6) or None,
        "sapm":    to_float(item_val(7)),
        "str_def": item_val(8) or None,
        "td_avg":  to_float(item_val(10)),
        "td_acc":  item_val(11) or None,
        "td_def":  item_val(12) or None,
        "sub_avg": to_float(item_val(13)),
    }


def _scrape_letter_page(char: str) -> list[dict]:
    """Scrape the fighter listing page for a single letter. Returns list of fighter dicts."""
    url = f"{BASE_URL}/statistics/fighters?char={char}&page=all"
    try:
        soup = get_soup(url)
    except Exception as e:
        log.warning(f"Failed to scrape fighters page '{char}': {e}")
        return []

    rows = soup.select("tr.b-statistics__table-row")
    fighters = []
    for row in rows:
        cols = row.select("td.b-statistics__table-col")
        if len(cols) < 10:
            continue
        link = row.select_one("a.b-link")
        fighter_url = link["href"] if link else ""

        fighters.append({
            "fighter_url":  fighter_url,
            "first_name":   cols[0].get_text(strip=True),
            "last_name":    cols[1].get_text(strip=True),
            "nickname":     cols[2].get_text(strip=True),
            "height":       cols[3].get_text(strip=True),
            "weight":       cols[4].get_text(strip=True),
            "reach":        cols[5].get_text(strip=True),
            "stance":       cols[6].get_text(strip=True),
            "wins":         cols[7].get_text(strip=True),
            "losses":       cols[8].get_text(strip=True),
            "draws":        cols[9].get_text(strip=True),
        })
    return fighters


def scrape_all_fighters(profile_urls: set | None = None) -> pd.DataFrame:
    """Scrapes fighter data from ufcstats.com listing pages + profiles.

    Args:
        profile_urls: If provided, only fetch individual profile pages for
            fighters whose URL is in this set. If None, fetch all profiles.
    """
    # Phase 1: scrape all 26 listing pages in parallel
    log.info("Scraping fighter listing pages (a-z)...")
    all_fighters = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = pool.map(_scrape_letter_page, "abcdefghijklmnopqrstuvwxyz")
        for fighters_list in results:
            all_fighters.extend(fighters_list)
    log.info(f"Found {len(all_fighters)} fighters from listing pages")

    # Phase 2: scrape individual profiles in parallel
    urls_to_fetch = [
        (i, f["fighter_url"]) for i, f in enumerate(all_fighters)
        if f["fighter_url"] and (profile_urls is None or f["fighter_url"] in profile_urls)
    ]
    if not urls_to_fetch:
        return pd.DataFrame(all_fighters)

    log.info(f"Fetching {len(urls_to_fetch)} fighter profiles...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(scrape_fighter_profile, url): (idx, url)
            for idx, url in urls_to_fetch
        }
        done = 0
        for future in as_completed(futures):
            idx, url = futures[future]
            try:
                profile = future.result()
                all_fighters[idx].update(profile)
            except Exception as e:
                log.warning(f"Profile failed {url}: {e}")
            done += 1
            if done % 200 == 0:
                log.info(f"  Profiles: {done}/{len(urls_to_fetch)}")

    log.info(f"Total fighters: {len(all_fighters)}")
    return pd.DataFrame(all_fighters)


# ─────────────────────────────────────────────
# 5. INCREMENTAL PIPELINE
# ─────────────────────────────────────────────

def upsert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame, pk: str):
    """Insert or replace rows from a DataFrame into a SQLite table."""
    if df.empty:
        return
    cols = ", ".join(df.columns)
    placeholders = ", ".join(["?"] * len(df.columns))
    sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"
    conn.executemany(sql, df.itertuples(index=False, name=None))


def get_known_event_urls(conn: sqlite3.Connection) -> set:
    rows = conn.execute("SELECT event_url FROM events").fetchall()
    return {r[0] for r in rows}


def get_fighter_urls_for_events(conn: sqlite3.Connection, event_urls: set) -> set:
    """Get unique fighter profile URLs for fighters who fought in the given events."""
    if not event_urls:
        return set()
    placeholders = ",".join(["?"] * len(event_urls))
    rows = conn.execute(f"""
        SELECT DISTINCT f.fighter_url
        FROM fights fi
        JOIN fighters f ON (
            (f.first_name || ' ' || f.last_name) = fi.fighter_1
            OR (f.first_name || ' ' || f.last_name) = fi.fighter_2
        )
        WHERE fi.event_url IN ({placeholders})
    """, list(event_urls)).fetchall()
    return {r[0] for r in rows}


def run_pipeline(full_refresh: bool = False):
    """
    Main entry point.
    - full_refresh=True  → re-scrape everything (first run)
    - full_refresh=False → only scrape events not in ufc.db
    """
    log.info("═══ UFC Stats Pipeline Starting ═══")
    init_db()

    # ── Events ──
    all_events = scrape_all_events()

    with get_db() as conn:
        known_urls = get_known_event_urls(conn) if not full_refresh else set()
        new_events = all_events[~all_events["event_url"].isin(known_urls)]
        log.info(f"New events to scrape: {len(new_events)}")

        # Upsert all events metadata (always update dates/locations)
        upsert_df(conn, "events", all_events, "event_url")

    # ── Fights (Phase 1: scrape event pages in parallel) ──
    event_rows = [ev for _, ev in new_events.iterrows()]
    log.info(f"Scraping fight lists from {len(event_rows)} events...")

    def _scrape_event_with_meta(ev_row):
        """Scrape fights for an event and attach event metadata."""
        try:
            fights = scrape_event_fights(ev_row["event_url"])
        except Exception as e:
            log.warning(f"Failed to scrape event {ev_row['event_name']}: {e}")
            return []
        for f in fights:
            f["event_url"]      = ev_row["event_url"]
            f["event_name"]     = ev_row["event_name"]
            f["event_date"]     = ev_row["date"]
            f["event_location"] = ev_row["location"]
        return fights

    all_fights = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_scrape_event_with_meta, ev): ev["event_name"] for ev in event_rows}
        for future in as_completed(futures):
            fights = future.result()
            all_fights.extend(fights)
    log.info(f"Found {len(all_fights)} fights across {len(event_rows)} events")

    # ── Fights (Phase 2: scrape fight detail pages in parallel) ──
    fight_urls = [f["fight_url"] for f in all_fights if f.get("fight_url")]
    if fight_urls:
        log.info(f"Scraping {len(fight_urls)} fight detail pages...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(scrape_fight_stats, url): url for url in fight_urls}
            url_to_stats = {}
            done = 0
            for future in as_completed(futures):
                url = futures[future]
                try:
                    url_to_stats[url] = future.result()
                except Exception as e:
                    log.warning(f"Fight stats failed {url}: {e}")
                done += 1
                if done % 500 == 0:
                    log.info(f"  Fight stats: {done}/{len(fight_urls)}")

        for f in all_fights:
            url = f.get("fight_url", "")
            if url in url_to_stats:
                f.update(url_to_stats[url])

    # ── Save fights to DB ──
    if all_fights:
        fights_df = pd.DataFrame(all_fights)
        with get_db() as conn:
            # Only keep columns that exist in the fights table
            cur = conn.execute("PRAGMA table_info(fights)")
            valid_cols = {row[1] for row in cur.fetchall()}
            fights_df = fights_df[[c for c in fights_df.columns if c in valid_cols]]
            upsert_df(conn, "fights", fights_df, "fight_url")
        log.info(f"Saved {len(fights_df)} new fights to {DB_PATH}")

    # Propagate corrected event_date and event_location to all existing fights
    with get_db() as conn:
        conn.execute("""
            UPDATE fights
            SET event_date    = (SELECT e.date     FROM events e WHERE e.event_url = fights.event_url),
                event_location = (SELECT e.location FROM events e WHERE e.event_url = fights.event_url)
            WHERE EXISTS (SELECT 1 FROM events e WHERE e.event_url = fights.event_url)
        """)
        log.info("Updated event_date and event_location on all fights from events table")

    # ── Fighters ──
    # On incremental runs, only fetch profiles for fighters in new events
    if full_refresh:
        profile_urls = None  # fetch all profiles
    else:
        with get_db() as conn:
            new_event_urls = set(new_events["event_url"]) if not new_events.empty else set()
            profile_urls = get_fighter_urls_for_events(conn, new_event_urls)
        log.info(f"Will fetch {len(profile_urls) if profile_urls else 0} fighter profiles (incremental)")

    fighters_df = scrape_all_fighters(profile_urls=profile_urls)
    with get_db() as conn:
        # Only keep columns that exist in the fighters table
        cur = conn.execute("PRAGMA table_info(fighters)")
        valid_cols = {row[1] for row in cur.fetchall()}
        fighters_df = fighters_df[[c for c in fighters_df.columns if c in valid_cols]]
        upsert_df(conn, "fighters", fighters_df, "fighter_url")
    log.info(f"Saved {len(fighters_df)} fighters to {DB_PATH}")

    log.info(f"═══ Pipeline Complete — DB: {DB_PATH} ═══")
    return all_events, all_fights


if __name__ == "__main__":
    import sys
    full = "--full" in sys.argv
    run_pipeline(full_refresh=full)
