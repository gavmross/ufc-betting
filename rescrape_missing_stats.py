"""
rescrape_missing_stats.py — one-off repair for fights whose row exists but whose
per-fight stat columns are empty.

Root cause: ufcstats.com's proof-of-work interstitial ("Checking your browser…")
was landing in the scraped HTML for a stretch of mid-2026 events, so the fight
detail pages parsed to nothing. scraper.get_soup() now clears the challenge; this
script re-fetches the detail pages for the affected fights and fills them in.

Usage:
    python rescrape_missing_stats.py            # repair 2020-01-01 onward
    python rescrape_missing_stats.py --since 2015-01-01
    python rescrape_missing_stats.py --dry-run
"""

import argparse
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

from scraper import DB_PATH, MAX_WORKERS, get_db, scrape_fight_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def find_missing(since: str) -> list[str]:
    """fight_urls with an empty totals column, for events on/after `since`."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT fight_url
            FROM fights
            WHERE event_date >= ?
              AND (tot_sig_str_f1 IS NULL OR tot_sig_str_f1 = '')
            ORDER BY event_date
            """,
            (since,),
        ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2020-01-01",
                    help="only repair fights on/after this ISO date (default 2020-01-01)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list affected fights without scraping")
    args = ap.parse_args()

    urls = find_missing(args.since)
    log.info(f"{len(urls)} fights missing stats since {args.since}")
    if not urls or args.dry_run:
        for u in urls:
            log.info(f"  {u}")
        return

    valid_cols = {row[1] for row in
                  sqlite3.connect(DB_PATH).execute("PRAGMA table_info(fights)")}

    fixed, empty, failed = 0, 0, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scrape_fight_stats, u): u for u in urls}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                stats = fut.result()
            except Exception as e:
                log.warning(f"scrape failed {url}: {e}")
                failed += 1
                continue
            if not stats or not stats.get("tot_sig_str_f1"):
                # Genuinely has no stats on the site (very old / DQ / no-contest).
                empty += 1
                continue
            cols = [c for c in stats if c in valid_cols]
            assignments = ", ".join(f"{c} = :{c}" for c in cols)
            with get_db() as conn:
                conn.execute(
                    f"UPDATE fights SET {assignments} WHERE fight_url = :fight_url",
                    {**{c: stats[c] for c in cols}, "fight_url": url},
                )
            fixed += 1
            if fixed % 20 == 0:
                log.info(f"  repaired {fixed}...")

    log.info(f"done — repaired {fixed}, still-empty (no data on site) {empty}, failed {failed}")


if __name__ == "__main__":
    main()
