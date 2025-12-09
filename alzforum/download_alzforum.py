"""
Download Alzforum data sources (AlzPedia, AlzBiomarker, AlzRisk, Therapeutics)
and save raw HTML locally for reproducibility.

This script does TWO things per source:
1) Build an index CSV of pages (IDs + URLs).
2) Download each page's HTML into alzforum/raw_html/<source>/.

IMPORTANT:
- The CSS selectors / URL patterns for index extraction are educated guesses.
  You will likely need to open the site in your browser, inspect the HTML,
  and tweak the `build_*_index()` functions accordingly.
"""

import csv
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# --------------------------
# PATHS & BASIC CONFIG
# --------------------------

ALZFORUM_ROOT = Path(__file__).resolve().parent

RAW_HTML_DIR = ALZFORUM_ROOT / "raw_html"
RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = ALZFORUM_ROOT / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Subdirs for each source
ALZPEDIA_HTML_DIR = RAW_HTML_DIR / "alzpedia"
ALZBIOMARKER_HTML_DIR = RAW_HTML_DIR / "alzbiomarker"
ALZRISK_HTML_DIR = RAW_HTML_DIR / "alzrisk"
THERAPEUTICS_HTML_DIR = RAW_HTML_DIR / "therapeutics"

for d in [ALZPEDIA_HTML_DIR, ALZBIOMARKER_HTML_DIR, ALZRISK_HTML_DIR, THERAPEUTICS_HTML_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# --------------------------
# HTTP HELPERS
# --------------------------

# Use a browser-like user-agent to reduce chances of 403
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_html(url: str, sleep_seconds: float = 0.5) -> Optional[str]:
    """Fetch HTML from a URL with basic error handling and politeness delays."""
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"[WARN] got status {resp.status_code} for {url}")
            return None
        time.sleep(sleep_seconds)
        return resp.text
    except requests.RequestException as e:
        print(f"[ERROR] Request failed for {url}: {e}")
        return None


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------
# INDEX BUILDERS
# --------------------------

def build_alzpedia_index(index_path: Path) -> None:
    """
    Build an index of AlzPedia entities.

    Strategy (approximate):
    - Fetch an AlzPedia index page that lists entries.
    - Collect <a> tags whose href looks like '/alzpedia/<slug>'.
    - Avoid sub-URLs like '/alzpedia/papers/...'.

    NOTE:
    - The selectors here are *guesses*.
      Open https://www.alzforum.org/alzpedia in your browser,
      inspect the HTML, and adjust the logic if necessary.
    """
    base_url = "https://www.alzforum.org/alzpedia"
    print(f"[ALZPEDIA] Fetching index page: {base_url}")
    html = fetch_html(base_url)
    if html is None:
        print("[ALZPEDIA] Failed to fetch index; please build index manually.")
        return

    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a", href=True)

    records: List[Dict[str, str]] = []
    seen = set()

    for a in links:
        href = a["href"]
        # Example pattern: '/alzpedia/app'
        if not href.startswith("/alzpedia/"):
            continue
        if "/papers/" in href:
            # skip paper-style entries
            continue

        full_url = "https://www.alzforum.org" + href
        slug = href.split("/")[-1]
        if slug in seen:
            continue
        seen.add(slug)

        title_text = a.get_text(strip=True)
        records.append(
            {
                "entity_id": slug,
                "url": full_url,
                "title": title_text,
            }
        )

    if not records:
        print("[ALZPEDIA] WARNING: no records found – check selectors.")
    else:
        print(f"[ALZPEDIA] Found {len(records)} candidate entries.")

    # Write CSV
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["entity_id", "url", "title"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[ALZPEDIA] Index saved to {index_path}")


def build_alzbiomarker_index(index_path: Path) -> None:
    """
    Build an index for AlzBiomarker using the single 'Versioning History' page.

    We only need:
      https://www.alzforum.org/alzbiomarker/about-alzbiomarker/versioning-history

    This page contains the big table with all biomarkers, comparisons, and
    version 3.0 effect sizes. We'll parse that later in process_alzforum.py.
    """

    RECORDS = [
        {
            "biomarker_id": "versioning_history",
            "url": "https://www.alzforum.org/alzbiomarker/about-alzbiomarker/versioning-history",
            "name": "AlzBiomarker Versioning History",
        }
    ]

    print(f"[ALZBIOMARKER] Building index with {len(RECORDS)} entry (versioning history page).")

    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["biomarker_id", "url", "name"])
        writer.writeheader()
        writer.writerows(RECORDS)

    print(f"[ALZBIOMARKER] Index saved to {index_path}")


def build_alzrisk_index(index_path: Path) -> None:
    """
    Build an index of AlzRisk risk factor pages.

    Strategy (approximate):
    - Fetch the AlzRisk landing page.
    - Collect links that look like '/alzrisk/<slug>' and are risk-factor pages.

    You may need to refine which <a> tags to use based on the page structure.
    """
    base_url = "https://www.alzforum.org/alzrisk"
    print(f"[ALZRISK] Fetching index page: {base_url}")
    html = fetch_html(base_url)
    if html is None:
        print("[ALZRISK] Failed to fetch index; please build index manually.")
        return

    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a", href=True)

    records: List[Dict[str, str]] = []
    seen = set()

    for a in links:
        href = a["href"]
        if not href.startswith("/alzrisk/"):
            continue

        # skip obvious non-risk pages if needed (e.g., '/alzrisk' or '/alzrisk/tools')
        parts = href.strip("/").split("/")
        if len(parts) < 2:
            # e.g., '/alzrisk'
            continue

        slug = parts[-1]
        if slug in seen:
            continue
        seen.add(slug)

        full_url = "https://www.alzforum.org" + href
        title_text = a.get_text(strip=True)
        records.append(
            {
                "risk_factor_id": slug,
                "url": full_url,
                "name": title_text,
            }
        )

    if not records:
        print("[ALZRISK] WARNING: no records found – check selectors.")
    else:
        print(f"[ALZRISK] Found {len(records)} candidate risk factor entries.")

    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["risk_factor_id", "url", "name"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[ALZRISK] Index saved to {index_path}")


def build_therapeutics_index(index_path: Path) -> None:
    """
    Build an index of Therapeutics database entries.

    Strategy (approximate):
    - Fetch the Therapeutics landing or 'all therapeutics' list page.
    - Collect links of the form '/therapeutics/<slug>'.

    NOTE:
    - You might need to find the specific URL that lists *all* therapeutics.
      Sometimes the site provides a filterable table or a dedicated listing page.
      Once you know that URL, replace `base_url` below if needed.
    """
    base_url = "https://www.alzforum.org/therapeutics"
    print(f"[THERAPEUTICS] Fetching index page: {base_url}")
    html = fetch_html(base_url)
    if html is None:
        print("[THERAPEUTICS] Failed to fetch index; please build index manually.")
        return

    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a", href=True)

    records: List[Dict[str, str]] = []
    seen = set()

    for a in links:
        href = a["href"]
        # Example pattern: '/therapeutics/lecanemab'
        if not href.startswith("/therapeutics/"):
            continue

        parts = href.strip("/").split("/")
        if len(parts) < 2:
            # e.g., '/therapeutics'
            continue

        slug = parts[-1]
        if slug in seen:
            continue
        seen.add(slug)

        full_url = "https://www.alzforum.org" + href
        name = a.get_text(strip=True)
        records.append(
            {
                "therapeutic_id": slug,
                "url": full_url,
                "name": name,
            }
        )

    if not records:
        print("[THERAPEUTICS] WARNING: no records found – check selectors.")
    else:
        print(f"[THERAPEUTICS] Found {len(records)} candidate therapeutics.")

    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["therapeutic_id", "url", "name"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[THERAPEUTICS] Index saved to {index_path}")


# --------------------------
# DOWNLOAD HTML FOR EACH INDEX
# --------------------------

def download_from_index(
    index_path: Path,
    id_column: str,
    url_column: str,
    target_dir: Path,
    sleep_seconds: float = 0.5,
) -> None:
    """
    Given an index CSV with ID + URL columns, download each page HTML
    into target_dir as '<id>.html', skipping files that already exist.
    """
    if not index_path.exists():
        print(f"[DOWNLOAD] Index file not found: {index_path}, skipping.")
        return

    print(f"[DOWNLOAD] Loading index from {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[DOWNLOAD] {len(rows)} rows found in index.")

    for row in tqdm(rows, desc=f"Downloading to {target_dir.name}"):
        item_id = row[id_column]
        url = row[url_column]
        dest = target_dir / f"{item_id}.html"

        if dest.exists():
            continue

        html = fetch_html(url, sleep_seconds=sleep_seconds)
        if html is None:
            print(f"[DOWNLOAD] Failed to fetch {url}, skipping.")
            continue
        save_text(dest, html)


# --------------------------
# MAIN
# --------------------------

def main():
    print("=== Alzheimer’s KG – Alzforum Download Script ===\n")

    # Index paths
    alzpedia_index = PROCESSED_DIR / "alzpedia_index.csv"
    alzbiomarker_index = PROCESSED_DIR / "alzbiomarker_index.csv"
    alzrisk_index = PROCESSED_DIR / "alzrisk_index.csv"
    therapeutics_index = PROCESSED_DIR / "therapeutics_index.csv"

    # 1) Build indices if they don't exist
    if not alzpedia_index.exists():
        build_alzpedia_index(alzpedia_index)
    else:
        print(f"[ALZPEDIA] Index already exists: {alzpedia_index}")

    if not alzbiomarker_index.exists():
        build_alzbiomarker_index(alzbiomarker_index)
    else:
        print(f"[ALZBIOMARKER] Index already exists: {alzbiomarker_index}")

    # if not alzrisk_index.exists():
    #     build_alzrisk_index(alzrisk_index)
    # else:
    #     print(f"[ALZRISK] Index already exists: {alzrisk_index}")

    if not therapeutics_index.exists():
        build_therapeutics_index(therapeutics_index)
    else:
        print(f"[THERAPEUTICS] Index already exists: {therapeutics_index}")

    print()

    # 2) Download HTML per source
    download_from_index(
        alzpedia_index,
        id_column="entity_id",
        url_column="url",
        target_dir=ALZPEDIA_HTML_DIR,
    )

    download_from_index(
        alzbiomarker_index,
        id_column="biomarker_id",
        url_column="url",
        target_dir=ALZBIOMARKER_HTML_DIR,
    )

    # download_from_index(
    #     alzrisk_index,
    #     id_column="risk_factor_id",
    #     url_column="url",
    #     target_dir=ALZRISK_HTML_DIR,
    # )

    download_from_index(
        therapeutics_index,
        id_column="therapeutic_id",
        url_column="url",
        target_dir=THERAPEUTICS_HTML_DIR,
    )

    print("\n=== DONE ===")
    print(f"Raw HTML is stored under: {RAW_HTML_DIR}")


if __name__ == "__main__":
    main()