from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# -------------------------------------------------------------------
# Paths (assume you run this from project_root/alzforum)
# -------------------------------------------------------------------

ALZFORUM_ROOT = Path(__file__).resolve().parent
RAW_HTML_DIR = ALZFORUM_ROOT / "raw_html" / "therapeutics"
PROCESSED_DIR = ALZFORUM_ROOT / "processed"

BASE_URL = "https://www.alzforum.org"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def read_html(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.replace("\xa0", " ").split())


def unique_pipe_join(values: pd.Series) -> str:
    """Join non-empty strings with ' | ' preserving first-seen order."""
    seen = []
    for v in values:
        if not isinstance(v, str):
            continue
        v = clean_text(v)
        if not v:
            continue
        if v not in seen:
            seen.append(v)
    return " | ".join(seen)


# -------------------------------------------------------------------
# Parsing of search result pages
# -------------------------------------------------------------------

def parse_search_results_table(
    table,
    search_id: str,
    search_name: str,
    page_url: str,
) -> List[Dict]:
    """
    Parse the Search Results table on a therapeutics search page.

    Table columns:
      Name | Synonyms | FDA Status | Company | Target Type | Therapy Type | Approved For
    """
    rows: List[Dict] = []
    if table is None:
        return rows

    tbody = table.find("tbody")
    if not tbody:
        return rows

    for tr in tbody.find_all("tr"):
        # First TH cell contains the Name + detail link
        th = tr.find("th", scope="row")
        if th is None:
            continue

        link = th.find("a", href=True)
        if not link:
            continue

        name = clean_text(link.get_text(" ", strip=True))
        href = link["href"]
        therapeutic_url = urljoin(BASE_URL, href)
        # slug from /therapeutics/<slug>
        therapeutic_id = href.rstrip("/").split("/")[-1]

        # Remaining columns in fixed order
        tds = tr.find_all("td")
        # Guard for weird/incomplete rows
        if len(tds) < 6:
            continue

        synonyms = clean_text(tds[0].get_text(" ", strip=True))
        fda_status = clean_text(tds[1].get_text(" ", strip=True))
        company = clean_text(tds[2].get_text(" ", strip=True))
        target_type = clean_text(tds[3].get_text(" ", strip=True))
        therapy_type = clean_text(tds[4].get_text(" ", strip=True))
        approved_for = clean_text(tds[5].get_text(" ", strip=True))

        rows.append(
            {
                # context: which search combo produced this row
                "search_id": search_id,
                "search_name": search_name,
                "search_url": page_url,

                # therapeutic-level info
                "therapeutic_id": therapeutic_id,
                "therapeutic_name": name,
                "therapeutic_url": therapeutic_url,
                "synonyms": synonyms,
                "fda_status": fda_status,
                "company": company,
                "target_type": target_type,
                "therapy_type": therapy_type,
                "approved_for": approved_for,
            }
        )

    return rows


def parse_therapeutics_search_page(
    html: str,
    search_id: str,
    search_name: str,
    page_url: str,
) -> List[Dict]:
    """
    Given a therapeutics *search* page HTML (like the one you showed),
    extract all rows from the Search Results table.

    If the page has no results (no table), this returns an empty list
    and the caller just moves on.
    """
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article", id="article")
    if article is None:
        # Not a standard therapeutics page; ignore (e.g. timeline)
        return []

    results_section = article.find("section", id="results")
    if results_section is None:
        # e.g. timeline / non-search pages
        return []

    table = results_section.find("table")
    if table is None:
        # This is the "0 therapeutics" case: no table, just text.
        return []

    return parse_search_results_table(
        table=table,
        search_id=search_id,
        search_name=search_name,
        page_url=page_url,
    )


# -------------------------------------------------------------------
# Build aggregated therapeutics_entities from search results
# -------------------------------------------------------------------

def build_therapeutics_entities(search_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the raw search results to one row per therapeutic_id.

    We keep:
      - therapeutic_id (slug)
      - name
      - url
      - synonyms (all distinct, pipe-joined)
      - fda_statuses (all distinct, pipe-joined)
      - companies (all distinct)
      - target_types (all distinct)
      - therapy_types (all distinct)
      - approved_for (all distinct)
      - source_search_ids (which search pages this drug appeared in)

    This is a *partial* therapeutics_entities table:
    mechanism_summary, primary_indication, status_overall, etc.
    will be filled in later from the individual therapeutic pages.
    """
    if search_df.empty:
        return pd.DataFrame(
            columns=[
                "therapeutic_id",
                "name",
                "url",
                "synonyms",
                "fda_statuses",
                "companies",
                "target_types",
                "therapy_types",
                "approved_for",
                "source_search_ids",
            ]
        )

    group_cols = ["therapeutic_id"]

    agg_df = (
        search_df
        .sort_values(["therapeutic_id", "therapeutic_name"])
        .groupby(group_cols, as_index=False)
        .agg(
            name=("therapeutic_name", "first"),
            url=("therapeutic_url", "first"),
            synonyms=("synonyms", unique_pipe_join),
            fda_statuses=("fda_status", unique_pipe_join),
            companies=("company", unique_pipe_join),
            target_types=("target_type", unique_pipe_join),
            therapy_types=("therapy_type", unique_pipe_join),
            approved_for=("approved_for", unique_pipe_join),
            source_search_ids=("search_id", unique_pipe_join),
        )
    )

    return agg_df


# -------------------------------------------------------------------
# Top-level driver
# -------------------------------------------------------------------

def process_therapeutics() -> None:
    """
    Main entry point for current Therapeutics processing:

      INPUT:
        - processed/therapeutics_index.csv
        - raw_html/therapeutics/<search_id>.html
          (these are the search pages you already downloaded)

      OUTPUT:
        - processed/therapeutics_search_results.csv
          (one row per search-results row)
        - processed/therapeutics_entities.csv
          (one row per unique therapeutic slug)
    """
    index_csv = PROCESSED_DIR / "therapeutics_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    index_df = pd.read_csv(index_csv)

    all_rows: List[Dict] = []

    for _, row in index_df.iterrows():
        search_id = str(row["therapeutic_id"])
        url = row["url"]
        name = str(row.get("name", ""))

        html_path = RAW_HTML_DIR / f"{search_id}.html"
        if not html_path.exists():
            print(f"[WARN] HTML file not found for search_id={search_id} at {html_path}, skipping.")
            continue

        html = read_html(html_path)
        rows = parse_therapeutics_search_page(
            html=html,
            search_id=search_id,
            search_name=name,
            page_url=url,
        )

        if not rows:
            print(f"[INFO] No search results found on page {search_id} (this is fine for 0-therapeutic combos).")
        all_rows.extend(rows)

    # Raw search results (long table)
    search_df = pd.DataFrame(all_rows)

    # Aggregated per-therapeutic entity table
    entities_df = build_therapeutics_entities(search_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    search_df.to_csv(PROCESSED_DIR / "therapeutics_search_results.csv", index=False)
    entities_df.to_csv(PROCESSED_DIR / "therapeutics_entities.csv", index=False)

    print(f"[THERAPEUTICS] Wrote {len(search_df)} search-result rows.")
    print(f"[THERAPEUTICS] Wrote {len(entities_df)} unique therapeutics.")


if __name__ == "__main__":
    process_therapeutics()