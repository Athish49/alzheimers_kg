from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# -------------------------------------------------------------------
# Paths (assume you run this from project_root/alzforum)
# -------------------------------------------------------------------

ALZFORUM_ROOT = Path(__file__).resolve().parent
RAW_HTML_DIR = ALZFORUM_ROOT / "raw_html" / "alzbiomarker"
PROCESSED_DIR = ALZFORUM_ROOT / "processed"

BASE_URL = "https://www.alzforum.org"


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def read_html(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    # normalise whitespace, strip NBSP, etc.
    return " ".join(text.replace("\xa0", " ").split())


def parse_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    value = clean_text(value)
    if value in ("", "-"):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    value = clean_text(value)
    if value in ("", "-"):
        return None
    # Treat things like "<0.0001" as 0.0001 (lower bound)
    if value.startswith("<"):
        value = value[1:]
    try:
        return float(value)
    except ValueError:
        return None


def make_biomarker_key(analyte_label: str) -> str:
    """
    Turn strings like 'Aβ42 (CSF)' or 'albumin ratio' into a stable ID:
        'abeta42_csf', 'albumin_ratio'
    """
    text = analyte_label
    text = text.lower()
    # replace greek beta with 'beta'
    text = text.replace("β", "beta")
    # remove parentheses but keep contents
    text = text.replace("(", "_").replace(")", "")
    # replace non-alphanum with underscore
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def split_analyte_and_fluid(analyte_label: str) -> Tuple[str, Optional[str]]:
    """
    'Aβ42 (CSF)'            -> ('Aβ42', 'CSF')
    'Aβ40 (Plasma/Serum)'   -> ('Aβ40', 'Plasma/Serum')
    'albumin ratio'         -> ('albumin ratio', None)
    """
    m = re.match(r"^(.*?)(?:\s*\((.+)\))?$", analyte_label.strip())
    if not m:
        return analyte_label.strip(), None
    core = m.group(1).strip()
    fluid = m.group(2).strip() if m.group(2) else None
    return core, fluid


def classify_analyte(core_name: str) -> str:
    """
    Very lightweight heuristic to assign analyte_class.
    This is *not* perfect; we can refine later.
    """
    name = core_name.lower()
    if any(tok in name for tok in ["abeta", "aβ", "amyloid"]):
        return "amyloid"
    if "tau" in name:
        return "tau"
    if any(tok in name for tok in ["nfl", "neurofilament", "nf-l"]):
        return "neurodegeneration"
    if any(tok in name for tok in ["gfap", "mcp", "s100", "trem", "ykl"]):
        return "inflammation"
    return "other"


def extract_intro(article) -> str:
    """
    Grab the paragraphs at the top of the article, before the first table.
    """
    paras: List[str] = []
    for child in article.children:
        name = getattr(child, "name", None)
        if name == "p":
            paras.append(clean_text(child.get_text(" ", strip=True)))
        elif name == "table":
            break
    return "\n\n".join(p for p in paras if p)


# -------------------------------------------------------------------
# Parsers for the two big tables
# -------------------------------------------------------------------

def parse_main_versions_table(
    table,
    biomarker_page_id: str,
    page_name: str,
    url: str,
) -> List[Dict]:
    """
    Parse the first big table (AD vs CTRL, MCI-AD vs MCI-Stable).

    Layout per data row:
        biomarker | n(1.x) | eff(1.x) | p(1.x) | n(2.x) | eff(2.x) | p(2.x)
                  | n(3.0) | eff(3.0) | p(3.0) | meta links
    """
    rows: List[Dict] = []
    tbody = table.find("tbody")
    if not tbody:
        return rows

    current_comparison: Optional[str] = None

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        cell_texts = [clean_text(td.get_text(" ", strip=True)) for td in tds]

        # Header row for a comparison group, e.g. "AD vs CTRL"
        if len(cell_texts) >= 2 and cell_texts[1].startswith("#"):
            current_comparison = cell_texts[0]
            continue

        # Data rows should have ~11 cells; guard against odd rows
        if len(cell_texts) < 11:
            continue

        analyte_label = cell_texts[0]
        biomarker_key = make_biomarker_key(analyte_label)

        def val(idx: int) -> str:
            return cell_texts[idx] if idx < len(cell_texts) else ""

        meta_td = tds[-1]
        meta_text = clean_text(meta_td.get_text(" ", strip=True))
        meta_links = [urljoin(BASE_URL, a["href"]) for a in meta_td.find_all("a", href=True)]

        # Fixed order: v1.x, v2.x, v3.0
        versions = [
            ("1.x", val(1), val(2), val(3)),
            ("2.x", val(4), val(5), val(6)),
            ("3.0", val(7), val(8), val(9)),
        ]

        for version_label, n_raw, eff_raw, p_raw in versions:
            if (
                clean_text(n_raw) in ("", "-")
                and clean_text(eff_raw) in ("", "-")
                and clean_text(p_raw) in ("", "-")
            ):
                continue  # no information for this version

            core_name, fluid = split_analyte_and_fluid(analyte_label)

            rows.append(
                {
                    "biomarker_page_id": biomarker_page_id,
                    "page_name": page_name,
                    "source_url": url,
                    "section": "version_summary",
                    "comparison": current_comparison,  # AD vs CTRL / MCI-AD vs MCI-Stable
                    "biomarker_key": biomarker_key,
                    "analyte_label": analyte_label,
                    "analyte_core": core_name,
                    "fluid": fluid,
                    "analyte_class": classify_analyte(core_name),
                    "version": version_label,
                    "n_raw": clean_text(n_raw),
                    "effect_size_raw": clean_text(eff_raw),
                    "p_value_raw": clean_text(p_raw),
                    "n": parse_int(n_raw),
                    "effect_size": parse_float(eff_raw),
                    "p_value": parse_float(p_raw),
                    "meta_text": meta_text,
                    "meta_urls": ";".join(meta_links),
                }
            )

    return rows


def parse_cross_disease_table(
    table,
    biomarker_page_id: str,
    page_name: str,
    url: str,
) -> List[Dict]:
    """
    Parse the second big table ("Cross Diseases (non-AD vs AD)").
    """
    rows: List[Dict] = []
    tbody = table.find("tbody")
    if not tbody:
        return rows

    current_comparison: Optional[str] = None

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        cell_texts = [clean_text(td.get_text(" ", strip=True)) for td in tds]

        # comparison header row, e.g. "ALS vs AD"
        if len(cell_texts) >= 2 and cell_texts[1].startswith("#"):
            current_comparison = cell_texts[0]
            continue

        if len(cell_texts) < 4:
            continue  # not a standard data row

        analyte_label = cell_texts[0]
        if not analyte_label or analyte_label.lower().startswith("cross diseases"):
            continue

        n_raw, eff_raw, p_raw = cell_texts[1], cell_texts[2], cell_texts[3]
        meta_td = tds[-1]
        meta_text = clean_text(meta_td.get_text(" ", strip=True))
        meta_links = [urljoin(BASE_URL, a["href"]) for a in meta_td.find_all("a", href=True)]

        core_name, fluid = split_analyte_and_fluid(analyte_label)
        biomarker_key = make_biomarker_key(analyte_label)

        rows.append(
            {
                "biomarker_page_id": biomarker_page_id,
                "page_name": page_name,
                "source_url": url,
                "section": "cross_diseases",
                "comparison": current_comparison,  # e.g. ALS vs AD, CTRL vs AD
                "biomarker_key": biomarker_key,
                "analyte_label": analyte_label,
                "analyte_core": core_name,
                "fluid": fluid,
                "analyte_class": classify_analyte(core_name),
                "version": "3.0",  # cross-disease table only has v3.0 columns
                "n_raw": n_raw,
                "effect_size_raw": eff_raw,
                "p_value_raw": p_raw,
                "n": parse_int(n_raw),
                "effect_size": parse_float(eff_raw),
                "p_value": parse_float(p_raw),
                "meta_text": meta_text,
                "meta_urls": ";".join(meta_links),
            }
        )

    return rows


def parse_versioning_history(
    html: str,
    biomarker_page_id: str,
    page_name: str,
    url: str,
) -> Tuple[Dict, List[Dict]]:
    """
    Parse the whole 'Versioning History' page into:
      - page-level metadata (title, intro text)
      - row-level effect-size summaries from both tables
    """
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article", id="article")
    if article is None:
        raise ValueError("Could not find main <article id='article'> content")

    title_el = article.find("h1", class_="page-title")
    subtitle_el = article.find("h2", class_="pane-subtitle")

    page_meta = {
        "biomarker_page_id": biomarker_page_id,
        "name": page_name,
        "source_url": url,
        "page_title": clean_text(title_el.get_text(strip=True)) if title_el else "",
        "page_subtitle": clean_text(subtitle_el.get_text(strip=True)) if subtitle_el else "",
        "intro_text": extract_intro(article),
    }

    tables = article.find_all("table")
    all_rows: List[Dict] = []

    if len(tables) >= 1:
        all_rows.extend(parse_main_versions_table(tables[0], biomarker_page_id, page_name, url))
    if len(tables) >= 2:
        all_rows.extend(parse_cross_disease_table(tables[1], biomarker_page_id, page_name, url))

    return page_meta, all_rows


# -------------------------------------------------------------------
# Top-level processing
# -------------------------------------------------------------------

def build_biomarker_table(effects_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the long effects table down to one row per biomarker_key,
    giving us biomarker-level metadata:
      biomarker_key, analyte_core, analyte_class, fluid, example_label
    """
    if effects_df.empty:
        return pd.DataFrame(
            columns=["biomarker_key", "analyte_core", "analyte_class", "fluid", "analyte_label_example"]
        )

    grp = (
        effects_df
        .sort_values(["biomarker_key", "section"])
        .groupby("biomarker_key", as_index=False)
        .agg(
            analyte_core=("analyte_core", "first"),
            analyte_class=("analyte_class", "first"),
            fluid=("fluid", "first"),
            analyte_label_example=("analyte_label", "first"),
        )
    )
    return grp


def process_alzbiomarker() -> None:
    """
    Main entry point:
      - reads processed/alzbiomarker_index.csv
      - parses raw_html/alzbiomarker/versioning_history.html
      - writes:
          processed/alzbiomarker_pages.csv
          processed/alzbiomarker_effects.csv
          processed/alzbiomarker_biomarkers.csv
    """
    index_csv = PROCESSED_DIR / "alzbiomarker_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    index_df = pd.read_csv(index_csv)

    page_meta_rows: List[Dict] = []
    effects_rows: List[Dict] = []

    for _, row in index_df.iterrows():
        biomarker_page_id = str(row["biomarker_id"])  # "versioning_history"
        url = row["url"]
        name = row["name"]

        html_path = RAW_HTML_DIR / f"{biomarker_page_id}.html"
        if not html_path.exists():
            print(f"[WARN] HTML file not found for {biomarker_page_id} at {html_path}, skipping.")
            continue

        html = read_html(html_path)
        page_meta, rows = parse_versioning_history(html, biomarker_page_id, name, url)

        page_meta_rows.append(page_meta)
        effects_rows.extend(rows)

    pages_df = pd.DataFrame(page_meta_rows)
    effects_df = pd.DataFrame(effects_rows)

    # Build biomarker-level table from long effects table
    biomarkers_df = build_biomarker_table(effects_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pages_df.to_csv(PROCESSED_DIR / "alzbiomarker_pages.csv", index=False)
    effects_df.to_csv(PROCESSED_DIR / "alzbiomarker_effects.csv", index=False)
    biomarkers_df.to_csv(PROCESSED_DIR / "alzbiomarker_biomarkers.csv", index=False)

    print(f"[ALZBIOMARKER] Wrote {len(pages_df)} page-metadata rows.")
    print(f"[ALZBIOMARKER] Wrote {len(effects_df)} effect-size rows.")
    print(f"[ALZBIOMARKER] Wrote {len(biomarkers_df)} unique biomarker rows.")


if __name__ == "__main__":
    process_alzbiomarker()