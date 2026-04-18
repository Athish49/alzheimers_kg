"""
Process individual Alzforum Therapeutics pages.

Inputs:
  - processed/therapeutics_entities.csv   (built from search results)
  - raw_html/therapeutic_pages/<slug>.html   (downloaded on demand)

Outputs:
  - processed/therapeutics_entities_enriched.csv
      (base therapeutics_entities.csv + extra columns from detail pages)
  - processed/therapeutics_targets.csv
      (one row per therapeutic x high-level target type)
  - processed/therapeutics_trials.csv
      (ONE SUMMARY ROW per therapeutic for now)
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

ALZFORUM_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ALZFORUM_ROOT / "processed"

THERAPEUTIC_PAGES_DIR = ALZFORUM_ROOT / "raw_html" / "therapeutic_pages"
THERAPEUTIC_PAGES_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.alzforum.org"

# -------------------------------------------------------------------
# HTTP helpers
# -------------------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_html(url: str, sleep_seconds: float = 0.5) -> Optional[str]:
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


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.replace("\xa0", " ").split())


def strip_timeline_suffix(s: Optional[str]) -> Optional[str]:
    """
    Remove trailing '(timeline)' from strings like 'Tau (timeline)'.
    Returns the cleaned string or None if input is falsy.
    """
    if not s:
        return s
    s = re.sub(r"\s*\(timeline\)\s*", "", s)
    return s.strip() or None


def read_or_download_html(slug: str, url: str) -> Optional[str]:
    """
    Check raw_html/therapeutic_pages/<slug>.html;
    if missing, download from url.
    """
    path = THERAPEUTIC_PAGES_DIR / f"{slug}.html"
    if path.exists():
        return path.read_text(encoding="utf-8")

    html = fetch_html(url)
    if html is None:
        return None

    path.write_text(html, encoding="utf-8")
    return html


# -------------------------------------------------------------------
# Parsing helpers
# -------------------------------------------------------------------

def find_article(soup: BeautifulSoup):
    article = soup.find("article", id="article")
    if article is None:
        article = soup.find("main") or soup
    return article


def find_section_block(article, section_keywords: List[str]) -> Optional[str]:
    """
    Legacy fallback: find first H1/H2/H3 whose text contains any keyword,
    then concatenate text of siblings until next H1/H2/H3.
    """
    headings = article.find_all(["h1", "h2", "h3"])
    target_heading = None
    for h in headings:
        txt = h.get_text(strip=True).lower()
        if any(kw.lower() in txt for kw in section_keywords):
            target_heading = h
            break

    if target_heading is None:
        return None

    chunks: List[str] = []
    for sib in target_heading.next_siblings:
        name = getattr(sib, "name", None)
        if name in ("h1", "h2", "h3"):
            break
        if name in ("p", "ul", "ol", "div"):
            chunks.append(clean_text(sib.get_text(" ", strip=True)))

    return "\n\n".join([c for c in chunks if c])


def extract_overview_section(article):
    return article.find(id="overview")


def extract_background_section(article):
    # try explicit mechanism-of-action first
    for sid in ["mechanism-of-action", "mechanism"]:
        sec = article.find(id=sid)
        if sec:
            return sec
    # then generic background
    sec = article.find(id="background")
    if sec:
        return sec
    return None


def section_text(section) -> Optional[str]:
    if section is None:
        return None
    return clean_text(section.get_text(" ", strip=True))


def parse_overview_kv_from_section(section) -> Dict[str, str]:
    """
    Parse <section id="overview"> which looks like:

      <p>
        <strong>Name:</strong> AADvac1<br/>
        <span><strong>Synonyms:</strong> Axon peptide 108 ...</span><br/>
        ...
      </p>
    """
    kv: Dict[str, str] = {}

    key_map = {
        "Name": "name",
        "Synonyms": "synonyms",
        "Therapy Type": "therapy_type",
        "Target Type": "target_type",
        "Condition(s)": "conditions",
        "Conditions": "conditions",
        "U.S. FDA Status": "fda_status",
        "US. FDA Status": "fda_status",
        "US FDA Status": "fda_status",
        "Company": "company",
        "Approved For": "approved_for",
    }

    for strong in section.find_all("strong"):
        label_raw = clean_text(strong.get_text(" ", strip=True))
        if not label_raw.endswith(":"):
            continue
        label = label_raw[:-1]  # drop trailing ':'

        # collect siblings until <br/> (or end of this block)
        value_parts: List[str] = []
        for sib in strong.next_siblings:
            name = getattr(sib, "name", None)
            if name == "br":
                break
            if isinstance(sib, str):
                value_parts.append(sib)
            else:
                value_parts.append(sib.get_text(" ", strip=True))

        value = clean_text(" ".join(value_parts))
        if not value:
            continue

        key = key_map.get(label, label.lower().replace(" ", "_"))
        kv[key] = value

    return kv


def parse_overview_kv_from_text(overview_text: str) -> Dict[str, str]:
    """
    Fallback regex parser if we ever hit a page without the usual
    <strong>Label:</strong> structure.
    """
    if not overview_text:
        return {}

    text = re.sub(r"\s+", " ", overview_text)

    labels_regex = (
        r"Name:|Synonyms:|Therapy Type:|Target Type:|Condition\(s\):|"
        r"U\.S\. FDA Status:|US\. FDA Status:|US FDA Status:|Company:|Approved For:"
    )

    def capture(label: str) -> Optional[str]:
        pattern = rf"{re.escape(label)}\s*(.+?)\s*(?={labels_regex}|$)"
        m = re.search(pattern, text)
        if not m:
            return None
        return clean_text(m.group(1))

    kv = {
        "name": capture("Name:"),
        "synonyms": capture("Synonyms:"),
        "therapy_type": capture("Therapy Type:"),
        "target_type": capture("Target Type:"),
        "conditions": capture("Condition(s):"),
        "fda_status": (
            capture("U.S. FDA Status:")
            or capture("US. FDA Status:")
            or capture("US FDA Status:")
        ),
        "company": capture("Company:"),
        "approved_for": capture("Approved For:"),
    }

    return {k: v for k, v in kv.items() if v}


def summarise_mechanism(
    mech_text: Optional[str], max_sentences: int = 2
) -> Optional[str]:
    """
    Take the full 'Background' / mechanism text and produce a short summary,
    stripping a leading 'Background' prefix if present.
    """
    if not mech_text:
        return None

    raw = mech_text.replace("\n", " ")

    # Drop leading "Background" (case-insensitive)
    raw = re.sub(r"^\s*background\s*", "", raw, flags=re.IGNORECASE)

    # Crude sentence split
    parts = [p.strip() for p in re.split(r"\.(\s+|$)", raw) if p.strip()]
    if not parts:
        return None

    summary = ". ".join(parts[:max_sentences])
    if not summary.endswith("."):
        summary += "."
    return summary


def infer_trial_phase_and_status(
    fda_status: Optional[str],
) -> Tuple[Optional[int], Optional[bool], Optional[str]]:
    """
    Use the U.S. FDA Status string to infer max phase and overall status.
    """
    if not fda_status:
        return None, None, None

    text = fda_status.lower()

    phase_nums: List[int] = []
    for m in re.finditer(r"phase\s*(\d)", text):
        try:
            phase_nums.append(int(m.group(1)))
        except ValueError:
            continue

    trial_phase_max = max(phase_nums) if phase_nums else None
    has_phase3 = trial_phase_max is not None and trial_phase_max >= 3

    if "approved" in text:
        status = "approved"
    elif any(w in text for w in ["discontinued", "terminated", "halted", "suspended"]):
        status = "discontinued"
    elif phase_nums:
        status = "ongoing"
    else:
        status = "unknown"

    return trial_phase_max, has_phase3, status


def infer_timeline_stats(article) -> Tuple[Optional[int], Optional[int]]:
    """
    Look at the Clinical Trial Timeline table (if present) and infer:
      - max phase across trials
      - number of trials with a timeline-span
    """
    sec = article.find(id="timeline")
    if not sec:
        return None, None

    table = sec.find("table")
    if not table:
        return None, None

    tbody = table.find("tbody") or table
    max_phase: Optional[int] = None
    trial_count = 0

    for tr in tbody.find_all("tr"):
        span_div = tr.find("div", class_="timeline-span")
        if not span_div:
            continue

        trial_count += 1
        classes = span_div.get("class", [])
        for cl in classes:
            m = re.match(r"phase-(\d+)", cl)
            if m:
                phase = int(m.group(1))
                if max_phase is None or phase > max_phase:
                    max_phase = phase
                break

    return max_phase, trial_count or None


def explode_target_types(
    therapeutic_id: str,
    target_type_field: Optional[str],
    therapy_type_field: Optional[str],
    mechanism_text: Optional[str],
) -> List[Dict]:
    """
    Convert comma-separated Target Type into rows for therapeutics_targets.csv.
    """
    if not target_type_field:
        return []

    raw_targets = [t.strip() for t in target_type_field.split(",") if t.strip()]

    def clean_target_label(t: str) -> str:
        # remove trailing "(timeline)" and extra spaces
        t = re.sub(r"\s*\(timeline\)\s*$", "", t)
        return t.strip()

    target_types = [clean_target_label(t) for t in raw_targets if clean_target_label(t)]

    if not target_types:
        return []

    ther_lower = (therapy_type_field or "").lower()

    if "immunotherapy" in ther_lower:
        action_type = "antibody"
    elif "dna/rna" in ther_lower or "rna" in ther_lower:
        action_type = "gene_therapy"
    elif "small molecule" in ther_lower:
        action_type = "small_molecule"
    elif "dietary" in ther_lower or "supplement" in ther_lower:
        action_type = "supplement"
    elif "procedural" in ther_lower or "device" in ther_lower:
        action_type = "device_or_procedure"
    else:
        action_type = None

    rows: List[Dict] = []
    for i, t in enumerate(target_types):
        rows.append(
            {
                "therapeutic_id": therapeutic_id,
                "target_name": t,
                "target_kind": "pathway_or_process",
                "action_type": action_type,
                "is_primary_target": True if i == 0 else False,
                "target_notes": summarise_mechanism(
                    mechanism_text, max_sentences=1
                )
                if mechanism_text
                else None,
            }
        )

    return rows


def extract_last_updated(article) -> Optional[str]:
    for p in article.find_all("p"):
        txt = clean_text(p.get_text(" ", strip=True))
        if txt.lower().startswith("last updated:"):
            return txt.split(":", 1)[1].strip()
    return None


# -------------------------------------------------------------------
# Main per-page parser
# -------------------------------------------------------------------

def parse_therapeutic_page(
    html: str, therapeutic_id: str, url: str
) -> Tuple[Dict, List[Dict], List[Dict]]:
    soup = BeautifulSoup(html, "lxml")
    article = find_article(soup)

    title_el = article.find("h1", class_="page-title") or article.find("h1")
    page_title = clean_text(title_el.get_text(strip=True)) if title_el else ""

    # --- Overview / key-value pairs ---
    overview_section = extract_overview_section(article)
    overview_block = section_text(overview_section) or ""
    overview_kv: Dict[str, str] = {}
    if overview_section is not None:
        overview_kv = parse_overview_kv_from_section(overview_section)
    elif overview_block:
        overview_kv = parse_overview_kv_from_text(overview_block)

    # --- Mechanism / background text ---
    bg_section = extract_background_section(article)
    mech_text = section_text(bg_section) or find_section_block(article, ["mechanism", "background"]) or ""
    mech_text = mech_text or None
    mech_summary = summarise_mechanism(mech_text)

    # --- Trial status from FDA string and timeline table ---
    fda_status = overview_kv.get("fda_status")
    phase_from_status, has_phase3_status, status_overall = infer_trial_phase_and_status(
        fda_status
    )
    phase_from_timeline, trial_count = infer_timeline_stats(article)

    # choose the best max phase
    phases = [p for p in [phase_from_status, phase_from_timeline] if p is not None]
    trial_phase_max = max(phases) if phases else None
    has_phase3 = (
        has_phase3_status
        if has_phase3_status is not None
        else (trial_phase_max is not None and trial_phase_max >= 3)
    )

    primary_indication = overview_kv.get("conditions")
    approved_for = overview_kv.get("approved_for")
    last_updated = extract_last_updated(article)

    # Clean therapy / target types (strip "(timeline)")
    raw_therapy_type = overview_kv.get("therapy_type")
    raw_target_type = overview_kv.get("target_type")

    detail_therapy_type_clean = strip_timeline_suffix(raw_therapy_type)

    if raw_target_type:
        parts = [p.strip() for p in raw_target_type.split(",") if p.strip()]
        clean_parts = [strip_timeline_suffix(p) for p in parts if strip_timeline_suffix(p)]
        detail_target_type_clean = ", ".join(clean_parts) if clean_parts else None
    else:
        detail_target_type_clean = None

    # ENTITY EXTRA (one row per therapeutic)
    entity_extra = {
        "therapeutic_id": therapeutic_id,
        "page_title": page_title,
        "detail_name": overview_kv.get("name"),
        "detail_synonyms": overview_kv.get("synonyms"),
        "detail_therapy_type": detail_therapy_type_clean,
        "detail_target_type": detail_target_type_clean,
        "detail_conditions": primary_indication,
        "detail_fda_status": fda_status,
        "detail_company": overview_kv.get("company"),
        "detail_approved_for": approved_for,
        "mechanism_summary": mech_summary,
        "mechanism_raw": mech_text,
        "overview_raw": overview_block,
        "status_overall": status_overall,
        "trial_phase_max": trial_phase_max,
        "has_phase3": has_phase3,
        "trial_count_timeline": trial_count,
        "last_updated": last_updated,
        "detail_url": url,
    }

    # TARGETS TABLE
    target_rows = explode_target_types(
        therapeutic_id=therapeutic_id,
        target_type_field=overview_kv.get("target_type"),
        therapy_type_field=overview_kv.get("therapy_type"),
        mechanism_text=mech_text,
    )

    # TRIAL SUMMARY TABLE (one row per therapeutic)
    trial_rows = [
        {
            "therapeutic_id": therapeutic_id,
            "indication": primary_indication,
            "trial_phase_max": trial_phase_max,
            "has_phase3": has_phase3,
            "status": status_overall,
            "trial_count": trial_count,
            "notes": fda_status,
        }
    ]

    return entity_extra, target_rows, trial_rows


# -------------------------------------------------------------------
# Top-level pipeline
# -------------------------------------------------------------------

def process_therapeutic_details() -> None:
    base_entities_path = PROCESSED_DIR / "therapeutics_entities.csv"
    if not base_entities_path.exists():
        raise FileNotFoundError(f"Base entities CSV not found: {base_entities_path}")

    base_df = pd.read_csv(base_entities_path)

    entity_extra_rows: List[Dict] = []
    all_target_rows: List[Dict] = []
    all_trial_rows: List[Dict] = []

    for _, row in base_df.iterrows():
        therapeutic_id = str(row["therapeutic_id"])
        url = str(row["url"])

        if not therapeutic_id or therapeutic_id.startswith("?"):
            continue

        html = read_or_download_html(therapeutic_id, url)
        if html is None:
            print(f"[WARN] Skipping {therapeutic_id}: could not fetch HTML")
            continue

        try:
            entity_extra, target_rows, trial_rows = parse_therapeutic_page(
                html, therapeutic_id, url
            )
        except Exception as e:
            print(f"[ERROR] Failed to parse {therapeutic_id}: {e}")
            continue

        entity_extra_rows.append(entity_extra)
        all_target_rows.extend(target_rows)
        all_trial_rows.extend(trial_rows)

    extra_df = pd.DataFrame(entity_extra_rows)
    targets_df = pd.DataFrame(all_target_rows)
    trials_df = pd.DataFrame(all_trial_rows)

    # merge extra fields into base entities
    if not extra_df.empty:
        enriched_df = base_df.merge(extra_df, on="therapeutic_id", how="left")
    else:
        print("[WARN] No therapeutic detail pages parsed; enriched entities will match base.")
        enriched_df = base_df.copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    enriched_path = PROCESSED_DIR / "therapeutics_entities_enriched.csv"
    enriched_df.to_csv(enriched_path, index=False)
    targets_df.to_csv(PROCESSED_DIR / "therapeutics_targets.csv", index=False)
    trials_df.to_csv(PROCESSED_DIR / "therapeutics_trials.csv", index=False)

    print(
        f"[THERAPEUTIC DETAILS] Wrote {len(enriched_df)} enriched entity rows -> {enriched_path.name}"
    )
    print(
        f"[THERAPEUTIC DETAILS] Wrote {len(targets_df)} target rows -> therapeutics_targets.csv"
    )
    print(
        f"[THERAPEUTIC DETAILS] Wrote {len(trials_df)} trial rows -> therapeutics_trials.csv"
    )


if __name__ == "__main__":
    process_therapeutic_details()