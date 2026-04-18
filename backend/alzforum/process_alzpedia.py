"""
Process AlzPedia HTML pages into structured CSVs:

- alzpedia_entities.csv  (one row per AlzPedia entity)
- alzpedia_sections.csv  (one row per entity x section)

Input:
- processed/alzpedia_index.csv          (built by download_alzforum.py)
- raw_html/alzpedia/<entity_id>.html    (downloaded pages)

This script is intentionally conservative and heuristic-based.
We can refine category / section detection later if needed.
"""

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

from bs4 import BeautifulSoup

# --------------------------
# PATHS
# --------------------------

ALZFORUM_ROOT = Path(__file__).resolve().parent
RAW_ALZPEDIA_DIR = ALZFORUM_ROOT / "raw_html" / "alzpedia"
PROCESSED_DIR = ALZFORUM_ROOT / "processed"

ALZPEDIA_INDEX_PATH = PROCESSED_DIR / "alzpedia_index.csv"
ALZPEDIA_ENTITIES_OUT = PROCESSED_DIR / "alzpedia_entities.csv"
ALZPEDIA_SECTIONS_OUT = PROCESSED_DIR / "alzpedia_sections.csv"


# --------------------------
# DATA MODELS
# --------------------------

@dataclass
class AlzPediaEntity:
    entity_id: str
    name: str
    url: str
    synonyms: str
    short_summary: str
    category: str
    has_function_section: bool
    has_pathology_section: bool
    has_genetics_section: bool
    has_therapeutics_section: bool


@dataclass
class AlzPediaSection:
    entity_id: str
    section_name: str        # normalized identifier (e.g. "overview")
    section_title: str       # human-readable heading
    section_order: int       # 1-based
    text: str


# --------------------------
# HELPERS
# --------------------------

def clean_text(text: Optional[str]) -> str:
    """Normalize whitespace in extracted text."""
    if not text:
        return ""
    return " ".join(text.split())


def guess_category(name: str, synonyms: str, overview_text: str) -> str:
    """
    Very simple heuristic for now.
    Most AlzPedia entries we pulled are genes/proteins/pathology.
    We'll tag them as 'protein_or_gene' and refine later if needed.
    """
    _ = " ".join([name or "", synonyms or "", overview_text or ""]).lower()

    # You could add smarter logic here later (e.g. look for 'gene', 'protein', 'pathology')
    return "protein_or_gene"


def extract_synonyms(soup: BeautifulSoup) -> str:
    """
    Extract the 'Synonyms' line from the intro-text-synonyms block, if present.
    Example HTML:
      <div class="intro-text-synonyms">
        <p class="snapshot"><strong>Synonyms: </strong>ADAM-10, AD10, ...</p>
      </div>
    """
    block = soup.find("div", class_="intro-text-synonyms")
    if not block:
        return ""

    p = block.find("p", class_="snapshot") or block.find("p")
    if not p:
        return ""

    text = p.get_text(" ", strip=True)
    # Remove leading "Synonyms:" if present
    if text.lower().startswith("synonyms"):
        # split on ':' once
        parts = text.split(":", 1)
        if len(parts) == 2:
            return clean_text(parts[1])
    return clean_text(text)


def extract_sections(
    soup: BeautifulSoup, entity_id: str
) -> Tuple[List[AlzPediaSection], dict]:
    """
    Extract sections from the main content column.

    Returns:
      - list of AlzPediaSection rows
      - section_presence dict to drive boolean flags
    """
    sections_out: List[AlzPediaSection] = []

    primary_div = soup.find("div", class_="primary")
    if not primary_div:
        return sections_out, {}

    sections = primary_div.find_all("section", recursive=False)
    if not sections:
        # Fallback: some pages might nest sections deeper
        sections = primary_div.find_all("section")

    section_presence = {
        "function": False,
        "pathology": False,
        "genetics": False,
        "therapeutics": False,
    }

    for order, sec in enumerate(sections, start=1):
        sec_id = (sec.get("id") or "").strip()
        # Title from pane-title heading if present
        heading = sec.find(["h1", "h2", "h3"], class_="pane-title")
        if heading is None:
            # Sometimes the section has no explicit pane-title
            heading = sec.find(["h1", "h2", "h3"])

        if heading:
            section_title = clean_text(heading.get_text(" ", strip=True))
        elif sec_id:
            section_title = sec_id.replace("-", " ").title()
        else:
            section_title = f"Section {order}"

        # Remove headings from text extraction to avoid duplication
        for h in sec.find_all(["h1", "h2", "h3"]):
            h.extract()

        text = clean_text(sec.get_text(" ", strip=True))

        # Normalized name: prefer id if present
        if sec_id:
            section_name = sec_id.lower()
        else:
            section_name = section_title.lower().replace(" ", "_")

        sections_out.append(
            AlzPediaSection(
                entity_id=entity_id,
                section_name=section_name,
                section_title=section_title,
                section_order=order,
                text=text,
            )
        )

        lower_combo = f"{sec_id} {section_title}".lower()
        if "function" in lower_combo:
            section_presence["function"] = True
        if "patholog" in lower_combo:
            section_presence["pathology"] = True
        if "genetic" in lower_combo:
            section_presence["genetics"] = True
        if "therapeutic" in lower_combo:
            section_presence["therapeutics"] = True

    return sections_out, section_presence


def parse_alzpedia_html(
    html: str, entity_id: str, url: str, index_title: str
) -> Tuple[AlzPediaEntity, List[AlzPediaSection]]:
    """
    Parse a single AlzPedia HTML page into:
      - one AlzPediaEntity
      - many AlzPediaSection
    """
    soup = BeautifulSoup(html, "lxml")

    # Name: <h1 class="entry-title">ADAM10</h1>
    name_el = soup.find("h1", class_="entry-title")
    if name_el:
        name = clean_text(name_el.get_text(" ", strip=True))
    else:
        name = index_title  # fallback

    synonyms = extract_synonyms(soup)

    # Sections & presence flags
    section_rows, presence_flags = extract_sections(soup, entity_id)

    # Short summary: first paragraph of the overview section, if present
    overview_text = ""
    overview_section = None
    for sec in section_rows:
        if sec.section_name == "overview":
            overview_section = sec
            break
    if overview_section:
        overview_text = overview_section.text

    short_summary = ""
    if overview_text:
        # Take only the first sentence or first ~400 chars as a compact summary
        # Simple heuristic: split on '. '
        parts = overview_text.split(". ")
        if parts:
            first_sentence = parts[0].strip()
            # Add back a period if it looks like a complete sentence
            if not first_sentence.endswith("."):
                first_sentence += "."
            short_summary = first_sentence

    category = guess_category(name, synonyms, overview_text)

    entity = AlzPediaEntity(
        entity_id=entity_id,
        name=name,
        url=url,
        synonyms=synonyms,
        short_summary=short_summary,
        category=category,
        has_function_section=presence_flags.get("function", False),
        has_pathology_section=presence_flags.get("pathology", False),
        has_genetics_section=presence_flags.get("genetics", False),
        has_therapeutics_section=presence_flags.get("therapeutics", False),
    )

    return entity, section_rows


# --------------------------
# MAIN PIPELINE
# --------------------------

def load_index(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_entities(rows: List[AlzPediaEntity], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("[WARN] No AlzPedia entities parsed; nothing to write.")
        return

    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"[ALZPEDIA] Wrote {len(rows)} entities to {path}")


def write_sections(rows: List[AlzPediaSection], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("[WARN] No AlzPedia sections parsed; nothing to write.")
        return

    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"[ALZPEDIA] Wrote {len(rows)} sections to {path}")


def main():
    print("=== Alzheimer’s KG – Process AlzPedia ===")

    index_rows = load_index(ALZPEDIA_INDEX_PATH)
    print(f"[ALZPEDIA] Loaded {len(index_rows)} index rows from {ALZPEDIA_INDEX_PATH}")

    all_entities: List[AlzPediaEntity] = []
    all_sections: List[AlzPediaSection] = []

    for row in index_rows:
        entity_id = row["entity_id"]
        url = row.get("url", "")
        title = row.get("title", "")

        html_path = RAW_ALZPEDIA_DIR / f"{entity_id}.html"
        if not html_path.exists():
            print(f"[ALZPEDIA] WARNING: HTML file not found for {entity_id}: {html_path}")
            continue

        html = html_path.read_text(encoding="utf-8")
        try:
            entity, sections = parse_alzpedia_html(html, entity_id, url, title)
            all_entities.append(entity)
            all_sections.extend(sections)
        except Exception as e:
            print(f"[ALZPEDIA] ERROR parsing {entity_id}: {e}")

    write_entities(all_entities, ALZPEDIA_ENTITIES_OUT)
    write_sections(all_sections, ALZPEDIA_SECTIONS_OUT)

    print("=== DONE (AlzPedia) ===")


if __name__ == "__main__":
    main()