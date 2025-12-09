# ontology/download_ontologies.py

import os
from pathlib import Path
from typing import Dict
import requests
from tqdm import tqdm


# ---------- CONFIG ----------

# Root folder for ontology stuff (relative to project root)
ONTOLOGY_ROOT = Path(__file__).resolve().parent

RAW_DIR = ONTOLOGY_ROOT / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Ontology download URLs (OWL or similar)
# These purls are standard OBO/GO/ChEBI/MONDO/HPO/PRO endpoints.
ONTOLOGY_SOURCES: Dict[str, str] = {
    # Disease ontology (Mondo)  [oai_citation:0‡obofoundry.org](https://obofoundry.org/ontology/mondo.html?utm_source=chatgpt.com)
    "mondo.owl": "http://purl.obolibrary.org/obo/mondo.owl",

    # Human Phenotype Ontology (HPO)  [oai_citation:1‡ontobee.org](https://ontobee.org/ontology/hp?utm_source=chatgpt.com)
    "hp.owl": "http://purl.obolibrary.org/obo/hp.owl",

    # Gene Ontology - basic acyclic version (we'll later subset)  [oai_citation:2‡Gene Ontology Resource](https://geneontology.org/docs/download-ontology/?utm_source=chatgpt.com)
    "go-basic.owl": "http://purl.obolibrary.org/obo/go/go-basic.owl",

    # Protein Ontology (PRO)  [oai_citation:3‡obofoundry.org](https://obofoundry.org/ontology/pr.html?utm_source=chatgpt.com)
    "pr.owl": "http://purl.obolibrary.org/obo/pr.owl",

    # ChEBI ontology (FULL variant)  [oai_citation:4‡obofoundry.org](https://obofoundry.org/ontology/chebi.html?utm_source=chatgpt.com)
    "chebi.owl": "http://purl.obolibrary.org/obo/chebi.owl",
}

# HGNC complete gene set (TSV)
# NOTE: HGNC moved downloads to a Google Storage bucket. The older FTP URL
# now 404s; use the GCS link below. If it fails, visit
# https://www.genenames.org/download/statistics-and-files/ and download
# "Complete HGNC approved dataset (TXT)" manually.  [oai_citation:5‡genenames.org](https://www.genenames.org/download/statistics-and-files/?utm_source=chatgpt.com)
HGNC_FILENAME = "hgnc_complete_set.txt"
HGNC_URL_DEFAULT = (
    "https://storage.googleapis.com/public-download-files/"
    "hgnc/tsv/tsv/hgnc_complete_set.txt"
)


# ---------- HELPERS ----------

def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """
    Stream-download a file from `url` to `dest_path` with a progress bar.
    If the file already exists, it will be skipped.
    """
    if dest_path.exists():
        print(f"[SKIP] {dest_path.name} already exists, skipping download.")
        return

    print(f"[INFO] Downloading {url} -> {dest_path}")

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))

            # Progress bar
            progress = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest_path.name,
            )

            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        progress.update(len(chunk))

            progress.close()

        print(f"[OK] Saved to {dest_path}")

    except requests.RequestException as e:
        print(f"[ERROR] Failed to download {url}")
        print(f"        {e}")


def download_ontologies() -> None:
    """Download MONDO, HPO, GO, PRO, ChEBI into ontology/raw/."""
    print(f"[INFO] Raw ontology directory: {RAW_DIR}")
    for filename, url in ONTOLOGY_SOURCES.items():
        dest = RAW_DIR / filename
        download_file(url, dest)


def download_hgnc_gene_table(url: str = HGNC_URL_DEFAULT) -> None:
    """
    Download HGNC complete gene dataset as TSV into ontology/raw/.
    URL may change over time; if this fails, check the HGNC downloads page.
    """
    dest = RAW_DIR / HGNC_FILENAME
    print(f"[INFO] Attempting to download HGNC gene table from:\n       {url}")
    download_file(url, dest)


# ---------- MAIN ----------

def main():
    print("=== Alzheimer’s KG – Ontology Download Script ===")
    print()

    # 1) Ontologies (MONDO, HPO, GO, PRO, ChEBI)
    download_ontologies()

    # 2) HGNC gene table (optional – may need manual refresh if URL changes)
    download_hgnc_gene_table()

    print()
    print("[DONE] Ontology + HGNC downloads complete.")
    print(f"       Files are in: {RAW_DIR}")


if __name__ == "__main__":
    main()