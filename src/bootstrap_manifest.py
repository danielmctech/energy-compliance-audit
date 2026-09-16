"""
Rebuilds outputs/corpus_metadata.json from the PDFs on disk.

The manifest is a hand-maintained input to notebook 01, but it can be
regenerated deterministically: every PDF under data/raw/eu/en/<type>/ gets an
entry whose metadata fields are extracted from the document text itself
(see src/ground_truth.py). Set needs_review=true so notebook 01 flags each
entry for the metadata-completeness audit.

Usage:  python3 src/bootstrap_manifest.py
"""
import json, re, sys
from datetime import datetime
from pathlib import Path

def _repo_root() -> Path:
    cur = Path(__file__).resolve().parent
    for cand in [cur, *cur.parents]:
        if (cand / "notebooks").is_dir() and (cand / "data").is_dir():
            return cand
    return cur.parent

ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "src"))
import common as c  # noqa: E402
from parsing import ground_truth as GT  # noqa: E402

REQUIRED_FIELDS = [
    "filename", "folder_key", "instrument_type", "domain",
    "short_tag", "jurisdiction", "full_title", "url",
    "effective_date", "in_force_date", "applicability", "status",
    "compliance_risk_level", "key_articles_for_audit", "tags",
]

FOLDER_MAP = {
    "regulations":  "data/raw/eu/en/regulations",
    "directives":   "data/raw/eu/en/directives",
    "guidance":     "data/raw/eu/en/guidance",
    "network_codes": "data/raw/eu/en/network_codes",
}

# folder_key -> (instrument_type, domain hints)
DOMAIN = {
    "regulations":   "energy + cross-cutting",
    "directives":    "energy + cross-cutting",
    "guidance":      "energy market guidance",
    "network_codes": "ENTSO-E network codes",
}

KNOWN_TAGS = {
    "gdpr_2016_679": ["data-protection", "processor", "controller"],
    "dora_2022_2554": ["operational-resilience"],
    "nis2_dir_2022_2555": ["cybersecurity"],
    "eu_ai_act_2024_1689": ["ai", "high-risk"],
    "mica_2023_1114": ["capital-markets"],
    "remit_1227_2011": ["internal-market", "energy"],
    "elec_reg_2019_943": ["electricity", "market", "energy"],
    "elec_dir_2019_944": ["electricity", "market", "energy"],
    "em*": ["electricity", "energy"],
    "remit_ii_*": ["internal-market", "energy"],
    "entso*": ["energy"],
    "acer_*": ["remit", "energy"],
    "know_your_contract*": ["consumer", "energy"],
    "metering_data_2023_1162": ["metering", "data", "energy"],
    "eidas_*": ["eidas", "identity", "digital-services"],
    "data_act_2023_2854": ["data-act", "energy"],
    "dlt_pilot_2022_858": ["dlt", "gas", "energy"],
}

def _tags_for(stem: str, folder_key: str) -> list[str]:
    """best-effort tags from a small known list; always include the folder type"""
    base = {"network_codes": ["entso-e", "network-code"],
            "guidance": ["guidance"],
            "regulations": ["regulation"],
            "directives": ["directive"]}[folder_key]
    hits: list[str] = []
    for pat, tags in KNOWN_TAGS.items():
        name = pat.rstrip("*")
        if (pat.endswith("*") and stem.startswith(name)) or (not pat.endswith("*") and stem == pat):
            hits = tags
            break
    hits = [t for t in hits if t not in base]
    return base + hits

def build_entry(pdf: Path, folder_key: str) -> dict:
    stem = pdf.stem
    md = c.md_for_pdf(pdf)
    text = md.read_text() if md.exists() else (pdf.read_bytes()[:0].decode() or "")

    instrument = GT._instrument(text) or ""
    title = GT._extract_title_heuristic(text) or stem.replace("_", " ").title()
    arts = GT._extract_article_numbers(text)
    pub_date = GT._publication_date(text)
    risk = "high" if any(k in stem for k in ("dora", "nis2", "gdpr")) else "medium"

    # guidance / white papers often have no numbered articles; use the main
    # section headings as audit anchors in that case
    if not arts:
        arts = [h.strip().lstrip("# _").strip("*").strip()
                for h in GT._headings(text) if re.match(r"^[IVX]+\.\s", h)][:10]

    entry = {
        "filename": pdf.name,
        "folder_key": folder_key,
        "instrument_type": folder_key,
        "domain": DOMAIN[folder_key],
        "short_tag": stem.split("_")[0],
        "jurisdiction": "eu",
        "full_title": title,
        "url": "",  # source link is optional; not invented by the bootstrapper
        "effective_date": (pub_date or "")[8:] if pub_date and len(pub_date) >= 10 else "",
        "in_force_date": (pub_date or "")[8:] if pub_date and len(pub_date) >= 10 else "",
        "applicability": "direct",
        "status": "in_force",
        "compliance_risk_level": risk,
        "key_articles_for_audit": arts[:12],
        "tags": _tags_for(stem, folder_key),
        "instrument": instrument,
        "needs_review": True,
    }
    # enforce the required field set (add empty strings if a heuristic failed)
    for f in REQUIRED_FIELDS:
        entry.setdefault(f, "" if not isinstance(entry.get(f), (list, dict)) else [])
    return entry


def main():
    docs = []
    for folder_key, rel in FOLDER_MAP.items():
        d = ROOT / rel
        for pdf in sorted(d.glob("*.pdf")):
            docs.append(build_entry(pdf, folder_key))
    order = {k: i for i, k in enumerate(FOLDER_MAP)}
    docs.sort(key=lambda e: (order[e["folder_key"]], e["filename"]))

    manifest = {
        "_schema_version": "1.1",
        "_last_updated": datetime.now().isoformat(timespec="seconds"),
        "_generated_by": "src/bootstrap_manifest.py (deterministic rebuild from disk)",
        "folder_map": FOLDER_MAP,
        "documents": docs,
    }
    out = ROOT / "outputs" / "corpus_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"rebuilt manifest: {len(docs)} documents -> {out}")
    empty = [e["filename"] for e in docs if not e["full_title"] or e["full_title"] == e["filename"]]
    if empty:
        print(f"warning: no title found for {len(empty)} docs: {empty[:5]}")


if __name__ == "__main__":
    main()
