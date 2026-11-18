# Builds data/corpus/chunks.jsonl from the playbook, the two SOP versions, the code book and the
# precedent case cards, and checks every chunk carries REQUIRED_FIELDS.
# Run with `python run.py corpus`.

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.llm.corpus.case_cards import build_case_cards, case_card_to_chunk

HEADING_RE = re.compile(r"^###\s+\[([A-Za-z0-9_.-]+)\]\s*(.*)$")
META_LINE_RE = re.compile(r"^\*\s+\*\*Metadata:\*\*\s*(.+)$")
META_KV_RE = re.compile(r"`([^:`]+):\s*([^`]+)`")

REQUIRED_FIELDS = (
    "id",
    "text",
    "doc",
    "section",
    "version",
    "valid_from",
    "valid_to",
    "counter_type",
    "jurisdiction",
    "synthetic",
)


def _parse_meta_value(raw: str):
    val = raw.strip()
    if val in ("None", "null", "NULL"):
        return None
    if val == "true":
        return True
    if val == "false":
        return False
    return val


def parse_markdown_chunks(path: Path, default_doc: str) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    raw_sections: list[tuple[str, str, dict, list[str]]] = []

    curr_id: str | None = None
    curr_title: str = ""
    curr_meta: dict = {}
    curr_body: list[str] = []

    def flush():
        nonlocal curr_id, curr_title, curr_meta, curr_body
        if curr_id is not None:
            raw_sections.append((curr_id, curr_title, curr_meta, curr_body))
        curr_id = None
        curr_title = ""
        curr_meta = {}
        curr_body = []

    for line in lines:
        m_head = HEADING_RE.match(line)
        if m_head:
            flush()
            curr_id = m_head.group(1).strip()
            curr_title = m_head.group(2).strip()
            continue

        if curr_id is None:
            continue

        # A level-2 heading or horizontal rule ends the current chunk block.
        if line.startswith("## ") or line.strip() == "---":
            flush()
            continue

        m_meta = META_LINE_RE.match(line.strip())
        if m_meta and not curr_meta:
            for k, v in META_KV_RE.findall(m_meta.group(1)):
                curr_meta[k.strip()] = _parse_meta_value(v)
            continue

        curr_body.append(line)

    flush()

    chunks: list[dict] = []
    for cid, title, meta, body_lines in raw_sections:
        doc = str(meta.get("doc", default_doc))
        header = f"[{cid}] {title}".strip()
        body_text = "\n".join(body_lines).strip()
        full_text = f"{header}\n{body_text}".strip() if body_text else header

        # Per-document defaults, used when the metadata line does not set the field.
        if doc == "playbook":
            section, version, valid_from, synthetic = cid.lower(), "v1.0", "2005-01-01", False
        elif doc == "sop":
            # The clause number and version come from the ID, e.g. SOP-v2025-2.1.
            parts = cid.split("-", 2)
            section = parts[2] if len(parts) == 3 else cid
            version = parts[1] if len(parts) >= 2 else "v2025"
            valid_from, synthetic = "2025-01-01", True
        elif doc == "codebook":
            field = str(meta.get("field", ""))
            section, version, valid_from, synthetic = field, "v1.0", "2005-01-01", False
        else:
            raise ValueError(f"Unsupported doc type {doc!r} in {path}")

        chunk = {
            "id": cid,
            "text": full_text,
            "doc": doc,
            "section": str(meta.get("section", section)),
            "version": str(meta.get("version", version)),
            "valid_from": str(meta.get("valid_from", valid_from)),
            "valid_to": meta.get("valid_to", None),
            "counter_type": str(meta.get("counter_type", "both")),
            "jurisdiction": str(meta.get("jurisdiction", "TN")),
            "synthetic": bool(meta.get("synthetic", synthetic)),
        }
        if doc == "codebook":
            chunk["field"] = field
            chunk["value"] = str(meta.get("value", ""))
            chunk["assumed"] = bool(meta.get("assumed", False))
        chunks.append(chunk)

    return chunks


def validate_chunks(chunks: list[dict]) -> None:
    seen_ids: set[str] = set()
    for i, c in enumerate(chunks):
        missing = [k for k in REQUIRED_FIELDS if k not in c]
        if missing:
            raise ValueError(f"Chunk #{i} ({c.get('id')}) missing required fields: {missing}")
        cid = c["id"]
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"Chunk #{i} has invalid id: {cid!r}")
        if cid in seen_ids:
            raise ValueError(f"Duplicate chunk id in corpus: {cid}")
        seen_ids.add(cid)
        if not isinstance(c["text"], str) or not c["text"].startswith(f"[{cid}]"):
            raise ValueError(f"Chunk {cid} text must start with '[{cid}]'")
        if c["counter_type"] not in ("ELEC", "GAZ", "both"):
            raise ValueError(f"Chunk {cid} has invalid counter_type: {c['counter_type']!r}")
        if not isinstance(c["synthetic"], bool):
            raise ValueError(f"Chunk {cid} 'synthetic' field must be bool")


def build_corpus_chunks(
    source_dir: Path = Path("data/corpus/source_docs"),
    case_cards_path: Path = Path("data/corpus/case_cards.jsonl"),
    out_path: Path = Path("data/corpus/chunks.jsonl"),
    raw_clients_path: Path = Path("data/raw/client_train.csv"),
    split_dir: Path = Path("data/splits"),
    interim_dir: Path = Path("data/interim"),
    processed_dir: Path = Path("data/processed"),
    rebuild_cases: bool = False,
) -> list[dict]:
    playbook_chunks = parse_markdown_chunks(source_dir / "playbook.md", "playbook")
    sop_2023_chunks = parse_markdown_chunks(source_dir / "sop_v2023.md", "sop")
    sop_2025_chunks = parse_markdown_chunks(source_dir / "sop_v2025.md", "sop")
    codebook_chunks = parse_markdown_chunks(source_dir / "codebook.md", "codebook")

    if len(playbook_chunks) != 12:
        raise ValueError(f"Expected 12 playbook chunks, got {len(playbook_chunks)}")
    if len(sop_2023_chunks) != 13:
        raise ValueError(f"Expected 13 sop_v2023 chunks, got {len(sop_2023_chunks)}")
    if len(sop_2025_chunks) != 13:
        raise ValueError(f"Expected 13 sop_v2025 chunks, got {len(sop_2025_chunks)}")
    if len(codebook_chunks) != 82:
        raise ValueError(f"Expected 82 codebook chunks, got {len(codebook_chunks)}")

    if rebuild_cases or not case_cards_path.is_file():
        cards = build_case_cards(
            raw_clients_path=raw_clients_path,
            split_dir=split_dir,
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            out_path=case_cards_path,
        )
    else:
        cards = [
            json.loads(line)
            for line in case_cards_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    case_chunks = [case_card_to_chunk(card) for card in cards]
    all_chunks = playbook_chunks + sop_2023_chunks + sop_2025_chunks + codebook_chunks + case_chunks
    validate_chunks(all_chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in all_chunks),
        encoding="utf-8",
    )
    return all_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Build data/corpus/chunks.jsonl from playbook, SOPs, codebook, and case cards."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/corpus/source_docs"))
    parser.add_argument("--case-cards", type=Path, default=Path("data/corpus/case_cards.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/corpus/chunks.jsonl"))
    parser.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--rebuild-cases",
        action="store_true",
        help="Regenerate case_cards.jsonl before building chunks.jsonl",
    )
    args = parser.parse_args()
    try:
        chunks = build_corpus_chunks(
            source_dir=args.source_dir,
            case_cards_path=args.case_cards,
            out_path=args.out,
            raw_clients_path=args.clients,
            split_dir=args.splits,
            interim_dir=args.interim,
            processed_dir=args.processed,
            rebuild_cases=args.rebuild_cases,
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"build_chunks: {exc}\n")

    counts: dict[str, int] = {}
    for c in chunks:
        counts[c["doc"]] = counts.get(c["doc"], 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Wrote {len(chunks):,} chunks ({summary}) to {args.out}")


if __name__ == "__main__":
    main()
