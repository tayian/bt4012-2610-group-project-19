# Check that data/raw holds the two course files, byte for byte, before anything reads them.
#
# data/checksums.txt lists each file's SHA-256, line count (header included) and size.

import argparse
import hashlib
import sys
from pathlib import Path

DOWNLOAD = (
    "Copy the provided data files client_train.csv and invoice_train.csv into {raw_dir}/ "
    "(see Quick start in README.md)."
)


def read_checksums(path):
    # One file per line: name, sha256, lines, bytes; lines starting with # are comments.
    expected = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, sha256, lines, size = line.split()
            expected[name] = {"sha256": sha256, "lines": int(lines), "bytes": int(size)}
    return expected


def file_facts(path, chunk=1 << 20):
    # SHA-256, line count and size, read in one pass.
    digest, lines, size, last = hashlib.sha256(), 0, 0, b"\n"
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
            lines += block.count(b"\n")
            size += len(block)
            last = block[-1:]
    return {"sha256": digest.hexdigest(), "lines": lines + (last != b"\n"), "bytes": size}


def check_raw(raw_dir, checksums):
    # Return warnings for extra files; raise ValueError if a listed file is missing or differs.
    raw_dir = Path(raw_dir)
    expected = read_checksums(checksums)
    problems = []
    for name, facts in expected.items():
        path = raw_dir / name
        if not path.is_file():
            problems.append(f"{path} is missing.")
            continue
        # The size is checked first so a wrong file fails without hashing it.
        actual = {"bytes": path.stat().st_size}
        if actual["bytes"] == facts["bytes"]:
            actual = file_facts(path)
        wrong = [
            f"{key} {actual[key]} (expected {facts[key]})"
            for key in ("bytes", "lines", "sha256")
            if key in actual and actual[key] != facts[key]
        ]
        if wrong:
            problems.append(f"{path} is not the course file: {'; '.join(wrong)}.")
    if problems:
        raise ValueError("\n".join([*problems, DOWNLOAD.format(raw_dir=raw_dir)]))
    extra = sorted(
        p.name for p in raw_dir.iterdir() if p.name not in expected and p.name != ".DS_Store"
    )
    return [f"{raw_dir / name} is not used by the pipeline." for name in extra]


def main():
    parser = argparse.ArgumentParser(
        description="Check the raw course files against data/checksums.txt."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--checksums", type=Path, default=Path("data/checksums.txt"))
    args = parser.parse_args()
    try:
        warnings = check_raw(args.raw_dir, args.checksums)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"check_raw: {exc}\n")
    for warning in warnings:
        print(f"check_raw: warning: {warning}", file=sys.stderr)
    print(f"Raw files match {args.checksums}.")


if __name__ == "__main__":
    main()
