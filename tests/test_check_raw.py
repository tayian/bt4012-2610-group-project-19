# The raw file check on small temporary files (the real 344 MB file is never hashed here).

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.data.check_raw import check_raw, file_facts

ROOT = Path(__file__).resolve().parents[1]
# The invoice file has no final newline, so its last line still counts.
CONTENT = {
    "client_train.csv": (b"client_id,target\nA,0\nB,1\n", 3),
    "invoice_train.csv": (b"client_id,months_number\nA,4", 2),
}


class CheckRawTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.raw = Path(self.temporary.name) / "raw"
        self.raw.mkdir()
        lines = []
        for name, (data, n_lines) in CONTENT.items():
            (self.raw / name).write_bytes(data)
            lines.append(f"{name} {hashlib.sha256(data).hexdigest()} {n_lines} {len(data)}")
        self.checksums = Path(self.temporary.name) / "checksums.txt"
        self.checksums.write_text("# file sha256 lines bytes\n" + "\n".join(lines) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_matching_files_pass(self):
        self.assertEqual(check_raw(self.raw, self.checksums), [])
        for name, (_, n_lines) in CONTENT.items():
            self.assertEqual(file_facts(self.raw / name)["lines"], n_lines)

    def test_missing_file_says_where_to_put_it(self):
        (self.raw / "invoice_train.csv").unlink()
        with self.assertRaisesRegex(
            ValueError, "invoice_train.csv is missing(.|\n)*Copy the provided data files"
        ):
            check_raw(self.raw, self.checksums)

    def test_same_size_different_content_fails_on_hash(self):
        (self.raw / "client_train.csv").write_bytes(
            CONTENT["client_train.csv"][0].replace(b"B,1", b"B,0")
        )
        with self.assertRaisesRegex(ValueError, "client_train.csv is not the course file: sha256"):
            check_raw(self.raw, self.checksums)

    def test_different_size_fails(self):
        (self.raw / "client_train.csv").write_bytes(b"client_id,target\n")
        with self.assertRaisesRegex(ValueError, "bytes 17"):
            check_raw(self.raw, self.checksums)

    def test_extra_files_only_warn(self):
        (self.raw / "client_test.csv").write_text("x\n")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.data.check_raw",
                "--raw-dir",
                str(self.raw),
                "--checksums",
                str(self.checksums),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("client_test.csv is not used", result.stderr)

    def test_cli_exits_2_on_mismatch(self):
        (self.raw / "client_train.csv").unlink()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.data.check_raw",
                "--raw-dir",
                str(self.raw),
                "--checksums",
                str(self.checksums),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("is missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
