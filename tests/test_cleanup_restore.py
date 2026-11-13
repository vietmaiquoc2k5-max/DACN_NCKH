"""Restore archives without overwriting local bytes or escaping the chosen root."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.restore_cleanup_archive import restore


class TestCleanupRestore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "fixture.zip"
        self.manifest = self.root / "manifest.json"
        self.target = self.root / "restored"
        self.make_archive("runs/fixture/raw.stderr", b"")

    def make_archive(self, name, content):
        with zipfile.ZipFile(self.archive, "w") as z:
            z.writestr(name, content)
        self.manifest.write_text(json.dumps({"archive": {"sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(), "members": 1},
            "raw_artifacts": [{"archive_member": name, "sha256": hashlib.sha256(content).hexdigest()}]}))

    def test_dry_run_and_restore_empty_evidence(self):
        result = restore(self.archive, self.manifest, self.target)
        self.assertEqual(result["missing"], 1)
        self.assertFalse(self.target.exists())
        result = restore(self.archive, self.manifest, self.target, apply=True)
        self.assertEqual(result["restored"], 1)
        self.assertEqual((self.target / "runs/fixture/raw.stderr").read_bytes(), b"")
        self.assertEqual(restore(self.archive, self.manifest, self.target)["already_identical"], 1)

    def test_different_existing_bytes_never_overwritten(self):
        path = self.target / "runs/fixture/raw.stderr"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"unique local data")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            restore(self.archive, self.manifest, self.target, apply=True)
        self.assertEqual(path.read_bytes(), b"unique local data")

    def test_changed_archive_rejected(self):
        self.archive.write_bytes(b"changed archive")
        with self.assertRaisesRegex(ValueError, "Archive SHA256 mismatch"):
            restore(self.archive, self.manifest, self.target, apply=True)
        self.assertFalse(self.target.exists())

    def test_traversal_rejected_before_writes(self):
        self.make_archive("../outside.txt", b"payload")
        with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
            restore(self.archive, self.manifest, self.target, apply=True)
        self.assertFalse((self.root / "outside.txt").exists())
