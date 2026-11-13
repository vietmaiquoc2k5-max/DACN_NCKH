from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from apply_fix import OLD, NEW, apply

class TestFix(unittest.TestCase):
    def test_filter(self) -> None:
        rows = [{'kind':'negative','available':True},{'kind':'negative','available':False},{'kind':'positive','available':True}]
        self.assertEqual(len([r for r in rows if r['kind']=='negative' and r.get('available') is True]), 1)
        self.assertEqual(len(rows), 3)

    def test_patch_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'ground_truth_studio' / 'app.py'
            f.parent.mkdir()
            f.write_text('def example():\n'+OLD+'\n',encoding='utf-8')
            apply(Path(d))
            self.assertIn(NEW, f.read_text(encoding='utf-8'))
            before = f.read_bytes()
            apply(Path(d))
            self.assertEqual(before, f.read_bytes())
            self.assertTrue(f.with_name('app.py.pre_catalog_filter.bak').is_file())

    def test_guard(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'ground_truth_studio' / 'app.py'
            f.parent.mkdir()
            f.write_text('pass\n',encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'BLOCKED_VERSION_MISMATCH'):
                apply(Path(d))
            self.assertEqual(f.read_text(encoding='utf-8'),'pass\n')

if __name__ == '__main__':
    unittest.main()
