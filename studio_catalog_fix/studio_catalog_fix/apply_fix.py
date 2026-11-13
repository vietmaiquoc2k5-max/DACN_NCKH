"""Minimal fail-safe UI fix: hide missing source entries from review dropdown."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

OLD = '        available = [row for row in repository.videos() if row["kind"] == kind]'
NEW = '''        # Preserve all catalog entries in SQLite; show only accessible source files.
        # This is a UI filter, not scientific dataset eligibility validation.
        all_kind = [row for row in repository.videos() if row["kind"] == kind]
        available = [row for row in all_kind if row.get("available") is True]
        missing_count = len(all_kind) - len(available)
        st.caption(f"Catalog: {len(all_kind)} mục {kind}; {len(available)} có file; {missing_count} thiếu nguồn (ẩn khỏi danh sách).")'''


def apply(repo: Path) -> None:
    path = repo / 'ground_truth_studio' / 'app.py'
    content = path.read_text(encoding='utf-8')
    if NEW in content:
        print('ALREADY_APPLIED')
        return
    if content.count(OLD) != 1:
        raise RuntimeError('BLOCKED_VERSION_MISMATCH: expected line does not match; no writes')
    new_content = content.replace(OLD, NEW, 1)
    compile(new_content, str(path), 'exec')
    backup = path.with_name('app.py.pre_catalog_filter.bak')
    if backup.exists() and backup.read_bytes() != path.read_bytes():
        raise RuntimeError('BLOCKED_BACKUP_CONFLICT: backup exists with different bytes')
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(new_content, encoding='utf-8', newline='')
    print(f'PATCH_APPLIED: {path}')
    print('This changes UI only; the manifest, database and video files were not modified.')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    apply(parser.parse_args().repo.resolve())


if __name__ == '__main__':
    main()
