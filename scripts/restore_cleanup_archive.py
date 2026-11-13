"""Verify a cleanup archive and restore missing files without overwriting data."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    """Hash a file with bounded memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def restore(archive: Path, manifest: Path, target: Path, *, apply: bool = False) -> dict[str, Any]:
    """Verify archive/members, then restore only missing paths when explicitly applied.

    Args:
        archive: Verified local ZIP or an identical copy on another storage device.
        manifest: Storage manifest containing archive SHA and raw member identities.
        target: Repository root to restore beneath; paths must stay inside it.
        apply: Write missing files after validation; otherwise perform a dry run.

    Returns:
        Counts of identical existing files and missing files available to restore.

    Raises:
        ValueError: If checksums, member paths or existing file bytes conflict.
    """
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if digest(archive) != data["archive"]["sha256"]:
        raise ValueError("Archive SHA256 mismatch")
    root = target.resolve()
    raw_hashes = {row["archive_member"]: row["sha256"] for row in data["raw_artifacts"]}
    existing, missing = 0, []
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len(members) != data["archive"]["members"] or len({m.filename for m in members}) != len(members):
            raise ValueError("Archive member count/identity conflict")
        if z.testzip() is not None:
            raise ValueError("Archive CRC mismatch")
        for member in members:
            name = member.filename
            relative = Path(name)
            if member.is_dir() or relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
                raise ValueError("Unsafe archive path")
            destination = (root / relative).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Archive path escapes target")
            with z.open(member) as stream:
                hasher = hashlib.sha256()
                while block := stream.read(1024 * 1024):
                    hasher.update(block)
                member_hash = hasher.hexdigest()
            if name in raw_hashes and member_hash != raw_hashes[name]:
                raise ValueError("Raw evidence member SHA256 mismatch")
            if destination.exists():
                if not destination.is_file() or digest(destination) != member_hash:
                    raise ValueError("Existing destination differs; refusing overwrite: " + name)
                existing += 1
            else:
                missing.append((name, destination))
        # Validate every member and conflict before the first filesystem write.
        if apply:
            for name, destination in missing:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.resolve().is_relative_to(root):
                    raise ValueError("Target path changed during restore")
                with z.open(name) as stream, destination.open("xb") as output:
                    while block := stream.read(1024 * 1024):
                        output.write(block)
    return {"archive_verified": True, "members": len(members), "already_identical": existing,
            "missing": len(missing), "restored": len(missing) if apply else 0, "dry_run": not apply}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("docs/artifact_storage_manifest.json"))
    parser.add_argument("--target-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--apply", action="store_true", help="Restore missing files after all validation")
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, args.manifest, args.target_root, apply=args.apply)))


if __name__ == "__main__":
    main()
