"""Configuration and confined filesystem access."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def confined(root: Path, relative: str) -> Path:
    """Resolve a relative path, rejecting traversal, absolute paths and symlink escapes."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative or "\\" in relative:
        raise ValueError("Unsafe relative path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Path escapes its authorized root")
    return resolved


@dataclass(frozen=True)
class Settings:
    repo: Path = ROOT
    private: Path = ROOT.parent / "week4_private_human_review"

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo", self.repo.resolve())
        object.__setattr__(self, "private", self.private.resolve())
        if self.private.is_relative_to(self.repo) and not self.private.is_relative_to(self.repo / ".tools"):
            raise ValueError("Private data cannot be written into repository/source GT directories")

    @classmethod
    def load(cls) -> "Settings":
        config = os.environ.get("GT_STUDIO_CONFIG")
        values = json.loads(Path(config).read_text(encoding="utf-8")) if config else {}
        result = cls(Path(values.get("repo_root", ROOT)).resolve(),
                     Path(os.environ.get("GT_STUDIO_PRIVATE_DIR", values.get("private_dir", ROOT.parent / "week4_private_human_review"))).resolve())
        if result.private.is_relative_to(result.repo) and not result.private.is_relative_to(result.repo / ".tools"):
            raise ValueError("Private review data must be outside the repository (or .tools for tests)")
        return result

    @property
    def ffmpeg(self) -> str | None:
        return shutil.which("ffmpeg") or self._portable("ffmpeg")

    @property
    def ffprobe(self) -> str | None:
        return shutil.which("ffprobe") or self._portable("ffprobe")

    def _portable(self, name: str) -> str | None:
        path = self.repo / ".tools/ffmpeg" / (name + (".exe" if os.name == "nt" else ""))
        return str(path) if path.is_file() else None
