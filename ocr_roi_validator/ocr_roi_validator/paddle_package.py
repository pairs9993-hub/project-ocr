from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class PaddleModelPackage:
    root: Path
    detector_model_dir: Path
    dictionary: Path
    recognizer_model_dirs: Dict[str, Path]


class PaddlePackageError(Exception):
    pass


def _ensure_infer_dir(path: Path, field_name: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise PaddlePackageError(f"{field_name} directory not found: {path}")
    model = path / "inference.pdmodel"
    params = path / "inference.pdiparams"
    if not model.exists() or not params.exists():
        raise PaddlePackageError(f"{field_name} must contain inference.pdmodel and inference.pdiparams: {path}")
    return path


def _resolve_file(root: Path, rel_path: str, field_name: str) -> Path:
    path = (root / rel_path).resolve()
    if not path.exists() or not path.is_file():
        raise PaddlePackageError(f"{field_name} file not found: {path}")
    return path


def load_paddle_model_package(package_dir: Path) -> PaddleModelPackage:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise PaddlePackageError(f"manifest.json not found in {package_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    det_rel = manifest.get("detector_model_dir")
    dict_rel = manifest.get("dictionary")
    rec_raw = manifest.get("recognizer_model_dirs")

    if not det_rel or not dict_rel or not isinstance(rec_raw, dict):
        raise PaddlePackageError(
            "manifest.json must include detector_model_dir, dictionary, recognizer_model_dirs"
        )

    detector_model_dir = _ensure_infer_dir((package_dir / det_rel).resolve(), "detector_model_dir")
    dictionary = _resolve_file(package_dir, dict_rel, "dictionary")

    recognizer_model_dirs: Dict[str, Path] = {}
    for lang, rel in rec_raw.items():
        recognizer_model_dirs[lang] = _ensure_infer_dir((package_dir / rel).resolve(), f"recognizer_model_dirs.{lang}")

    return PaddleModelPackage(
        root=package_dir,
        detector_model_dir=detector_model_dir,
        dictionary=dictionary,
        recognizer_model_dirs=recognizer_model_dirs,
    )
