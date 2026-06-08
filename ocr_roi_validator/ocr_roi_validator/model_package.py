from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class OCRModelPackage:
    root: Path
    detector_model: Path
    dictionary: Path
    recognizers: Dict[str, Path]
    preprocess: Dict[str, Any]


class ModelPackageError(Exception):
    pass


REQUIRED_PREPROCESS_KEYS = {
    "det_limit_type",
    "det_limit_side_len",
    "det_mean",
    "det_std",
    "det_box_thresh",
    "det_unclip_ratio",
    "det_donot_use_dilation",
    "use_cls",
}


def _resolve_file(root: Path, rel_path: str, field_name: str) -> Path:
    path = (root / rel_path).resolve()
    if not path.exists() or not path.is_file():
        raise ModelPackageError(f"{field_name} file not found: {path}")
    return path


def load_model_package(package_dir: Path) -> OCRModelPackage:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise ModelPackageError(f"manifest.json not found in {package_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    detector_rel = manifest.get("detector_model")
    dictionary_rel = manifest.get("dictionary")
    recognizers_raw = manifest.get("recognizers")
    preprocess = manifest.get("preprocess", {})

    if not detector_rel or not dictionary_rel or not isinstance(recognizers_raw, dict):
        raise ModelPackageError("manifest.json must include detector_model, dictionary, recognizers")

    missing = REQUIRED_PREPROCESS_KEYS - set(preprocess.keys())
    if missing:
        raise ModelPackageError(f"preprocess missing keys: {sorted(missing)}")

    detector_model = _resolve_file(package_dir, detector_rel, "detector_model")
    dictionary = _resolve_file(package_dir, dictionary_rel, "dictionary")

    recognizers: Dict[str, Path] = {}
    for lang, rel in recognizers_raw.items():
        recognizers[lang] = _resolve_file(package_dir, rel, f"recognizers.{lang}")

    for required_lang in ("en_es", "fr", "zh"):
        if required_lang not in recognizers:
            raise ModelPackageError(f"recognizers must include: {required_lang}")

    return OCRModelPackage(
        root=package_dir,
        detector_model=detector_model,
        dictionary=dictionary,
        recognizers=recognizers,
        preprocess=preprocess,
    )
