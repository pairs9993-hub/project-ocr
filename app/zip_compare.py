"""ZIP unpacking and pairing logic for the ZIP test mode.

- Reads images directly from the uploaded ZIP (in-memory, no extraction
  to disk required).
- Pairs test vs reference by *file stem* (basename without extension),
  case-insensitive. Path components inside the zip are ignored, only
  the leaf filename matters.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ZipImage:
    stem: str             # case-folded stem used for matching
    display_name: str     # original leaf filename (with extension)
    image: Image.Image    # already loaded (RGB)


@dataclass
class PairResult:
    matched: List[Tuple[ZipImage, ZipImage]]    # (test, reference)
    only_test: List[ZipImage]
    only_ref: List[ZipImage]


def load_zip_images(zip_bytes: bytes) -> List[ZipImage]:
    """Open a zip from bytes and return all image entries (recursively)."""
    out: List[ZipImage] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename).name
            if not name or name.startswith("."):
                continue
            ext = PurePosixPath(name).suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            try:
                with zf.open(info) as fh:
                    data = fh.read()
                img = Image.open(io.BytesIO(data))
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
            except (UnidentifiedImageError, OSError):
                continue
            stem = PurePosixPath(name).stem.casefold()
            out.append(ZipImage(stem=stem, display_name=name, image=img))
    # de-duplicate (last wins) so two files with same stem inside the zip
    # collapse into one — easier to reason about pairings.
    by_stem: Dict[str, ZipImage] = {}
    for zi in out:
        by_stem[zi.stem] = zi
    return list(by_stem.values())


def pair_by_stem(
    test_imgs: List[ZipImage], ref_imgs: List[ZipImage]
) -> PairResult:
    by_stem_test: Dict[str, ZipImage] = {zi.stem: zi for zi in test_imgs}
    by_stem_ref: Dict[str, ZipImage] = {zi.stem: zi for zi in ref_imgs}

    matched: List[Tuple[ZipImage, ZipImage]] = []
    only_test: List[ZipImage] = []
    only_ref: List[ZipImage] = []

    for stem, t in by_stem_test.items():
        r = by_stem_ref.get(stem)
        if r is not None:
            matched.append((t, r))
        else:
            only_test.append(t)
    for stem, r in by_stem_ref.items():
        if stem not in by_stem_test:
            only_ref.append(r)

    matched.sort(key=lambda p: p[0].display_name.lower())
    only_test.sort(key=lambda z: z.display_name.lower())
    only_ref.sort(key=lambda z: z.display_name.lower())
    return PairResult(matched=matched, only_test=only_test, only_ref=only_ref)
