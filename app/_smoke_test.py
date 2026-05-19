"""End-to-end smoke test (no Streamlit involved).

Picks a handful of images from dataset/test/, builds two in-memory zips
(test = images as-is; reference = same images mirrored to verify the
pairing+ocr+excel chain end to end), runs the full pipeline, and writes
a sample report to artifacts/app_smoke/report.xlsx.
"""

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from PIL import Image

from excel_report import ReportRow, build_workbook, workbook_to_bytes
from metrics import cer, verdict, wer
from ocr_runner import ocr_image
from zip_compare import load_zip_images, pair_by_stem


def make_zip(paths):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
    return buf.getvalue()


def main():
    test_dir = ROOT / "dataset" / "test" / "images"
    samples = sorted(test_dir.glob("*.png"))[:4]
    if not samples:
        print("no test images found")
        return 1
    print(f"using {len(samples)} samples")

    # test zip = first half; reference zip = same files (so OCR(test)==OCR(ref))
    test_zip_bytes = make_zip(samples)
    ref_zip_bytes = make_zip(samples)

    test_imgs = load_zip_images(test_zip_bytes)
    ref_imgs = load_zip_images(ref_zip_bytes)
    print(f"loaded test={len(test_imgs)} ref={len(ref_imgs)}")

    pair = pair_by_stem(test_imgs, ref_imgs)
    print(f"matched={len(pair.matched)} only_test={len(pair.only_test)} only_ref={len(pair.only_ref)}")

    rows = []
    for t_zi, r_zi in pair.matched:
        t = ocr_image(t_zi.image)
        r = ocr_image(r_zi.image)
        c_v = cer(r.text, t.text)
        w_v = wer(r.text, t.text)
        v = verdict(c_v)
        print(f"  {t_zi.display_name}  CER={c_v:.3f}  WER={w_v:.3f}  -> {v}")
        rows.append(
            ReportRow(
                filename=t_zi.display_name,
                test_image=t_zi.image,
                ref_image=r_zi.image,
                test_text=t.text,
                ref_text=r.text,
                cer_v=c_v,
                wer_v=w_v,
                verdict=v,
            )
        )

    wb = build_workbook(rows, [], [])
    out = ROOT / "artifacts" / "app_smoke" / "report.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(workbook_to_bytes(wb))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
