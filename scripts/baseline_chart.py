"""Render a clean CER-by-script bar chart and a Unicode-aware example panel."""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_validator.evaluate import script_of


PROJECT = Path(__file__).resolve().parent.parent
ART = PROJECT / "artifacts/baseline"
TEST = PROJECT / "dataset/test"


# Unicode-capable font candidates (Windows + cross-platform)
UNICODE_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(size: int) -> ImageFont.FreeTypeFont:
    for p in UNICODE_FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


# ---------- bar chart ----------


def cer_by_script(items):
    buckets = {}
    for it in items:
        buckets.setdefault(script_of(it["language"]), []).append(it["cer"])
    out = []
    for sc, vs in buckets.items():
        out.append((sc, len(vs), sum(vs) / len(vs)))
    out.sort(key=lambda r: -r[2])
    return out


def render_bar_chart(rows, title: str, out_path: Path,
                     width: int = 820, row_h: int = 40, pad: int = 24):
    height = pad * 2 + row_h * len(rows) + 60
    img = Image.new("RGB", (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    fnt = find_font(16)
    fnt_big = find_font(20)
    fnt_small = find_font(13)

    draw.text((pad, pad), title, font=fnt_big, fill=(240, 240, 240))
    label_x = pad
    bar_x = pad + 220
    bar_w_max = width - bar_x - 80
    y = pad + 50

    for sc, n, mc in rows:
        # label
        draw.text((label_x, y + 8), f"{sc} (n={n})", font=fnt, fill=(200, 200, 200))
        # bar background
        draw.rectangle((bar_x, y + 4, bar_x + bar_w_max, y + row_h - 4),
                       fill=(50, 50, 56))
        # bar fill — color codes severity
        if mc < 0.20:
            col = (90, 170, 90)
        elif mc < 0.50:
            col = (220, 180, 60)
        else:
            col = (220, 90, 70)
        bw = int(bar_w_max * mc)
        draw.rectangle((bar_x, y + 4, bar_x + bw, y + row_h - 4), fill=col)
        # value
        draw.text((bar_x + bw + 8, y + 8), f"{mc*100:.1f}%",
                  font=fnt, fill=(240, 240, 240))
        y += row_h

    # legend
    leg_y = y + 8
    draw.text((pad, leg_y), "green<20%  yellow 20–50%  red>50%",
              font=fnt_small, fill=(160, 160, 160))
    img.save(out_path)


# ---------- Unicode-aware example panel ----------


def render_example(item, target_w: int = 600) -> Image.Image:
    img_path = TEST / item["image_path"]
    bgr = Image.open(img_path).convert("RGB")
    w, h = bgr.size
    scale = target_w / w
    bgr = bgr.resize((target_w, int(h * scale)))

    fnt = find_font(15)
    fnt_h = find_font(13)

    header_h = 22
    text_h = 80
    panel = Image.new("RGB", (target_w, bgr.height + header_h + text_h),
                      (24, 24, 28))
    panel.paste(bgr, (0, header_h))
    d = ImageDraw.Draw(panel)
    d.rectangle((0, 0, target_w, header_h), fill=(40, 40, 48))
    head = (f"{Path(item['image_path']).name}   lang={item['language']}  "
            f"bg={item['background']}  CER={item['cer']*100:.1f}%")
    d.text((6, 4), head, font=fnt_h, fill=(220, 220, 220))

    ty = header_h + bgr.height + 4
    ref = "REF: " + item["ref"].replace("\n", " | ")
    hyp = "HYP: " + (item["hyp"] or "<empty>").replace("\n", " | ")
    d.text((6, ty), ref[:140], font=fnt, fill=(160, 220, 160))
    d.text((6, ty + 36), hyp[:140], font=fnt, fill=(160, 200, 240))
    return panel


def render_per_script_panel(items, out_path: Path):
    # one example per script: pick the median-CER sample (representative)
    by_script = {}
    for it in items:
        sc = script_of(it["language"])
        by_script.setdefault(sc, []).append(it)
    picks = []
    for sc, vs in sorted(by_script.items()):
        vs_sorted = sorted(vs, key=lambda x: x["cer"])
        picks.append((sc, vs_sorted[len(vs_sorted) // 2]))

    panels = [render_example(it) for _, it in picks]
    cols = 2
    pw = max(p.width for p in panels)
    ph = max(p.height for p in panels)
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", (pw * cols + 12 * (cols - 1),
                               ph * rows + 12 * (rows - 1)),
                       (10, 10, 12))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        canvas.paste(p, (c * (pw + 12), r * (ph + 12)))
    canvas.save(out_path)


def main():
    raw = load(ART / "raw_results.jsonl")
    rows = cer_by_script(raw)
    render_bar_chart(rows, "Baseline CER by script (raw, RapidOCR pretrained, n=1500)",
                     ART / "chart_cer_by_script.png")
    render_per_script_panel(raw, ART / "examples_unicode.png")
    print(f"wrote {ART / 'chart_cer_by_script.png'}")
    print(f"wrote {ART / 'examples_unicode.png'}")


if __name__ == "__main__":
    main()
