from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONTS = (
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/calibri.ttf"),
)


@dataclass(frozen=True)
class TextPair:
    language: str
    category: str
    expected: str
    defective: str


PAIRS = (
    TextPair("fr", "over_accent", "Veuillez allumer l'eau.", "Véuillez allumer l'eau."),
    TextPair("fr", "missing_accent", "Vérifiez la pression.", "Verifiez la pression."),
    TextPair("fr", "missing_accent", "Arrivée d'eau", "Arrivee d'eau"),
    TextPair("fr", "missing_accent", "Déséquilibre détecté", "Desequilibre detecte"),
    TextPair("fr", "missing_accent", "Joint d'étanchéité", "Joint d'etancheite"),
    TextPair("fr", "missing_accent", "Température élevée", "Temperature elevee"),
    TextPair("fr", "i_l_confusion", "Tuyaux d'alimentation", "Tuyaux d'alimentatlon"),
    TextPair("fr", "i_l_confusion", "Cycle disponible", "Cycle disponlble"),
    TextPair("es", "missing_tilde", "MAÑ.", "MAN."),
    TextPair("es", "missing_accent", "Configuración", "Configuracion"),
    TextPair("es", "missing_accent", "Más ciclos", "Mas ciclos"),
    TextPair("es", "missing_accent", "Función de vapor", "Funcion de vapor"),
    TextPair("es", "missing_accent", "Lavado rápido", "Lavado rapido"),
    TextPair("es", "missing_tilde", "Añadir prendas", "Anadir prendas"),
)

TRAINING_PAIRS = (
    TextPair("fr", "over_accent", "Veuillez fermer la porte.", "Véuillez fermer la porte."),
    TextPair("fr", "over_accent", "Veuillez patienter.", "Véuillez patienter."),
    TextPair("fr", "missing_accent", "Sélectionnez un programme", "Selectionnez un programme"),
    TextPair("fr", "missing_accent", "Démarrage différé", "Demarrage differe"),
    TextPair("fr", "missing_accent", "Prélavage activé", "Prelavage active"),
    TextPair("fr", "missing_accent", "Rinçage supplémentaire", "Rincage supplementaire"),
    TextPair("fr", "missing_accent", "Nettoyage terminé", "Nettoyage termine"),
    TextPair("fr", "missing_accent", "Porte verrouillée", "Porte verrouillee"),
    TextPair("fr", "missing_accent", "Température recommandée", "Temperature recommandee"),
    TextPair("fr", "missing_accent", "Économie d'énergie", "Economie d'energie"),
    TextPair("fr", "missing_accent", "Vêtements délicats", "Vetements delicats"),
    TextPair("fr", "missing_accent", "Cycle téléchargé", "Cycle telecharge"),
    TextPair("fr", "missing_accent", "Essorage sélectionné", "Essorage selectionne"),
    TextPair("fr", "missing_accent", "Détection de charge", "Detection de charge"),
    TextPair("fr", "missing_accent", "Ajoutez le détergent", "Ajoutez le detergent"),
    TextPair("fr", "missing_accent", "Option désactivée", "Option desactivee"),
    TextPair("fr", "i_l_confusion", "Niveau de salissure", "Nlveau de salissure"),
    TextPair("fr", "i_l_confusion", "Cycle intensif", "Cycle lntensif"),
    TextPair("fr", "i_l_confusion", "Lavage quotidien", "Lavage quotldien"),
    TextPair("fr", "i_l_confusion", "Option disponible", "Option disponlble"),
)


def available_fonts() -> list[Path]:
    fonts = [path for path in DEFAULT_FONTS if path.exists()]
    if not fonts:
        raise FileNotFoundError("No supported Windows UI font was found")
    return fonts


def render_text(
    text: str,
    font_path: Path,
    font_size: int,
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
    blur_radius: float,
    vertical_offset: int,
) -> Image.Image:
    scale = 3
    width = 640
    height = 80
    image = Image.new("RGB", (width * scale, height * scale), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), font_size * scale)
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = max(12 * scale, (width * scale - text_width) // 2)
    y = (height * scale - text_height) // 2 - bounds[1] + vertical_offset * scale
    draw.text((x, y), text, font=font, fill=foreground)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    if blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paired UI OCR diacritic stress images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "diacritic_stress",
    )
    parser.add_argument("--variants", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=("evaluation", "training"), default="evaluation")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fonts = available_fonts()
    pairs = PAIRS if args.profile == "evaluation" else TRAINING_PAIRS
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"

    palettes = (
        ((240, 240, 240), (12, 16, 20)),
        ((224, 229, 235), (28, 34, 40)),
        ((35, 39, 44), (238, 240, 242)),
        ((210, 218, 224), (44, 50, 56)),
    )
    rows = []
    for pair_index, pair in enumerate(pairs):
        for variant in range(args.variants):
            font_path = rng.choice(fonts)
            font_size = rng.choice((13, 14, 15, 16))
            foreground, background = rng.choice(palettes)
            blur_radius = rng.choice((0.0, 0.0, 0.2, 0.35))
            vertical_offset = rng.choice((-1, 0, 0, 1))
            for state, visible_text in (("normal", pair.expected), ("defect", pair.defective)):
                filename = f"{pair.language}_{pair_index:02d}_{variant:03d}_{state}.png"
                image = render_text(
                    visible_text,
                    font_path,
                    font_size,
                    foreground,
                    background,
                    blur_radius,
                    vertical_offset,
                )
                image.save(image_dir / filename)
                rows.append(
                    {
                        "filename": filename,
                        "image_path": f"images/{filename}",
                        "language": pair.language,
                        "category": pair.category,
                        "state": state,
                        "visible_text": visible_text,
                        "expected": pair.expected,
                        "font": font_path.name,
                        "font_size": font_size,
                        "blur_radius": blur_radius,
                        "vertical_offset": vertical_offset,
                    }
                )

    with manifest_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"rows={len(rows)} pairs={len(pairs)} variants={args.variants} profile={args.profile}")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())