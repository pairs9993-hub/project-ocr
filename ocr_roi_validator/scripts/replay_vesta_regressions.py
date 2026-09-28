"""Replay saved TM and overlapping-word captures; does not drive live Vesta."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from ocr_roi_validator.automation_ocr import FrozenValue
from ocr_roi_validator.gui import OCRValidatorGUI
from ocr_roi_validator.ocr_engine import OCREngine
from ocr_roi_validator.roi_preprocess import RoiPreprocessConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    processor = OCRValidatorGUI.__new__(OCRValidatorGUI)
    processor.engine = OCREngine(use_rapid_default=True)
    settings = {'language_var': 'en_es', 'roi_margin_var': '8', 'min_roi_side_var': '160',
                'fast_long_roi_var': True, 'auto_upscale_var': True,
                'context_detect_var': False, 'context_margin_var': '0'}
    for key, value in settings.items():
        setattr(processor, key, FrozenValue(value))
    output = {'scope': 'saved_capture_replay_not_live_vesta',
              'assumptions': ['Rapid default engine', 'direct ROI path', RoiPreprocessConfig().as_dict()],
              'samples': []}
    for path in args.reports:
        report = json.loads(path.read_text(encoding='utf-8'))
        for tc, roi_id in [('3', 1), ('7', 2)]:
            case = next(c for c in report['results'] if c['tc_number'] == tc)
            roi = next(r for r in case['rois'] if r['roi_id'] == roi_id)
            directory = Path(case['representative_image']).parent
            for name in ('first_ocr_frame.png', 'screen.png'):
                source = directory/name
                with Image.open(source) as image:
                    frame = image.convert('RGB')
                started = time.monotonic()
                result = processor._run_roi_ocr(frame, tuple(roi['rect']), roi['expected'])
                record = processor._last_ocr_input
                sample = {'report': str(path), 'tc': tc, 'roi': roi_id, 'source': str(source),
                          'raw': result.raw_text, 'evaluated': result.text,
                          'elapsed_sec': round(time.monotonic()-started, 4),
                          'input_sha256': hashlib.sha256(record.image.tobytes()).hexdigest(),
                          'superscript': result.superscript_evidence, 'overlap': result.overlap_evidence}
                output['samples'].append(sample)
                print(json.dumps({k: sample[k] for k in ('tc', 'roi', 'source', 'raw', 'evaluated', 'elapsed_sec')},
                                 ensure_ascii=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
