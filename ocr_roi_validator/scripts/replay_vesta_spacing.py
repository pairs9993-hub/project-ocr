"""Replay a saved first capture with recorded preprocessing, without live Vesta."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--tc', default='5')
    parser.add_argument('--roi', type=int, default=2)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    case = next(c for c in report['results'] if c['tc_number'] == args.tc)
    roi = next(r for r in case['rois'] if r['roi_id'] == args.roi)
    source = Path(case['representative_image'])
    metadata = json.loads((Path(roi['diagnostic'])/'metadata.json').read_text(encoding='utf-8'))
    config = metadata['preprocess']
    processor = OCRValidatorGUI.__new__(OCRValidatorGUI)
    processor.engine = OCREngine(use_rapid_default=True)
    settings = {'language_var': report['settings']['language'], 'roi_margin_var': str(config['margin']),
                'min_roi_side_var': str(config['min_side']), 'fast_long_roi_var': config['pad_long_roi'],
                'auto_upscale_var': config['auto_upscale'], 'context_detect_var': False,
                'context_margin_var': '0'}
    for key, value in settings.items():
        setattr(processor, key, FrozenValue(value))
    output = {'source': str(source), 'preprocess': config,
              'scope': 'saved_first_capture_replay_not_live_vesta',
              'assumptions': ['Rapid default engine', 'direct ROI path; context mode was not recorded'],
              'attempts': []}
    with Image.open(source) as image:
        frame = image.convert('RGB')
    for _ in range(2):
        started = time.monotonic()
        result = processor._run_roi_ocr(frame, tuple(roi['rect']), roi['expected'])
        record = processor._last_ocr_input
        output['attempts'].append({
            'raw': result.raw_text, 'evaluated': result.text,
            'elapsed_sec': round(time.monotonic()-started, 4),
            'input_size': record.image.size,
            'input_sha256': hashlib.sha256(record.image.tobytes()).hexdigest(),
            'image_spacing': result.spacing_evidence})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(output, ensure_ascii=True))


if __name__ == '__main__':
    main()
