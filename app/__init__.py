"""OCR Validation GUI app (Streamlit).

Self-contained desktop app for OCR validation on UI screenshots.
Two modes:
  - General: OCR a single image (optional reference text).
  - ZIP: upload a test-image .zip and a reference-image .zip, the app
    pairs them by file stem, OCRs both, computes CER/WER, and exports
    an Excel report with embedded thumbnails.
"""

__version__ = "0.1.0"
