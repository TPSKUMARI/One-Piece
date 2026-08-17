"""
rarity_rapid_ocr.py
────────────────────────────────────────────────────────────────────
OCR helpers for One Piece TCG card number + rarity extraction.

Upgraded from test_ocr.py:
  • strong_preprocess()  — 2.3× upscale + contrast boost before OCR
  • clean_rarity_text()  — fixes common OCR misreads (LC→C, CC→C …)
  • extract_rarity_only() — scan OCR lines for a rarity token
  • parse_card_number_and_rarity() — primary card-number + rarity parser
  • run_ocr_on_bytes()   — public entry point used by Flask route
                           accepts raw image bytes, returns (card_number, rarity)

Public API (unchanged — Flask route just calls run_ocr_on_bytes or
parse_card_number_and_rarity, both still present):
    parse_card_number_and_rarity(ocr_result) -> (card_number, rarity)
    run_ocr_on_bytes(image_bytes, engine)    -> (card_number, rarity)
"""

import re
import tempfile
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from rapidocr_onnxruntime import RapidOCR

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_RARITIES      = {"C", "UC", "R", "SR", "L", "SEC", "DON"}
CARD_NUMBER_PATTERN = re.compile(r'[A-Z]{2,4}\d{2}-\d{3}')


# ── Preprocessing ─────────────────────────────────────────────────────────────
def strong_preprocess(pil_image: Image.Image, scale: float = 2.3) -> np.ndarray:
    """
    Upscale + boost contrast + sharpen before passing to OCR engine.
    Returns an RGB numpy array ready for RapidOCR.
    """
    print(f"[Preprocess] contrast boost + {scale}x upscale...")
    gray = pil_image.convert('L')
    gray = ImageEnhance.Contrast(gray).enhance(2.6)
    gray = ImageEnhance.Brightness(gray).enhance(1.15)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3.0, percent=320, threshold=1))
    w, h = gray.size
    upscaled = gray.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    np_img = np.array(upscaled, dtype=np.uint8)
    return np.stack([np_img] * 3, axis=-1)


# ── Text-cleaning helpers ─────────────────────────────────────────────────────
def clean_rarity_text(text: str) -> str:
    """Strip non-alpha-numeric, remove digits, fix common OCR misreads."""
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    cleaned = re.sub(r'[0-9]+', '', cleaned)
    cleaned = re.sub(r'(LC|CC|C1|01|0C)', 'C', cleaned)   # common misreads
    return cleaned


def extract_rarity_only(ocr_result, mode: str = "normal"):
    """
    Scan OCR result lines for a rarity token.
    Returns the rarity string or None.
    """
    if not ocr_result:
        return None
    for _, text, conf in ocr_result:
        text = text.strip()
        print(f"    Crop ({mode}) OCR: '{text}' (conf={conf:.2f})")
        cleaned = clean_rarity_text(text)
        for token in re.findall(r'[A-Z]+', cleaned):
            if token in VALID_RARITIES or (len(token) == 1 and token in {'C', 'R', 'L'}):
                print(f"    → Rarity detected: **{token}**")
                return token
    return None


# ── Primary parser ────────────────────────────────────────────────────────────
def parse_card_number_and_rarity(ocr_result):
    """
    Scan ALL OCR lines for a card number + rarity.
    Returns (card_number, rarity) — both str, or None if not found.

    Compatible with the existing Flask route call:
        _, ocr_rarity = parse_card_number_and_rarity(ocr_result)
    """
    if not ocr_result:
        return None, None

    print(f"\n[Rarity Parser] Scanning {len(ocr_result)} OCR line(s)...")

    card_number = None
    rarity      = None

    for i, row in enumerate(ocr_result):
        _, text, conf = row
        text = text.strip()
        print(f"  Line {i+1} (conf={conf:.2f}): '{text}'")

        m = CARD_NUMBER_PATTERN.search(text)
        if m:
            cn = m.group()
            if card_number is None:
                card_number = cn

            # Look for rarity right after the card number on the same line
            remainder = text[m.end():].strip()
            cleaned   = clean_rarity_text(remainder)
            for token in re.findall(r'[A-Z]+', cleaned):
                if token in VALID_RARITIES:
                    rarity = token
                    break

        # Fallback: check the whole line for a rarity token
        if not rarity:
            cleaned = clean_rarity_text(text)
            for token in re.findall(r'[A-Z]+', cleaned):
                if token in VALID_RARITIES:
                    rarity = token
                    break

        if card_number and rarity:
            print(f"[Rarity Parser] ✅ Found → card_number='{card_number}'  rarity='{rarity}'")
            return card_number, rarity

    if card_number:
        print(f"[Rarity Parser] ⚠  card_number='{card_number}' found but NO rarity.")
    else:
        print("[Rarity Parser] ⚠  No card number found in any OCR line.")

    return card_number, rarity


# ── Multi-stage rarity recovery ───────────────────────────────────────────────
def _recover_rarity(preprocessed_np: np.ndarray, ocr_result, card_number: str, engine) -> str | None:
    """
    If primary parse didn't find a rarity, crop the area to the right of the
    card-number bounding box and retry OCR with extreme contrast + invert.
    Returns the recovered rarity string, or None.
    """
    print(f"\n[Rarity Recovery] Multi-stage attempt for {card_number}...")

    for box, text, _ in ocr_result:
        if card_number not in text and not CARD_NUMBER_PATTERN.search(text):
            continue

        xs     = [p[0] for p in box]
        ys     = [p[1] for p in box]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        height = max_y - min_y
        pad    = int(height * 0.35)

        for width_mult in [1.4, 1.8, 2.3]:
            crop_x1 = int(max_x + pad * 1.0)
            crop_x2 = int(crop_x1 + height * width_mult)
            crop_y1 = int(min_y - pad * 1.5)
            crop_y2 = int(max_y + pad * 1.5)

            h, w = preprocessed_np.shape[:2]
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            crop_x2 = min(w, crop_x2)
            crop_y2 = min(h, crop_y2)

            if crop_x2 - crop_x1 < 50:
                continue

            crop_np  = preprocessed_np[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_pil = Image.fromarray(crop_np).convert('L')

            # Stage 1: Extreme contrast
            enhanced = ImageEnhance.Contrast(crop_pil).enhance(4.5)
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2.2, percent=300))
            up       = enhanced.resize(
                (enhanced.width * 4, enhanced.height * 4), Image.Resampling.LANCZOS
            )
            final    = np.stack([np.array(up)] * 3, axis=-1)
            res, _   = engine(final)
            rarity   = extract_rarity_only(res, f"contrast-w{width_mult}")
            if rarity:
                break

            # Stage 2: Inverted
            inv     = ImageOps.invert(crop_pil)
            inv_up  = inv.resize((inv.width * 4, inv.height * 4), Image.Resampling.LANCZOS)
            inv_arr = np.stack([np.array(inv_up)] * 3, axis=-1)
            inv_res, _ = engine(inv_arr)
            rarity  = extract_rarity_only(inv_res, f"inverted-w{width_mult}")
            if rarity:
                break

        if rarity:
            print(f"✅ SUCCESS: Recovered rarity = **{rarity}**")
            return rarity

    print("⚠ Rarity still not detected after all recovery attempts.")
    return None


# ── Public entry point for Flask route ───────────────────────────────────────
def run_ocr_on_bytes(image_bytes: bytes, engine) -> tuple:
    """
    Full OCR pipeline on raw image bytes:
      1. Preprocess (upscale + contrast boost)
      2. Run RapidOCR
      3. Parse card number + rarity
      4. Multi-stage rarity recovery if needed

    Parameters
    ----------
    image_bytes : Raw bytes of the OCR image (e.g. the second uploaded file).
    engine      : An already-instantiated RapidOCR() instance (reuse for speed).

    Returns
    -------
    (card_number, rarity) — either may be None.
    """
    from PIL import Image as _PILImage
    import io as _io

    pil_image      = _PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
    SCALE          = 2.3
    preprocessed   = strong_preprocess(pil_image, scale=SCALE)

    ocr_result, _ = engine(preprocessed)

    if not ocr_result:
        print("No text detected by RapidOCR.")
        return None, None

    print(f"OCR detected {len(ocr_result)} line(s):")
    for i, (_, text, conf) in enumerate(ocr_result, 1):
        print(f"  [{i}] {text} (conf={conf:.3f})")

    card_number, rarity = parse_card_number_and_rarity(ocr_result)

    # Multi-stage recovery if rarity not yet found
    if rarity is None and card_number:
        rarity = _recover_rarity(preprocessed, ocr_result, card_number, engine)

    if rarity is None:
        print("⚠ Rarity not detected after all attempts.")

    return card_number, rarity