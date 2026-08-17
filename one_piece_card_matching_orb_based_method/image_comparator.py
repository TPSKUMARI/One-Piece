"""
image_comparator.py
────────────────────────────────────────────────────────────────────
Replaces the second Gemini call (variant matching) with a pure
image-comparison approach using pHash + SAMPLE-watermark masking
and inpainting from image_comparison.py.

Public API
──────────
find_best_variant(scan_bytes: bytes, variant_bytes_list: list[bytes]) -> int
    Returns the index of the variant image that is most similar
    to the uploaded scan image (lowest combined pHash distance).
    Falls back to 0 if anything goes wrong.
"""

import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import imagehash


# ─── Core helpers (adapted from image_comparison.py) ──────────────────────────

def _detect_sample_mask(img: np.ndarray) -> np.ndarray:
    """
    Create a binary mask where SAMPLE watermark pixels are white (255).
    Works on the near-white semi-transparent diagonal text region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Restrict mask to the diagonal band where SAMPLE text typically appears
    h, w = mask.shape
    region_mask = np.zeros_like(mask)
    region_mask[int(h * 0.2):int(h * 0.75), int(w * 0.05):int(w * 0.95)] = 255
    mask = cv2.bitwise_and(mask, region_mask)

    # Dilate slightly to fully cover text edges
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def _compare_with_mask(scan_path: str, db_path: str) -> float:
    """
    Strategy A — mask-based pHash comparison.
    Zeroes out the SAMPLE watermark region in both images before hashing.
    Returns pHash distance (lower = more similar).
    """
    scan = cv2.imread(scan_path)
    db   = cv2.imread(db_path)

    if scan is None or db is None:
        return float("inf")

    db = cv2.resize(db, (scan.shape[1], scan.shape[0]))

    # Build mask from the DB image (cleaner source for watermark detection)
    mask     = _detect_sample_mask(db)
    inv_mask = cv2.bitwise_not(mask)

    scan_masked = cv2.bitwise_and(scan, scan, mask=inv_mask)
    db_masked   = cv2.bitwise_and(db,   db,   mask=inv_mask)

    scan_pil = Image.fromarray(cv2.cvtColor(scan_masked, cv2.COLOR_BGR2RGB))
    db_pil   = Image.fromarray(cv2.cvtColor(db_masked,   cv2.COLOR_BGR2RGB))

    return float(imagehash.phash(scan_pil) - imagehash.phash(db_pil))


def _inpaint_sample(img_path: str) -> np.ndarray | None:
    """
    Strategy B helper — inpaint the SAMPLE watermark region using INPAINT_TELEA.
    Returns the restored image array, or None on failure.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    mask    = _detect_sample_mask(img)
    restored = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return restored


def _compare_with_inpaint(scan_path: str, db_path: str) -> float:
    """
    Strategy B — inpaint-then-pHash comparison.
    Restores both images before hashing.
    Returns pHash distance (lower = more similar).
    """
    scan_clean = _inpaint_sample(scan_path)
    db_clean   = _inpaint_sample(db_path)

    if scan_clean is None or db_clean is None:
        return float("inf")

    scan_pil = Image.fromarray(cv2.cvtColor(scan_clean, cv2.COLOR_BGR2RGB))
    db_pil   = Image.fromarray(cv2.cvtColor(db_clean,   cv2.COLOR_BGR2RGB))

    return float(imagehash.phash(scan_pil) - imagehash.phash(db_pil))


# ─── Public API ───────────────────────────────────────────────────────────────

def find_best_variant(scan_bytes: bytes, variant_bytes_list: list) -> int:
    """
    Compare the uploaded scan image against every fetched DB variant image and
    return the index of the best match (lowest combined pHash distance).

    Parameters
    ----------
    scan_bytes          : Raw bytes of the uploaded / Gemini-processed image.
    variant_bytes_list  : List of raw bytes, one entry per DB variant.
                          An entry may be None if the image could not be fetched.

    Returns
    -------
    int  Index of the best-matching variant (0-based).
         Returns 0 on any failure or if list is empty/singleton.
    """
    if not variant_bytes_list:
        return 0

    if len(variant_bytes_list) == 1:
        print("  [Comparator] Only 1 variant — skipping comparison, returning index 0.")
        return 0

    # Write scan to a temp file once
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            tf.write(scan_bytes)
            scan_tmp = tf.name
    except Exception as e:
        print(f"  [Comparator] Failed to write scan temp file: {e}")
        return 0

    results = []

    for idx, vbytes in enumerate(variant_bytes_list):
        if not vbytes:
            print(f"  [Comparator] Variant {idx}: no image bytes — skipping")
            results.append((idx, float("inf"), float("inf"), float("inf")))
            continue

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
                tf.write(vbytes)
                db_tmp = tf.name

            score_a = _compare_with_mask(scan_tmp, db_tmp)
            score_b = _compare_with_inpaint(scan_tmp, db_tmp)
            # Weight inpaint higher (same ratio as original image_comparison.py)
            combined = 0.4 * score_a + 0.6 * score_b

            results.append((idx, score_a, score_b, combined))
            print(
                f"  [Comparator] Variant {idx}: "
                f"mask_pHash={score_a:.0f}  inpaint_pHash={score_b:.0f}  combined={combined:.2f}"
            )

        except Exception as e:
            print(f"  [Comparator] Variant {idx} comparison error: {e}")
            results.append((idx, float("inf"), float("inf"), float("inf")))
        finally:
            try:
                os.unlink(db_tmp)
            except Exception:
                pass

    # Clean up scan temp file
    try:
        os.unlink(scan_tmp)
    except Exception:
        pass

    if not results:
        return 0

    # Sort by combined score ascending (lowest = best match)
    results.sort(key=lambda x: x[3])
    best_idx, a, b, c = results[0]
    print(
        f"  [Comparator] Best match → variant index {best_idx}  "
        f"(mask={a:.0f}, inpaint={b:.0f}, combined={c:.2f})"
    )
    return best_idx
