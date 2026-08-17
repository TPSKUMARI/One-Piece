"""
find_match.py
═══════════════════════════════════════════════════════════════════════════
ORB-based image matching for One Piece TCG card variant selection.

Two public APIs
───────────────

1.  find_best_variant(scan_bytes, variant_bytes_list)  →  int
        Drop-in replacement for image_comparator.find_best_variant().
        Accepts raw JPEG/PNG bytes and returns the 0-based index of the
        best-matching DB variant.  Used by app.py automatically.

2.  find_best_match(scanned_path, db_folder)           →  None
        CLI helper: compares a scanned image file against every image in
        a folder and prints a ranked table.

Algorithm
─────────
  • ORB (2000 keypoints) feature detection on grayscale images.
  • BFMatcher (Hamming) + Lowe ratio test (0.75) → good_matches.
  • RANSAC homography → geometric inlier count.
  • Ranking: inliers DESC, then good_matches DESC.

Robustness advantages over pHash
  ✓ Handles watermarks / SAMPLE text — ORB ignores flat regions
  ✓ Handles slight perspective distortion from physical scans
  ✓ Handles lighting / brightness variation
  ✓ No external dependencies beyond cv2 + numpy (already present)

Usage (CLI)
───────────
    python find_match.py <scanned_image> <folder_with_db_images>
    python find_match.py          # auto-detects *scanned* file in cwd
"""

import sys
import os
import glob

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_gray(path: str) -> np.ndarray:
    """Load an image from a file path and return a grayscale numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_gray_from_bytes(data: bytes) -> np.ndarray:
    """
    Decode raw image bytes (JPEG / PNG / WebP …) into a grayscale numpy array.
    Raises ValueError if the bytes cannot be decoded.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed — unsupported or corrupt image bytes")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ══════════════════════════════════════════════════════════════════════════════
# Core ORB matching
# ══════════════════════════════════════════════════════════════════════════════

def match_score(query_gray: np.ndarray, candidate_gray: np.ndarray) -> dict:
    """
    Run ORB feature matching between two grayscale images.

    Returns
    -------
    dict with keys:
        good_matches  – number of matches that pass Lowe's ratio test
        inliers       – number of homography inliers (RANSAC geometric check)

    Higher is better for both metrics.
    """
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(query_gray, None)
    kp2, des2 = orb.detectAndCompute(candidate_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {"good_matches": 0, "inliers": 0}

    # Brute-force matcher with Hamming distance (binary descriptors)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test — keep only unambiguous matches
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    inliers = 0
    if len(good) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.ravel().sum())

    return {"good_matches": len(good), "inliers": inliers}


# ══════════════════════════════════════════════════════════════════════════════
# Public API — bytes-based (used by app.py)
# ══════════════════════════════════════════════════════════════════════════════

def find_best_variant(scan_bytes: bytes, variant_bytes_list: list) -> int:
    """
    ORB-based variant selector — replaces image_comparator.find_best_variant().

    Compare the uploaded scan image against every fetched DB variant image and
    return the 0-based index of the best match (highest inliers, then highest
    good_matches).

    Parameters
    ----------
    scan_bytes          : Raw bytes of the uploaded / Gemini-processed image.
    variant_bytes_list  : List of raw bytes, one entry per DB variant.
                          An entry may be None if the image could not be fetched.

    Returns
    -------
    int
        0-based index of the best-matching variant.
        Returns 0 on any failure or if list is empty / singleton.
    """
    if not variant_bytes_list:
        return 0

    if len(variant_bytes_list) == 1:
        print("  [ORB] Only 1 variant — skipping comparison, returning index 0.")
        return 0

    # Decode the scan image once
    try:
        query_gray = load_gray_from_bytes(scan_bytes)
    except Exception as e:
        print(f"  [ORB] Failed to decode scan image: {e} — defaulting to index 0.")
        return 0

    results = []   # list of (idx, good_matches, inliers)

    for idx, vbytes in enumerate(variant_bytes_list):
        if not vbytes:
            print(f"  [ORB] Variant {idx}: no image bytes — skipping.")
            results.append((idx, 0, 0))
            continue

        try:
            candidate_gray = load_gray_from_bytes(vbytes)
            scores         = match_score(query_gray, candidate_gray)
            results.append((idx, scores["good_matches"], scores["inliers"]))
            print(
                f"  [ORB] Variant {idx}: "
                f"good_matches={scores['good_matches']:4d}  "
                f"inliers={scores['inliers']:4d}"
            )
        except Exception as e:
            print(f"  [ORB] Variant {idx} error: {e}")
            results.append((idx, 0, 0))

    if not results:
        return 0

    # Rank by inliers DESC, then good_matches DESC — pick the winner
    results.sort(key=lambda x: (x[2], x[1]), reverse=True)
    best_idx, best_gm, best_inliers = results[0]

    print(
        f"  [ORB] Best match → variant index {best_idx}  "
        f"(good_matches={best_gm}, inliers={best_inliers})"
    )
    return best_idx


# ══════════════════════════════════════════════════════════════════════════════
# Public API — file-path-based (used from CLI / testing)
# ══════════════════════════════════════════════════════════════════════════════

def find_best_match(scanned_path: str, db_folder: str) -> None:
    """
    CLI entry point: compare a scanned card image against all images in
    db_folder and print a ranked result table.
    """
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    candidates = []
    for ext in extensions:
        candidates.extend(glob.glob(os.path.join(db_folder, ext)))
    candidates = [
        p for p in candidates
        if os.path.abspath(p) != os.path.abspath(scanned_path)
    ]

    if not candidates:
        print("No candidate images found in", db_folder)
        return

    print(f"\nScanned image  : {scanned_path}")
    print(f"Comparing against {len(candidates)} database image(s)...\n")

    query   = load_gray(scanned_path)
    results = []   # list of (path, scores_dict)

    for path in candidates:
        try:
            candidate = load_gray(path)
            scores    = match_score(query, candidate)
            results.append((path, scores))
            print(
                f"  {os.path.basename(path):45s}  "
                f"good_matches={scores['good_matches']:4d}  "
                f"inliers={scores['inliers']:4d}"
            )
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(path)}: {e}")

    if not results:
        print("No results to compare.")
        return

    # Rank by inliers first, then good_matches
    results.sort(key=lambda x: (x[1]["inliers"], x[1]["good_matches"]), reverse=True)
    best_path, best_scores = results[0]

    print("\n" + "=" * 60)
    print("BEST MATCH:")
    print(f"  File         : {os.path.basename(best_path)}")
    print(f"  Full path    : {best_path}")
    print(f"  Good matches : {best_scores['good_matches']}")
    print(f"  Inliers      : {best_scores['inliers']}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) == 3:
        scanned = sys.argv[1]
        folder  = sys.argv[2]
    else:
        # Auto-detect: find a file with "scanned" in its name
        folder  = os.path.dirname(os.path.abspath(__file__))
        matches = glob.glob(os.path.join(folder, "*scanned*"))
        if not matches:
            print(
                "Could not auto-detect scanned image.\n"
                "Usage: python find_match.py <scanned_image> <db_folder>"
            )
            sys.exit(1)
        scanned = matches[0]

    find_best_match(scanned, folder)
