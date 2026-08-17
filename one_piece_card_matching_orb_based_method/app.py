#!/usr/bin/env python3
"""
One Piece Card Lookup - Smart Variant Matching
Run: python app.py  |  Open: http://localhost:5000
"""

import os, io, re, json, time
import base64, requests, csv, tempfile
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

from find_match import find_best_variant
from rarity_rapid_ocr import parse_card_number_and_rarity, run_ocr_on_bytes
from rapidocr_onnxruntime import RapidOCR as _RapidOCR

_ocr_engine = _RapidOCR()   # reuse one instance across requests

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=api_key)
app = Flask(__name__)


# ── Gemini Config ─────────────────────────────────────────────────────────
MODELS = ["gemini-3-flash-preview","gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-1.5-flash"]

GEMINI_PROMPT = """
You are an expert One Piece TCG card reader.

Return ONLY a JSON object with these exact keys:

- card_number : The card ID at the bottom-right corner of the card.
                Format is ALWAYS: 2-4 UPPERCASE LETTERS + 2 digits + DASH + 3 digits
                Examples: OP09-062, ST01-003, EB01-010, OP01-001, OP10-119
                HOW TO READ IT:
                At the bottom-right you will see something like: OP09-062 [UC] 1
                - "OP09-062" = card_number (this is ALL you want)
                - "[UC]" = rarity symbol — DO NOT include this
                - "1" = cost/DON count — DO NOT include this
                The card_number ALWAYS ends right before any box/bracket symbol.
                COMMON MISTAKES TO AVOID:
                - Do NOT append rarity letters: "OP09-062UC" is WRONG, "OP09-062" is RIGHT
                - Do NOT append extra digits: "OP09-0621" is WRONG
                - Do NOT confuse "O" (letter) with "0" (zero) in the prefix
                - The prefix letters are always a known set code like OP01-OP10, ST01-ST20, EB01, PRB01
                - Double-check the 3-digit number after the dash carefully
                Return null ONLY if completely unreadable.

- set_code    : Letters+digits before the dash. e.g. "OP09" from "OP09-062". null if unknown.

- card_name   : The large character/card name printed on the card.
                Japanese is fine e.g. "ニコ・ロビン", "氷河時代"
                null if unreadable.

- language    : "ja" for Japanese, "en" for English.

- illustration_desc : 5-8 words describing the card artwork only.

=====================================================
FINAL CHECKLIST — before returning your JSON:
=====================================================
  [ ] card_number does NOT include the rarity letters or trailing digits
  [ ] card_name is the name printed on the card, not a translation
  [ ] Response is ONLY valid JSON — no markdown, no explanation, no extra text

Return ONLY valid JSON. No markdown, no explanation, no extra text.
"""


gemini_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.0
)


# ── Database ──────────────────────────────────────────────────────────────
def normalize_id(card_id):
    if not card_id:
        return ""
    return card_id.strip().upper()


def clean_card_number(raw: str) -> str:
    if not raw:
        return raw
    raw = raw.strip().upper()
    m = re.match(r"([A-Z0-9]{2,5}-\d{3,4})", raw)
    if m:
        cleaned = m.group(1)
        if cleaned != raw:
            print(f"  card_number cleaned: '{raw}' -> '{cleaned}'")
        return cleaned
    return raw


def load_database(path):
    if not os.path.exists(path):
        print(f"No database at {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_url = "https://www.onepiece-cardgame.com"
    db       = {}

    for card_set in data.get("card_sets", []):
        for card in card_set.get("cards", []):
            rel  = card.get("img_url", "").lstrip(".")
            card["full_image_url"] = f"{base_url}{rel.split(chr(63))[0]}"

            raw_id = card.get("id", "").strip().upper()
            if raw_id:
                db[raw_id] = card

    print(f"Database loaded: {len(db)} cards")
    return db

database = load_database("cards.json")


# ── Find ALL variants by scanning entire DB ───────────────────────────────
def find_all_variants(card_number: str) -> list:
    if not card_number:
        return []

    base     = card_number.strip().upper()
    variants = []
    seen     = set()

    for key in sorted(database.keys()):
        if key == base or key.startswith(base):
            if key not in seen:
                card = dict(database[key])
                if key == base:
                    label = "Standard Art"
                else:
                    suffix = key[len(base):]
                    label  = f"Alternate Art ({suffix.strip('_')})"
                card["_variant_label"] = label
                card["_variant_key"]   = key
                variants.append(card)
                seen.add(key)

    print(f"  Found {len(variants)} variant(s) for {card_number}")
    return variants


# ── Gemini ────────────────────────────────────────────────────────────────
def parse_json(text):
    try:
        return json.loads(text)
    except:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return None


def call_gemini(image_bytes):
    for model in MODELS:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    GEMINI_PROMPT.strip(),
                ],
                config=gemini_config,
            )
            parsed = parse_json((resp.text or "").strip())
            if parsed:
                return {"ok": True, "data": parsed, "model": model}
            return {"ok": False, "error": "Could not parse JSON from Gemini"}
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                m2 = re.search(r"retry in (\d+)", err)
                time.sleep(int(m2.group(1)) + 2 if m2 else 30)
            continue
    return {"ok": False, "error": "All models failed or quota exceeded"}


# ── Fetch image helpers ───────────────────────────────────────────────────
_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.onepiece-cardgame.com/"
}

def _fetch_and_convert(url: str):
    """Internal helper: fetch URL → validate image → return JPEG bytes.
    Returns None on any error.
    """
    if not url:
        return None
    try:
        r = requests.get(url, headers=_FETCH_HEADERS, timeout=10)
        r.raise_for_status()
        if "image" not in r.headers.get("Content-Type", "").lower():
            print(f"  Not an image Content-Type [{url}]: {r.headers.get('Content-Type')}")
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        print(f"  [+] Fetched variant: {url.split('/')[-1]} ({len(r.content)//1024}KB)")
        return buf.getvalue()   # raw JPEG bytes
    except Exception as e:
        print(f"  [-] Fetch Image Error [{url}]: {e}")
        return None


def fetch_image_b64(url: str):
    """Fetch a DB card image and return a base64 data-URI, or None."""
    raw = _fetch_and_convert(url)
    if raw is None:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def fetch_image_bytes(url: str):
    """Fetch a DB card image and return raw JPEG bytes, or None."""
    return _fetch_and_convert(url)


# ── Save Results ──────────────────────────────────────────────────────────
RESULTS_JSON = "scan_results.json"
RESULTS_CSV  = "scan_results.csv"

CSV_FIELDS = [
    "timestamp", "source_image",
    "card_number", "card_name", "set_code", "rarity",
    "language", "model_used",
    "db_found", "db_card_id", "db_card_name",
    "db_rarity", "db_color", "db_type",
    "db_cost", "db_power", "db_counter", "db_effect",
]

def save_result(gemini_data: dict, model_used: str,
                variant_list: list, image_filename: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best      = variant_list[0] if variant_list else {}

    record = {
        "timestamp":    timestamp,
        "source_image": image_filename,
        "card_number":  gemini_data.get("card_number"),
        "card_name":    gemini_data.get("card_name"),
        "set_code":     gemini_data.get("set_code"),
        "rarity":       gemini_data.get("rarity"),
        "language":     gemini_data.get("language"),
        "model_used":   model_used,
        "db_found":     len(variant_list) > 0,
        "db_card_id":   best.get("id"),
        "db_card_name": best.get("name"),
        "db_rarity":    best.get("rarity"),
        "db_color":     best.get("color"),
        "db_type":      best.get("type"),
        "db_cost":      best.get("cost"),
        "db_power":     best.get("power"),
        "db_counter":   best.get("counter"),
        "db_effect":    best.get("effect"),
        "variants_found": [v.get("id", "") for v in variant_list],
    }

    existing = []
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(record)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    print(f"  Saved to {RESULTS_JSON} and {RESULTS_CSV}")
    return record


# ── Debug Route ───────────────────────────────────────────────────────────
@app.route("/debug/<card_id>")
def debug_card(card_id):
    card_id = card_id.strip().upper()
    matches = {k: v.get("id") for k, v in database.items() if k.startswith(card_id)}
    return jsonify({
        "searched_for": card_id,
        "total_found":  len(matches),
        "keys":         matches,
    })


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"})

    # ── Read all uploaded files eagerly (before any processing) ──────────
    raw      = request.files["image"].read()                  # main image
    ocr_file = request.files.get("ocr_image")
    ocr_raw  = ocr_file.read() if ocr_file else None          # ONLY use ocr_image

    print(f"  ocr_image received: {ocr_file is not None}  size: {len(ocr_raw) if ocr_raw else 0} bytes")

    # ── Gemini image ──────────────────────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        jpeg_bytes = buf.getvalue()
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid image: {e}"})

    print(f"  Gemini image size: {len(jpeg_bytes)//1024}KB")

    # ── RapidOCR rarity extraction (second image only) ────────────────────
    # Uses full test_ocr.py pipeline: strong_preprocess (2.3x upscale +
    # contrast boost) → primary parse → multi-stage crop/invert recovery.
    print("\n" + "="*50)
    print(" === RAPID OCR COLLECTED DATA ===")
    print("="*50)
    ocr_rarity = None
    ocr_rarity_source = "none"
    if ocr_raw:
        try:
            _, ocr_rarity = run_ocr_on_bytes(ocr_raw, _ocr_engine)
            if ocr_rarity:
                ocr_rarity_source = "rapidocr"
                print(f"\n => FOUND RARITY (RapidOCR): '{ocr_rarity}'")
            else:
                print("\n => NO VALID RARITY FOUND in OCR image")
        except Exception as e:
            print(f"RapidOCR error: {e}")
    else:
        print("No ocr_image uploaded — skipping rarity extraction")

    # Step 1: Gemini extracts card info from image
    result = call_gemini(jpeg_bytes)
    if not result["ok"]:
        return jsonify({"success": False, "error": result["error"]})

    g          = result["data"]
    model_used = result["model"]

    # Step 2: Retry once if card_number looks malformed
    raw_cn = (g.get("card_number") or "").strip().upper()
    if not re.match(r"[A-Z0-9]{2,5}-\d{3,4}", raw_cn):
        print(f"  card_number '{raw_cn}' looks wrong, retrying Gemini...")
        result2 = call_gemini(jpeg_bytes)
        if result2["ok"] and result2["data"].get("card_number"):
            g          = result2["data"]
            model_used = result2["model"]
            print(f"  Retry gave: {g.get('card_number')}")

    # Step 3: Clean card number
    raw_cn           = g.get("card_number", "") or ""
    g["card_number"] = clean_card_number(raw_cn)
    print("\n" + "="*50)
    print(" === GEMINI COLLECTED DATA ===")
    print("="*50)
    print(f"Model Used  : {model_used}")
    print(f"Card Number : {g['card_number']} (raw: '{raw_cn}')")
    print(f"Card Name   : {g.get('card_name')}")
    print(f"Set Code    : {g.get('set_code')}")
    print(f"Language    : {g.get('language')}")

    # Step 3b: Set rarity exclusively from RapidOCR result
    g["rarity"] = ocr_rarity
    g["rarity_source"] = "rapidocr" if ocr_rarity else "none"

    print("\n" + "="*50)
    print(" === FINAL COMBINED INFO ===")
    print("="*50)
    print(f"Active Card Number : {g['card_number']}")
    print(f"Active Rarity      : {g.get('rarity')}  [source: {g['rarity_source']}]")

    print("\n" + "="*50)
    print(" === REQUIRED OTHER DATA (DATABASE LOOKUP) ===")
    print("="*50)

    # Step 4: Find ALL variants in DB
    variants = find_all_variants(g.get("card_number", ""))

    # Step 5: Fetch official images for all variants
    # Store both base64 (for the UI) and raw bytes (for image comparison)
    variant_list       = []
    variant_raw_bytes  = []   # parallel list of raw JPEG bytes per variant

    for v in variants:
        url        = v.get("full_image_url", "")
        raw_bytes  = fetch_image_bytes(url)          # raw bytes for comparator
        b64        = (
            "data:image/jpeg;base64," + base64.b64encode(raw_bytes).decode()
            if raw_bytes else None
        )
        variant_raw_bytes.append(raw_bytes)          # may be None if fetch failed
        variant_list.append({
            "key":     v["_variant_key"],
            "label":   v["_variant_label"],
            "id":      v.get("id", ""),
            "name":    v.get("name", ""),
            "rarity":  v.get("rarity", ""),
            "color":   v.get("color", ""),
            "type":    v.get("type", ""),
            "cost":    v.get("cost"),
            "power":   v.get("power"),
            "counter": v.get("counter"),
            "effect":  v.get("effect", ""),
            "image":   b64,
        })

    # Step 6: Select best variant using ORB feature matching
    #         (ORB inlier + good-match ranking via find_match.find_best_variant)
    best_idx = 0
    if len(variant_list) > 1:
        print(f"  Multiple variants found ({len(variant_list)}). Running image comparison...")
        try:
            best_idx = find_best_variant(jpeg_bytes, variant_raw_bytes)
            if not isinstance(best_idx, int) or not (0 <= best_idx < len(variant_list)):
                best_idx = 0
        except Exception as e:
            print(f"  ORB Matching Error: {e}")
            best_idx = 0

    print(f"  Returning {len(variant_list)} variant(s), match = index {best_idx}")

    # Step 7: Save result to JSON + CSV
    saved = save_result(
        gemini_data    = g,
        model_used     = model_used,
        variant_list   = variant_list,
        image_filename = request.files["image"].filename or "unknown",
    )
    
    db_rarity = variant_list[best_idx].get("rarity") if variant_list else None

    return jsonify({
        "success":        True,
        "gemini":         g,
        "model_used":     model_used,
        "variants":       variant_list,
        "best_idx":       best_idx,
        "db_found":       len(variant_list) > 0,
        "db_rarity":      db_rarity,
        "saved":          True,
        "timestamp":      saved["timestamp"],
        "rarity_source":  g["rarity_source"],
    })


if __name__ == "__main__":
    print(f"API key: {api_key[:8]}...{api_key[-4:]}")
    print("Open: http://localhost:5000")
    app.run(debug=True, port=5000)