import os
import re
import json
import time
import base64
from google import genai
from google.genai import types

# ── Gemini Config ─────────────────────────────────────────────────────────
MODELS = ["gemini-3-flash-preview", "gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-1.5-flash"]

VARIANT_MATCH_PROMPT = """
You are an expert One Piece TCG card visual authenticator.
You are given a series of images:
- IMAGE 1 (The very first image) is a photo uploaded by a user of their physical card.
- IMAGE 2 and onwards are official digital variant images from the database.

Your job is to compare the physical card photo (IMAGE 1) to the official variant images, and determine WHICH official variant EXACTLY MATCHES the physical card.

Look closely at:
- The background artwork layout
- The character's pose, expression, and position
- Foil patterns, borders, or extended art treatments

Return ONLY a JSON object with this exact key:

- matched_variant_index : The 0-based index of the matching variant.
                          For example, if IMAGE 2 is the exact match, return 0.
                          If IMAGE 3 is the exact match, return 1.
                          If IMAGE 4 is the exact match, return 2.
                          And so on.

Return ONLY valid JSON. No markdown, no explanation, no extra text.
"""

gemini_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.0
)

def _parse_json(text):
    """Parse Gemini response as JSON."""
    def extract_dict(obj):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    return item
        return None

    try:
        obj = json.loads(text)
        result = extract_dict(obj)
        if result is not None:
            return result
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return extract_dict(obj)
        except Exception:
            pass

    return None


def call_gemini_variant_match(api_key: str, original_image_bytes: bytes, variant_images_b64: list) -> int:
    """
    Sends the original image and a list of variant images (in base64 format)
    to Gemini and asks it to return the index of the matching variant.
    
    Returns the 0-based index of the matching variant, or 0 if it fails.
    """
    if not variant_images_b64 or len(variant_images_b64) <= 1:
        return 0

    client = genai.Client(api_key=api_key)
    
    # Base64 decode the variant images
    variant_image_bytes_list = []
    for b64_str in variant_images_b64:
        # Strip the data URI prefix if it exists (e.g. "data:image/jpeg;base64,")
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        try:
            variant_image_bytes_list.append(base64.b64decode(b64_str))
        except Exception as e:
            print(f"  Matcher error decoding variant image: {e}")
            variant_image_bytes_list.append(None)
    
    contents = []
    
    # 1. Original Image User Uploaded
    contents.append(types.Part.from_bytes(data=original_image_bytes, mime_type="image/jpeg"))
    contents.append("The image above is IMAGE 1 (the physical card photo). The following images are the official DB variants.")
    
    # 2. Append all variant images
    for idx, v_bytes in enumerate(variant_image_bytes_list):
        if v_bytes:
            contents.append(types.Part.from_bytes(data=v_bytes, mime_type="image/jpeg"))
            contents.append(f"The image above is IMAGE {idx + 2} (Variant Index {idx}).")
            
    # 3. Append the prompt
    contents.append(VARIANT_MATCH_PROMPT.strip())

    for model in MODELS:
        try:
            print(f"  Matcher sending {1 + len(variant_images_b64)} images to Gemini ({model})...")
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=gemini_config,
            )
            parsed = _parse_json((resp.text or "").strip())
            
            if parsed and "matched_variant_index" in parsed:
                idx = int(parsed["matched_variant_index"])
                if 0 <= idx < len(variant_images_b64):
                    print(f"  Matcher successfully identified match: index {idx}")
                    return idx
                else:
                    print(f"  Matcher returned out-of-bounds index: {idx}")
            else:
                print(f"  Matcher could not parse valid index from JSON: {parsed}")
                
            return 0  # Fallback
            
        except Exception as e:
            err = str(e)
            print(f"  Matcher model {model} failed: {err}")
            if "429" in err or "quota" in err.lower():
                m2 = re.search(r"retry in (\d+)", err)
                time.sleep(int(m2.group(1)) + 2 if m2 else 30)
            continue
            
    print("  Matcher: All models failed or quota exceeded.")
    return 0  # Fallback to index 0
