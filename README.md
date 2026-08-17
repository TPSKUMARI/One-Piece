# One Piece Card Recognition

Experiments in recognizing One Piece TCG cards from photos — matching a scanned card against a card database and identifying its number, rarity, and variant.

## What's here

Several different approaches, tried side by side:

- **`gemini_based_one_piece_card_ocr/`** — Uses Google Gemini to read card details directly from an image.
- **`gemini_with_rapid_ocr_one_piece_japaniece_card/`** — Combines Gemini with RapidOCR for Japanese-language cards.
- **`Paddle_ocr_testing/`** — Card text extraction using PaddleOCR.
- **`one_piece_card_matching_orb_based_method/`** — A Flask web app that matches cards using ORB feature matching plus OCR for card number/rarity, with a simple browser UI (`index.html`).
- **`data/`** — Sample card images used for testing.
- **`test_original_capctures/`** — Test captures, split into cards found in the database vs. not.

## Running the Flask app

```bash
cd one_piece_card_matching_orb_based_method
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.

You'll need a `GOOGLE_API_KEY` set in a `.env` file for the Gemini-based scripts.

## Status

This is an exploratory project comparing different OCR/matching strategies to see which works best for identifying One Piece cards.
