"""PDF processing utilities: text extraction, OCR fallback and math formula extraction.

Features:
- Extract text from text-based PDFs (PyMuPDF)
- If page has little or no text, render page to image and OCR with pytesseract
- Optional MathPix integration (set MATHPIX_APP_ID and MATHPIX_APP_KEY env vars)
"""
from __future__ import annotations

import os
import io
import logging
import base64
from typing import List, Tuple, Dict, Optional

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

import requests

logger = logging.getLogger(__name__)


def _call_mathpix(image_bytes: bytes) -> Optional[Dict]:
    """Call MathPix API to extract LaTeX from an image (if configured).

    Requires environment variables: MATHPIX_APP_ID, MATHPIX_APP_KEY
    """
    app_id = os.getenv("MATHPIX_APP_ID")
    app_key = os.getenv("MATHPIX_APP_KEY")
    if not app_id or not app_key:
        logger.debug("MathPix credentials not set")
        return None

    headers = {
        "app_id": app_id,
        "app_key": app_key,
        "Content-type": "application/json",
    }
    img_b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {"src": f"data:image/png;base64,{img_b64}", "formats": ["latex_simplified"]}
    try:
        resp = requests.post("https://api.mathpix.com/v3/text", json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"MathPix request failed: {e}")
        return None


def extract_text_and_formulas_from_pdf(
    file_path: str,
    ocr_if_needed: bool = True,
    mathpix: bool = True,
    ocr_lang: str = "eng"
) -> Tuple[str, List[Dict]]:
    """Extract text from a PDF file and detect formulas.

    Returns (full_text, formulas) where formulas is a list of dicts:
    {"page": int, "latex": str, "confidence": float, "raw": str}
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required for PDF processing")

    doc = fitz.open(file_path)
    full_text_pages: List[str] = []
    formulas: List[Dict] = []

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        # extract text
        try:
            page_text = page.get_text("text") or ""
        except Exception:
            page_text = ""

        # if page text is small and OCR is allowed, rasterize and OCR
        if (not page_text.strip() or len(page_text.split()) < 20) and ocr_if_needed:
            if fitz is None or Image is None or pytesseract is None:
                logger.debug("OCR dependencies missing; skipping OCR for page %d", page_idx + 1)
            else:
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                try:
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    ocr_text = pytesseract.image_to_string(pil_img, lang=ocr_lang)
                    page_text = (page_text + "\n" + ocr_text).strip()
                except Exception as e:
                    logger.warning(f"OCR failed on page {page_idx+1}: {e}")

                # try math extraction via MathPix on the whole page image
                if mathpix:
                    mp_resp = _call_mathpix(img_bytes)
                    if mp_resp and isinstance(mp_resp, dict):
                        latex = mp_resp.get("latex_simplified") or mp_resp.get("latex")
                        if latex:
                            formulas.append({
                                "page": page_idx + 1,
                                "latex": latex,
                                "raw": mp_resp,
                                "confidence": mp_resp.get("confidence", 0.0),
                            })

        # also try to detect inline LaTeX fragments in page_text
        # simple heuristic: look for $...$ and \[ ... \]
        inline_formula_candidates = []
        try:
            import re

            inline_formula_candidates += re.findall(r"\$\$(.+?)\$\$", page_text, flags=re.S)
            inline_formula_candidates += re.findall(r"\$(.+?)\$", page_text, flags=re.S)
            inline_formula_candidates += re.findall(r"\\\[(.+?)\\\]", page_text, flags=re.S)
        except Exception:
            inline_formula_candidates = []

        for cand in inline_formula_candidates:
            cand_clean = cand.strip()
            if cand_clean:
                formulas.append({"page": page_idx + 1, "latex": cand_clean, "raw": cand_clean, "confidence": 1.0})

        full_text_pages.append(page_text)

    full_text = "\n\n".join(full_text_pages)
    return full_text, formulas


def is_pdf_image_based(file_path: str) -> bool:
    """Quick heuristic: true if majority of pages have little text"""
    if fitz is None:
        return False
    doc = fitz.open(file_path)
    image_pages = 0
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        if len(text.split()) < 30:
            image_pages += 1
    return image_pages >= (len(doc) / 2)
