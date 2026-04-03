"""
sketch_generator_new.py
=======================
High-quality pencil sketch generator — face + upper-body crop.

Background removal strategy (priority order):
  1. rembg  — deep-learning salient-object segmentation (best quality,
              requires `pip install rembg`). Runs on CPU automatically.
  2. GrabCut fallback — used if rembg is unavailable or errors out.

Only the person (face + torso) is kept. Background is replaced with
pure white before the sketch pipeline runs, so no desk / wall / floor
texture bleeds into the engraved output.
"""

import cv2
import numpy as np
import os

# ── rembg availability check ──────────────────────────────────────────────────
try:
    from rembg import remove as _rembg_remove
    from PIL import Image as _PILImage
    import io as _io
    _REMBG_AVAILABLE = True
    print("✓ rembg available — using deep-learning background removal")
except ImportError:
    _REMBG_AVAILABLE = False
    print("⚠  rembg not found — falling back to GrabCut  "
          "(pip install rembg for better quality)")


# ─────────────────────────────────────────────────────────────────────────────
#  Background removal — rembg (primary) + GrabCut (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _remove_background_rembg(img_bgr: np.ndarray) -> np.ndarray:
    """
    Use rembg (U2-Net) to isolate the foreground person.

    Converts BGR→RGBA via rembg, then composites the person on a
    pure-white background. The alpha channel gives pixel-accurate
    separation of hair, shoulders, and clothing edges.

    Returns BGR uint8 with white background.
    """
    # BGR → RGB PIL image
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_in = _PILImage.fromarray(rgb)

    # Run rembg (downloads U2-Net weights on first call ~170 MB)
    buf_in = _io.BytesIO()
    pil_in.save(buf_in, format="PNG")
    buf_in.seek(0)
    buf_out = _io.BytesIO(_rembg_remove(buf_in.read()))
    pil_out = _PILImage.open(buf_out).convert("RGBA")

    # Composite on white
    white = _PILImage.new("RGBA", pil_out.size, (255, 255, 255, 255))
    white.paste(pil_out, mask=pil_out.split()[3])
    result_rgb = np.array(white.convert("RGB"))
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def _remove_background_grabcut(img_bgr: np.ndarray) -> np.ndarray:
    """
    GrabCut fallback — good for centred portrait crops.
    Leaves only 6% margin L/R and 3%/5% top/bottom so GrabCut
    knows the narrow strip outside is background.
    """
    try:
        h, w = img_bgr.shape[:2]
        if h < 20 or w < 20:
            return img_bgr

        mx = max(4, int(w * 0.06))
        mt = max(3, int(h * 0.03))
        mb = max(4, int(h * 0.05))
        rect = (mx, mt, w - 2 * mx, h - mt - mb)

        mask      = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model,
                    iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close, iterations=3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  k_open,  iterations=1)

        result = img_bgr.copy()
        result[fg_mask == 0] = (255, 255, 255)
        return result

    except Exception as e:
        print(f"[sketch_generator] GrabCut failed ({e}), using original")
        return img_bgr


def _remove_background(img_bgr: np.ndarray) -> np.ndarray:
    """
    Dispatcher: try rembg first, fall back to GrabCut.
    Always returns BGR uint8 with white background.
    """
    if _REMBG_AVAILABLE:
        try:
            return _remove_background_rembg(img_bgr)
        except Exception as e:
            print(f"[sketch_generator] rembg error ({e}), falling back to GrabCut")
    return _remove_background_grabcut(img_bgr)


# ─────────────────────────────────────────────────────────────────────────────
#  Core parametric sketch builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_sketch(
    img_bgr:           np.ndarray,
    *,
    clahe_clip:        float = 2.0,
    clahe_tile:        int   = 8,
    blur_scale:        float = 0.05,
    blur_scale_min:    int   = 25,
    sharpen_strength:  float = 1.0,
    gamma:             float = 1.0,
    brightness_offset: int   = 0,
) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if brightness_offset != 0:
        gray = np.clip(gray.astype(np.int16) + brightness_offset, 0, 255).astype(np.uint8)

    if gamma != 1.0:
        lut  = np.array([min(255, int(((i / 255.0) ** gamma) * 255))
                         for i in range(256)], dtype=np.uint8)
        gray = cv2.LUT(gray, lut)

    clahe    = cv2.createCLAHE(clipLimit=clahe_clip,
                                tileGridSize=(clahe_tile, clahe_tile))
    gray_enh = clahe.apply(gray)

    invert      = cv2.bitwise_not(gray_enh)
    h, w        = gray_enh.shape[:2]
    k           = max(blur_scale_min, int(w * blur_scale) | 1)
    blur        = cv2.GaussianBlur(invert, (k, k), 0)
    invert_blur = cv2.bitwise_not(blur)
    sketch      = cv2.divide(gray_enh, invert_blur, scale=256.0)

    if sharpen_strength > 0:
        center = 1.0 + 8.0 * sharpen_strength
        edge   = -sharpen_strength
        kernel = np.array([[edge,  edge,  edge],
                            [edge, center, edge],
                            [edge,  edge,  edge]], dtype=np.float32)
        sketch = cv2.filter2D(sketch, -1, kernel)
        sketch = np.clip(sketch, 0, 255).astype(np.uint8)

    return sketch


# ─────────────────────────────────────────────────────────────────────────────
#  Variant presets
# ─────────────────────────────────────────────────────────────────────────────

SKETCH_VARIANTS = [
    ("Standard",      2.0,  8, 0.050, 25, 1.0, 1.00,   0),
    ("Soft",          1.5,  8, 0.040, 21, 0.5, 0.90,  10),
    ("Deep Contrast", 2.5,  8, 0.055, 27, 1.5, 1.10,  -5),
    ("Fine Detail",   2.0,  6, 0.025, 13, 1.2, 1.00,   0),
    ("Bold Strokes",  2.0, 10, 0.090, 45, 0.8, 1.00,   0),
    ("Bright Soft",   1.5,  8, 0.045, 23, 0.6, 0.80,  20),
    ("Dark Crisp",    2.5,  8, 0.050, 25, 1.8, 1.20, -10),
    ("High CLAHE",    4.0,  6, 0.050, 25, 1.0, 1.00,   0),
]

NUM_VARIANTS = len(SKETCH_VARIANTS)


def _apply_variant(img_bgr: np.ndarray, idx: int) -> np.ndarray:
    _, clip, tile, bscale, bmin, sharp, gam, bright = SKETCH_VARIANTS[idx]
    return _build_sketch(
        img_bgr,
        clahe_clip        = clip,
        clahe_tile        = tile,
        blur_scale        = bscale,
        blur_scale_min    = bmin,
        sharpen_strength  = sharp,
        gamma             = gam,
        brightness_offset = bright,
    )


def _score_sketch(sketch_gray: np.ndarray) -> float:
    lap_var   = cv2.Laplacian(sketch_gray, cv2.CV_64F).var()
    dyn_range = float(sketch_gray.max()) - float(sketch_gray.min())
    mean_val  = float(sketch_gray.mean())

    if mean_val < 80:
        penalty = (80 - mean_val) / 80 * 0.5
    elif mean_val > 220:
        penalty = (mean_val - 220) / 35 * 0.5
    else:
        penalty = 0.0

    return float(lap_var * 0.7 + dyn_range * 0.3 - penalty * 1000)


# ─────────────────────────────────────────────────────────────────────────────
#  Canvas builder
# ─────────────────────────────────────────────────────────────────────────────

def _put_text_fitted(canvas, text, x, y, font, font_scale, color, thickness, max_w):
    (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    if tw > max_w:
        font_scale = font_scale * max_w / tw
    cv2.putText(canvas, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _laser_canvas(sketch_gray, person_name, company,
                  variant_label="Standard", variant_num=1):
    h, w       = sketch_gray.shape[:2]
    sketch_bgr = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

    HEADER_H = 64
    FOOTER_H = 52
    canvas   = np.ones((HEADER_H + h + FOOTER_H, w, 3), dtype=np.uint8) * 255

    cv2.rectangle(canvas, (0, 0),            (w, HEADER_H),               (30, 30, 30), -1)
    cv2.rectangle(canvas, (0, HEADER_H + h), (w, HEADER_H + h + FOOTER_H), (30, 30, 30), -1)

    canvas[HEADER_H: HEADER_H + h, :] = sketch_bgr

    font = cv2.FONT_HERSHEY_DUPLEX

    _put_text_fitted(canvas, person_name.upper(),
                     20, HEADER_H - 16, font, 1.0, (255, 255, 255), 2, max(w // 2, 60))

    badge = f"#{variant_num} {variant_label}"
    bfs   = 0.55
    (btw, _), _ = cv2.getTextSize(badge, font, bfs, 1)
    cv2.putText(canvas, badge,
                (w - btw - 14, HEADER_H - 16),
                font, bfs, (180, 180, 255), 1, cv2.LINE_AA)

    engrave_text = f"Thanking {person_name}  \u2014  from {company}"
    _put_text_fitted(canvas, engrave_text,
                     20, HEADER_H + h + int(FOOTER_H * 0.65),
                     font, 0.60, (255, 255, 255), 1, w - 40)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_sketch(image_path: str, output_path: str) -> bool:
    """Plain sketch (variant 1, background removed)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        sketch = _apply_variant(_remove_background(img), 0)
        cv2.imwrite(output_path, sketch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"[generate_sketch] {e}")
        return False


def generate_sketch_with_label(image_path: str, output_path: str,
                                person_name: str,
                                position_text: str = None) -> bool:
    """Labelled sketch (variant 1, background removed)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        img_clean   = _remove_background(img)
        sketch_gray = _apply_variant(img_clean, 0)
        h, w        = sketch_gray.shape[:2]
        sketch_bgr  = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

        BAR    = 72
        canvas = np.ones((h + BAR, w, 3), dtype=np.uint8) * 255
        cv2.line(canvas, (0, BAR - 2), (w, BAR - 2), (180, 180, 180), 1)
        canvas[BAR:, :] = sketch_bgr

        font = cv2.FONT_HERSHEY_DUPLEX
        _put_text_fitted(canvas, person_name.upper(),
                         20, BAR - 22, font, 1.1, (30, 30, 30), 2, w - 40)
        if position_text:
            pfs   = 0.52
            (ptw, _), _ = cv2.getTextSize(position_text, font, pfs, 1)
            if ptw > w - 40:
                mc = int(len(position_text) * (w - 40) / ptw)
                position_text = position_text[:mc - 1] + "..."
            cv2.putText(canvas, position_text, (20, h + BAR - 10),
                        font, pfs, (100, 100, 100), 1, cv2.LINE_AA)

        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"[generate_sketch_with_label] {e}")
        return False


def generate_sketch_for_laser(image_path: str, output_path: str,
                               person_name: str,
                               company: str = "AABBCC") -> bool:
    """Laser-ready sketch — variant 1, background removed."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        img_clean   = _remove_background(img)
        sketch_gray = _apply_variant(img_clean, 0)
        canvas      = _laser_canvas(sketch_gray, person_name, company,
                                    variant_label="Standard", variant_num=1)
        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"[generate_sketch_for_laser] {e}")
        return False


def generate_sketch_variations(
    image_path:  str,
    output_dir:  str,
    person_name: str,
    company:     str = "AABBCC",
    base_name:   str = None,
) -> list:
    """
    Generate all 8 sketch variants with background removed.
    Background is removed ONCE (rembg or GrabCut) and shared across all variants.

    Returns list of dicts sorted best-first:
      [{ path, filename, variant_num, variant_label, score, is_best }, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[generate_sketch_variations] cannot read {image_path}")
        return []

    if base_name is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Remove background ONCE — shared by all 8 variants
    print(f"[sketch] Removing background ({'rembg' if _REMBG_AVAILABLE else 'GrabCut'})…")
    img_clean = _remove_background(img)

    results = []
    for i, (label, *_) in enumerate(SKETCH_VARIANTS):
        try:
            sketch_gray = _apply_variant(img_clean, i)
            canvas      = _laser_canvas(sketch_gray, person_name, company,
                                        variant_label=label, variant_num=i + 1)
            safe_label  = label.replace(" ", "")
            out_name    = f"sketch_{base_name}_v{i+1}_{safe_label}.jpg"
            out_path    = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

            score = _score_sketch(sketch_gray)
            results.append({
                "path":          out_path,
                "filename":      out_name,
                "variant_num":   i + 1,
                "variant_label": label,
                "score":         score,
                "is_best":       False,
            })
            print(f"  [sketch v{i+1}/{NUM_VARIANTS}] {out_name}  score={score:.1f}")
        except Exception as e:
            print(f"  [sketch v{i+1}] failed: {e}")

    if results:
        results.sort(key=lambda r: r["score"], reverse=True)
        results[0]["is_best"] = True
        print(f"[sketch] Best: v{results[0]['variant_num']} "
              f"{results[0]['variant_label']}  score={results[0]['score']:.1f}")

    return results