
import os, csv, math, io
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from detectors import detect_mser_boxes
from tqdm import tqdm
import yaml
import pandas as pd

# OCR
from paddleocr import PaddleOCR
from manga_ocr import MangaOcr

# Translation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@dataclass
class OCRBox:
    box: List[List[float]]   # 4 points [[x1,y1], ...]
    text: str
    score: float

def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, size=size)
    except Exception as e:
        raise RuntimeError(f"Font load failed: {e}. Update font_path in config.yaml.")

def polygon_to_bbox(poly: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def inpaint_region(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

def textbbox(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont):
    return draw.textbbox((0,0), text, font=font)

def fit_text_into_box(draw: ImageDraw.Draw, font_path: str, text: str, box: Tuple[int,int,int,int],
                      max_size: int, min_size: int, line_spacing: float):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    for size in range(max_size, min_size-1, -2):
        font = load_font(font_path, size)
        if not text.strip():
            return [], font
        words = text.replace('\n', ' ').split(' ')
        lines = []
        cur = ""
        for w_i in words:
            trial = (cur + " " + w_i).strip()
            bw, bh = textbbox(draw, trial, font)[2:]
            if bw <= w*0.96:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w_i
        if cur:
            lines.append(cur)
        line_h = textbbox(draw, "A", font)[3] - textbbox(draw, "A", font)[1]
        total_h = int(sum(line_h * (line_spacing if i>0 else 1.0) for i in range(len(lines))))
        if total_h <= h*0.96 and lines:
            return lines, font
    return [text], load_font(font_path, min_size)

def mask_from_boxes(img_shape: Tuple[int,int,int], boxes: List[Tuple[int,int,int,int]], padding: int) -> np.ndarray:
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for (x1,y1,x2,y2) in boxes:
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(img_shape[1]-1, x2 + padding)
        y2p = min(img_shape[0]-1, y2 + padding)
        mask[y1p:y2p, x1p:x2p] = 255
    return mask

def draw_horizontal_block(draw: ImageDraw.Draw, box, lines, font, fill, stroke_fill, stroke_width, line_spacing):
    x1,y1,x2,y2 = box
    line_h = textbbox(draw, "A", font)[3] - textbbox(draw, "A", font)[1]
    total_h = sum(int(line_h*(line_spacing if i>0 else 1.0)) for i in range(len(lines)))
    cur_y = int(y1 + (y2 - y1 - total_h)/2)
    for i, line in enumerate(lines):
        bw, bh = textbbox(draw, line, font)[2:]
        cur_x = int(x1 + (x2 - x1 - bw)/2)
        draw.text((cur_x, cur_y), line, font=font, fill=tuple(fill),
                  stroke_width=stroke_width, stroke_fill=tuple(stroke_fill))
        cur_y += int(line_h * (line_spacing if i>0 else 1.0))

def draw_vertical_block(draw: ImageDraw.Draw, box, text, font, fill, stroke_fill, stroke_width, line_spacing):
    # Simple vertical layout (top->bottom), centered horizontally within the box.
    # We split into characters and place each on a new line.
    x1,y1,x2,y2 = box
    chars = [c for c in text if c.strip()]  # crude; ignores spaces/punct width
    if not chars:
        return
    line_h = textbbox(draw, "文", font)[3] - textbbox(draw, "文", font)[1]
    total_h = int(len(chars) * line_h * line_spacing)
    cur_y = int(y1 + (y2 - y1 - total_h)/2)
    # center horizontally by measuring a typical glyph
    gw = textbbox(draw, "文", font)[2] - textbbox(draw, "文", font)[0]
    cur_x = int(x1 + (x2 - x1 - gw)/2)
    for i, ch in enumerate(chars):
        draw.text((cur_x, cur_y), ch, font=font, fill=tuple(fill),
                  stroke_width=stroke_width, stroke_fill=tuple(stroke_fill))
        cur_y += int(line_h * line_spacing)

def should_draw_vertical(box, mode: str, ratio_trigger: float) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    # auto
    x1,y1,x2,y2 = box
    w = max(1, x2-x1)
    h = max(1, y2-y1)
    return (h / w) >= ratio_trigger

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, help="Folder with input images (png/jpg)")
    ap.add_argument("--output", required=True, help="Folder to save outputs")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--csv", default="", help="Typeset-only mode: path to _extractions.csv")
    ap.add_argument("--mode", default="full", choices=["full", "typeset"], help="full = OCR+translate+typeset; typeset = read boxes+translations from CSV and redraw")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)
    dbg_dir = os.path.join(args.output, "_debug_overlays")
    os.makedirs(dbg_dir, exist_ok=True)

    target_lang = cfg.get("target_lang", "en")
    translator_model = cfg.get("translator_model", "Helsinki-NLP/opus-mt-ja-en")
    font_path = cfg.get("font_path")
    max_font_size = int(cfg.get("max_font_size", 36))
    min_font_size = int(cfg.get("min_font_size", 16))
    line_spacing = float(cfg.get("line_spacing", 1.15))
    bubble_padding = int(cfg.get("bubble_padding", 6))
    min_conf = float(cfg.get("min_confidence", 0.6))
    debug_draw = bool(cfg.get("debug_draw", True))
    force_horizontal = bool(cfg.get("force_horizontal", True))  # kept for backward compat
    use_gpu = bool(cfg.get("use_gpu", False))
    translator_batch = int(cfg.get("translator_batch", 8))
    torch_dtype_cfg = cfg.get("torch_dtype", "auto")
    vertical_mode = str(cfg.get("vertical_text", "off")).lower()
    detector = str(cfg.get("detector", "paddle")).lower()
    mser_min_area = int(cfg.get("mser_min_area", 200))
    mser_max_area_ratio = float(cfg.get("mser_max_area_ratio", 0.2))
    mser_delta = int(cfg.get("mser_delta", 5))
    mser_variation = float(cfg.get("mser_variation", 0.2))
    style_key = str(cfg.get("style", "zh_sans"))
    styles = cfg.get("styles", {})
    if style_key in styles:
        st = styles[style_key]
        font_path = st.get("font_path", font_path)
        text_fill = tuple(st.get("text_fill", list(text_fill)))
        stroke_fill = tuple(st.get("stroke_fill", list(stroke_fill)))
        stroke_width = int(st.get("stroke_width", stroke_width))
        line_spacing = float(st.get("line_spacing", line_spacing))
    sfx_enable = bool(cfg.get("sfx_enable", True))
    sfx_font_path = cfg.get("sfx_font_path", font_path)
    sfx_min_aspect = float(cfg.get("sfx_min_aspect", 3.0))
    sfx_alpha = int(cfg.get("sfx_alpha", 220))
    sfx_rotation_auto = bool(cfg.get("sfx_rotation_auto", True))
    sd_inpaint_enable = bool(cfg.get("sd_inpaint_enable", False))
    sd_rest_url = cfg.get("sd_rest_url", "http://127.0.0.1:8188")
    sd_export_masks_dir = cfg.get("sd_export_masks_dir", "./sd_masks")
    sd_prompt = cfg.get("sd_prompt", "clean background where text was removed, preserve manga art style")
    sd_strength = float(cfg.get("sd_strength", 0.6))

    vertical_ratio_trigger = float(cfg.get("vertical_ratio_trigger", 1.6))
    stroke_width = int(cfg.get("stroke_width", 2))
    stroke_fill = tuple(cfg.get("stroke_fill", [255,255,255]))
    text_fill = tuple(cfg.get("text_fill", [0,0,0]))
    translation_column = cfg.get("translation_column", "translation")
    csv_encoding = cfg.get("csv_encoding", "utf-8-sig")

    if args.mode == "typeset":
        if not args.csv:
            raise SystemExit("--mode typeset requires --csv pointing to an extractions CSV.")
        df = pd.read_csv(args.csv, encoding=csv_encoding)
        # group by file, typeset each image
        files = sorted(df["file"].unique())
        for fname in tqdm(files, desc="Typesetting from CSV"):
            # prefer original image path from input if provided; else try to find in same directory as CSV's folder
            src_paths = []
            if args.input:
                src_paths.append(os.path.join(args.input, fname))
            src_paths.append(os.path.join(os.path.dirname(args.csv), fname))
            src_paths.append(fname)
            in_path = None
            for p in src_paths:
                if os.path.exists(p):
                    in_path = p
                    break
            if in_path is None:
                print(f"Skip: source image for {fname} not found.")
                continue

            img_bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"Skip unreadable: {fname}")
                continue

            sub = df[df["file"] == fname]
            boxes_xyxy = [(int(x1),int(y1),int(x2),int(y2)) for x1,y1,x2,y2 in zip(sub["x1"],sub["y1"],sub["x2"],sub["y2"])]
            translations = [str(t) if t is not np.nan else "" for t in sub[translation_column].tolist()]
            # Optional overrides per bubble:
            # - orientation: 'v' (vertical), 'h' (horizontal), 'auto'
            # - font_size: integer px
            orient_list = sub['orientation'].tolist() if 'orientation' in sub.columns else [None]*len(sub)
            fsize_list = sub['font_size'].tolist() if 'font_size' in sub.columns else [None]*len(sub)


            mask = mask_from_boxes(img_bgr.shape, boxes_xyxy, bubble_padding)
            clean = inpaint_region(img_bgr, mask)

            out_pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(out_pil)

            for idx, (box, tr) in enumerate(zip(boxes_xyxy, translations)):
            if not tr or not tr.strip():
                continue
            # Per-bubble orientation override
            ovr = orient_list[idx] if idx < len(orient_list) else None
            if isinstance(ovr, str):
                ovr = ovr.strip().lower()
            if ovr in ('v','vertical'):
                orientation_vertical = True
            elif ovr in ('h','horizontal'):
                orientation_vertical = False
            else:
                if vertical_mode == 'off':
                    orientation_vertical = False
                elif vertical_mode == 'on':
                    orientation_vertical = True
                else:
                    orientation_vertical = should_draw_vertical(box, 'auto', vertical_ratio_trigger)

            # Per-bubble font size override
            fsz = None
            if idx < len(fsize_list):
                try:
                    fsz = int(fsize_list[idx]) if str(fsize_list[idx]).strip() not in ('', 'nan', 'None') else None
                except Exception:
                    fsz = None

            # SFX heuristic: very tall or very wide boxes
            w_box = box[2]-box[0]
            h_box = box[3]-box[1]
            aspect = max(h_box/w_box if w_box>0 else 99, w_box/h_box if h_box>0 else 99)
            is_sfx = sfx_enable and (aspect >= sfx_min_aspect)
            if orientation_vertical:
                if fsz is not None:
                    font = load_font(font_path, fsz)
                else:
                    for size in range(max_font_size, min_font_size-1, -2):
                        font = load_font(font_path, size)
                        line_h = draw.textbbox((0,0), '文', font=font)[3] - draw.textbbox((0,0), '文', font=font)[1]
                        chars = [c for c in tr if c.strip()]
                        total_h = int(len(chars) * line_h * line_spacing)
                        if total_h <= (box[3]-box[1])*0.96:
                            break
                    # Fit by font size roughly: shrink until height fits with line spacing
                    for size in range(max_font_size, min_font_size-1, -2):
                        font = load_font(font_path, size)
                        line_h = draw.textbbox((0,0), "文", font=font)[3] - draw.textbbox((0,0), "文", font=font)[1]
                        chars = [c for c in tr if c.strip()]
                        total_h = int(len(chars) * line_h * line_spacing)
                        if total_h <= (box[3]-box[1])*0.96:
                            break
                    if is_sfx:
                    font = load_font(sfx_font_path or font_path, font.size)
                    # draw on overlay with alpha
                    draw_vertical_block(draw_ov, box, tr, font, text_fill, stroke_fill, stroke_width, line_spacing)
                    out_pil.alpha_composite(overlay)
                    overlay = Image.new('RGBA', out_pil.size, (0,0,0,0))
                else:
                    draw_vertical_block(draw, box, tr, font, text_fill, stroke_fill, stroke_width, line_spacing)
                else:
                    lines, font = fit_text_into_box(draw, font_path, tr, box, max_font_size, min_font_size, line_spacing)
                    if is_sfx:
                    font = load_font(sfx_font_path or font_path, font.size)
                    draw_horizontal_block(draw_ov, box, lines, font, text_fill, stroke_fill, stroke_width, line_spacing)
                    out_pil.alpha_composite(overlay)
                    overlay = Image.new('RGBA', out_pil.size, (0,0,0,0))
                else:
                    draw_horizontal_block(draw, box, lines, font, text_fill, stroke_fill, stroke_width, line_spacing)

            out_img = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
            out_path = os.path.join(args.output, os.path.splitext(fname)[0] + ".png")
            cv2.imwrite(out_path, out_img)

            if debug_draw:
                dbg = img_bgr.copy()
                for (x1,y1,x2,y2) in boxes_xyxy:
                    cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.imwrite(os.path.join(dbg_dir, os.path.splitext(fname)[0] + "_boxes.png"), dbg)
        print("Typeset-only mode done.")
        return

    # ===== full mode (OCR + translation + typeset) =====
    print("Loading OCR engines...")
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="japan", use_gpu=use_gpu)
    except Exception:
        ocr = PaddleOCR(use_angle_cls=True, lang="japan")
    mocr = MangaOcr()

    print(f"Loading translator: {translator_model}")
    tok = AutoTokenizer.from_pretrained(translator_model)
    dtype_map = {"auto": None, "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(str(torch_dtype_cfg).lower(), None)
    if use_gpu and torch.cuda.is_available():
        mod = AutoModelForSeq2SeqLM.from_pretrained(translator_model, torch_dtype=torch_dtype if torch_dtype else None).to("cuda")
        device = "cuda"
    else:
        mod = AutoModelForSeq2SeqLM.from_pretrained(translator_model)
        device = "cpu"

    os.makedirs(args.output, exist_ok=True)
    rows = []
    images = [f for f in os.listdir(args.input) if f.lower().endswith((".png",".jpg",".jpeg"))]
    images.sort()

    for fname in tqdm(images, desc="Processing pages"):
        in_path = os.path.join(args.input, fname)
        img_bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Skip unreadable: {fname}")
            continue
        h, w = img_bgr.shape[:2]

        
        # 1) Text box detection
        boxes_xyxy = []
        jp_texts = []

        used_paddle = False
        if detector in ("paddle","hybrid"):
            ocr_res = ocr.ocr(in_path, cls=True)
            if isinstance(ocr_res, list) and len(ocr_res) > 0 and isinstance(ocr_res[0], list):
                used_paddle = True
                for line in ocr_res[0]:
                    poly = line[0]
                    (txt, score) = line[1]
                    if score < min_conf:
                        continue
                    x1,y1,x2,y2 = polygon_to_bbox(poly)
                    boxes_xyxy.append((x1,y1,x2,y2))

        if detector in ("mser","hybrid"):
            mser_boxes = detect_mser_boxes(img_bgr, min_area=mser_min_area, max_area_ratio=mser_max_area_ratio, delta=mser_delta, max_variation=mser_variation)
            boxes_xyxy.extend(mser_boxes)

        # Deduplicate boxes
        def nms_union(boxes, iou=0.3):
            if not boxes: return []
            import numpy as np
            keep = []
            used = [False]*len(boxes)
            def IoU(a,b):
                ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
                x1=max(ax1,bx1);y1=max(ay1,by1);x2=min(ax2,bx2);y2=min(ay2,by2)
                iw=max(0,x2-x1); ih=max(0,y2-y1); inter=iw*ih
                if inter==0: return 0.0
                area_a=(ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by1)
                return inter/(area_a+area_b-inter+1e-6)
            for i in range(len(boxes)):
                if used[i]: continue
                cur = list(boxes[i])
                used[i] = True
                merged=True
                while merged:
                    merged=False
                    for j in range(i+1,len(boxes)):
                        if used[j]: continue
                        if IoU(cur, boxes[j]) > iou:
                            cur=[min(cur[0],boxes[j][0]),min(cur[1],boxes[j][1]),max(cur[2],boxes[j][2]),max(cur[3],boxes[j][3])]
                            used[j]=True; merged=True
                keep.append(tuple(cur))
            return keep

        boxes_xyxy = nms_union(boxes_xyxy, iou=0.25)

        if not boxes_xyxy:
            # Fallback: whole image
            h, w = img_bgr.shape[:2]
            boxes_xyxy = [(int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))]

        # OCR text using Manga-OCR per box
        for (x1,y1,x2,y2) in boxes_xyxy:
            crop = img_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            jp = mocr(pil_crop)
            jp_texts.append(jp)

        # 2) Batched translation
        translations = []
        def translate_batch(texts):
            if not texts:
                return []
            nonempty_idx = [i for i,t in enumerate(texts) if str(t).strip()]
            outs = [""]*len(texts)
            if not nonempty_idx:
                return outs
            filtered = [texts[i] for i in nonempty_idx]
            inputs = tok(filtered, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.inference_mode():
                gen = mod.generate(**inputs, max_new_tokens=256, num_beams=4)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            for j, idx in enumerate(nonempty_idx):
                outs[idx] = dec[j]
            return outs

        for i in range(0, len(jp_texts), translator_batch):
            chunk = jp_texts[i:i+translator_batch]
            translations.extend(translate_batch(chunk))

        # Save extraction rows
        for (box, jp, tr) in zip(boxes_xyxy, jp_texts, translations):
            rows.append({
                "file": fname,
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "jp": jp, "translation": tr
            })

        # 3) Inpaint & typeset
        mask = mask_from_boxes(img_bgr.shape, boxes_xyxy, bubble_padding)
        clean = inpaint_region(img_bgr, mask)

        out_pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(out_pil)
        # RGBA overlay for SFX opacity
        overlay = Image.new('RGBA', out_pil.size, (0,0,0,0))
        draw_ov = ImageDraw.Draw(overlay)

        for (box, tr) in zip(boxes_xyxy, translations):
            if not tr.strip():
                continue

            orientation_vertical = should_draw_vertical(box, vertical_mode, vertical_ratio_trigger)
            # SFX heuristic: very tall or very wide boxes
            w_box = box[2]-box[0]
            h_box = box[3]-box[1]
            aspect = max(h_box/w_box if w_box>0 else 99, w_box/h_box if h_box>0 else 99)
            is_sfx = sfx_enable and (aspect >= sfx_min_aspect)
            if orientation_vertical:
                for size in range(max_font_size, min_font_size-1, -2):
                    font = load_font(font_path, size)
                    line_h = draw.textbbox((0,0), "文", font=font)[3] - draw.textbbox((0,0), "文", font=font)[1]
                    chars = [c for c in tr if c.strip()]
                    total_h = int(len(chars) * line_h * line_spacing)
                    if total_h <= (box[3]-box[1])*0.96:
                        break
                if is_sfx:
                    font = load_font(sfx_font_path or font_path, font.size)
                    # draw on overlay with alpha
                    draw_vertical_block(draw_ov, box, tr, font, text_fill, stroke_fill, stroke_width, line_spacing)
                    out_pil.alpha_composite(overlay)
                    overlay = Image.new('RGBA', out_pil.size, (0,0,0,0))
                else:
                    draw_vertical_block(draw, box, tr, font, text_fill, stroke_fill, stroke_width, line_spacing)
            else:
                lines, font = fit_text_into_box(draw, font_path, tr, box, max_font_size, min_font_size, line_spacing)
                if is_sfx:
                    font = load_font(sfx_font_path or font_path, font.size)
                    draw_horizontal_block(draw_ov, box, lines, font, text_fill, stroke_fill, stroke_width, line_spacing)
                    out_pil.alpha_composite(overlay)
                    overlay = Image.new('RGBA', out_pil.size, (0,0,0,0))
                else:
                    draw_horizontal_block(draw, box, lines, font, text_fill, stroke_fill, stroke_width, line_spacing)

        out_img = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output, os.path.splitext(fname)[0] + ".png")
        cv2.imwrite(out_path, out_img)

        if debug_draw:
            dbg = img_bgr.copy()
            for (x1,y1,x2,y2) in boxes_xyxy:
                cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.imwrite(os.path.join(dbg_dir, os.path.splitext(fname)[0] + "_boxes.png"), dbg)

    # Write CSV
    csv_path = os.path.join(args.output, "_extractions.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding=csv_encoding)
    print(f"Done. Outputs saved to: {args.output}")
    print(f"Extractions saved to: {csv_path}")

if __name__ == "__main__":
    main()
