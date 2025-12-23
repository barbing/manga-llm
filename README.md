
# Local Manga Translator & Typesetter (Offline)

This toolkit lets you batch **OCR Japanese manga**, **translate offline**, and **typeset** the translation back onto images. Designed to run fully local on Windows/macOS/Linux with an NVIDIA GPU (recommended) or CPU/Apple Silicon (slower).

## What you get
- **OCR**: PaddleOCR (Japanese) for text *boxes* + Manga-OCR for *high-accuracy Japanese text*.
- **Translation (offline)**: Hugging Face **Helsinki-NLP/opus-mt-ja-en** by default (swap for ja->zh or other pairs).
- **Inpainting & typesetting**: Remove original bubbles and draw translated text with auto font sizing/line wrapping.
- **Batch mode**: Point at a folder of PNG/JPGs; outputs edited images and a CSV of extracted/translated text.


---

## 1) Environment setup

### Windows (Conda)
```bash
# Install Miniconda if you don't have it
# https://docs.conda.io/en/latest/miniconda.html

conda create -n manga-llm python=3.10 -y
conda activate manga-llm

# (Optional, NVIDIA GPU) Install PyTorch with CUDA (choose version for your GPU)
# Check https://pytorch.org/get-started/locally/ for the exact command.
# Example for CUDA 12.1:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# CPU/Apple Silicon:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Now install the rest
pip install -r requirements.txt
```

### macOS (Apple Silicon OK)
```bash
conda create -n manga-llm python=3.10 -y
conda activate manga-llm

# CPU/Metal build (PyTorch for macOS):
pip install torch torchvision torchaudio

pip install -r requirements.txt
```

### Linux
```bash
conda create -n manga-llm python=3.10 -y
conda activate manga-llm
# Install Torch per https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

---

## 2) Configure
Edit `config.yaml` to choose:
- `target_lang`: `en` (English) or `zh` (Chinese) or other.
- `translator_model`: defaults to `Helsinki-NLP/opus-mt-ja-en`. For **ja->zh**, try `Helsinki-NLP/opus-mt-ja-zh`.
- `font_path`: path to a TTF font (e.g., *CC Wild Words*, *Anime Ace*, or a local CJK font for Chinese).
- `max_font_size`, `min_font_size`, `line_spacing`, `bubble_padding`.

---

## 3) Run
Put images into a folder, e.g. `./input_pages`. Then run:
```bash
python translate_manga.py --input ./input_pages --output ./output_pages
```

The script will create:
- `./output_pages/<same_filename>.png` — translated/typeset pages
- `./output_pages/_extractions.csv` — OCR boxes, original JP text, translation
- `./output_pages/_debug_overlays` — debug images showing boxes

> Tip: If you see bad boxes or awkward wrapping, adjust thresholds in `config.yaml` (e.g., `min_confidence`, `bubble_padding`) and re-run.

---

## 4) Optional: Local Chat LLM (general Q&A + translator)
For a privacy-first assistant that can "talk like online LLMs":

### A) **Ollama** (easiest)
1. Install Ollama: https://ollama.com/download
2. Pull a good small instruction model:
```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
# Alternatives:
# ollama pull llama3.1:8b-instruct-q4_K_M
# ollama pull phi3.5:3.8b-instruct-q4_K_M
```
3. Chat locally:
```bash
ollama run qwen2.5:7b-instruct-q4_K_M
```
4. Use it for translation too (better nuance) by pasting JP text extracted in the CSV, or integrate via HTTP in `translate_manga.py` (simple to add later).

### B) **LM Studio** (GUI)
- Download LM Studio, pick a 7B Instruct model, and chat with a clean UI. You can also expose a local endpoint for tools.

---

## 5) Fonts
Install a manga-friendly font and set `font_path` in `config.yaml`:
- **English**: *Anime Ace*, *CC Wild Words*, *Komika Hand*, *Open Sans*.
- **Chinese**: *Noto Sans CJK*, *Source Han Sans*.
- Make sure you have the license to use the font.

---

## 6) Notes & Tweaks
- **Vertical text**: Many manga use vertical Japanese. This tool rotates crops when needed and uses Manga-OCR (trained for manga text) for accuracy.
- **Speed**: GPU + batch sizes help. CPU will be slower.
- **Edge cases**: Handwritten SFX, dense background text, or unusual fonts may require manual touch-ups.

---

## 7) Roadmap (you can add later)
- Speech bubble detection via ML (CRAFT/DB/EAST) for trickier layouts.
- Stable Diffusion inpainting for complex backgrounds (ComfyUI Automatic nodes) instead of OpenCV inpaint.
- Integration with your local LLM (Ollama) for smart post-editing of translations.


## New modes & options
- **Typeset-only mode** (skip OCR/translate, use edited CSV):
  ```bash
  python translate_manga.py --mode typeset --csv ./output_pages/_extractions.csv --input ./input_pages --output ./output_pages_redraw
  ```
  You can change which column is used via `translation_column` in `config.yaml` (e.g., `translation_ollama`).

- **Vertical text**: Set `vertical_text` in `config.yaml` to `off` | `auto` | `on` (default `off`).  
  `auto` draws vertical if bubble `height/width >= vertical_ratio_trigger`.

- **Text stroke**: `stroke_width`, `stroke_fill`, `text_fill` for better readability on busy backgrounds.

- **CSV encoding**: `csv_encoding` controls how CSV is written/read (default `utf-8-sig`).

### Windows quick scripts
- **run_full.bat**:
  ```bat
  @echo off
  conda activate manga-llm
  python translate_manga.py --input .\input_pages --output .\output_pages
  pause
  ```
- **run_typeset.bat**:
  ```bat
  @echo off
  conda activate manga-llm
  python translate_manga.py --mode typeset --csv .\output_pages\_extractions.csv --input .\input_pages --output .\output_pages_redraw
  pause
  ```


## Power-ups
- **Detectors:** choose `detector: paddle | mser | hybrid` in `config.yaml`. MSER is robust on high-contrast manga; hybrid unions both then merges.
- **SFX lane:** long/tall boxes are treated as SFX; alternate font and alpha with `sfx_*` settings.
- **Style presets:** pick `style: zh_serif | zh_sans | en_comic` (overrides font/spacing/colors). Edit the `styles:` map to add your own.
- **Stable Diffusion / ComfyUI inpaint:** set `sd_inpaint_enable: true` and run ComfyUI with a REST workflow that accepts `image+mask`. Or run `inpaint_sd.py` to export masks for manual inpaint.
- **GUI:** `python gui.py` for a small app to run pipelines and preview box overlays.

### ComfyUI quick note
If you enable SD inpainting in the future, the script will export the image and mask and try to call `http://127.0.0.1:8188/inpaint`. If it fails, assets are saved to `sd_export_masks_dir` so you can inpaint manually in ComfyUI.
