# Manga Translator Remake (Electron + Vue)

This repository is a **clean rebuild** of the local manga translator. The old Python scripts are still present for reference, but the new implementation lives under:

- `app/` — Electron + Vue GUI (one-click workflow).
- `backend/` — New pipeline scaffold (Ollama-first, JSON-based).

## Goals (v2)
- One-screen GUI: import → output → start.
- Output language selection: **Simplified Chinese** or **English**.
- JSON workflow: export → human edit → re-render (no re-translation).
- Ollama-based translation with selectable models.
- Scalable architecture to support visual novel (VN) workflows later.

---

## 1) App (Electron + Vue)
The GUI uses Vue via CDN for fast iteration (no build step required during early development).

### Run (development)
```bash
cd app
npm install
npm run dev
```

---

## 2) Backend (Pipeline Scaffold)
The backend contains a lightweight, modular structure to evolve into:
- Bubble detection
- Manga OCR
- Ollama translation
- Inpainting + typeset
- JSON export/import

Current status: **scaffold + JSON contract + CLI skeleton**.

### Run (placeholder)
```bash
python -m backend.pipeline \
  --input ./input_pages \
  --output ./output_pages \
  --target-lang zh \
  --model huihui_ai/qwen3-abliterated:14b
```

---

## 3) JSON Output Format
Every processed page exports an editable JSON file:
```json
{
  "page": "001.png",
  "target_language": "zh-CN",
  "style": "modern_casual",
  "bubbles": [
    {
      "id": "b1",
      "bbox": [x, y, w, h],
      "original_text": "…",
      "translation": "…",
      "final_text": "…"
    }
  ]
}
```

---

## 4) Status
This is a **new codebase**. The focus is on:
1. Building the Electron GUI.
2. Finalizing the JSON workflow.
3. Incrementally swapping placeholders for real OCR + translation + typeset.

---

## 5) Legacy Reference
Legacy scripts remain in repo root for reference, but new development should be done under `app/` and `backend/`.
