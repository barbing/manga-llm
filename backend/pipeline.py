import argparse
import json
import os
from pathlib import Path

from backend.schema import Bubble, PagePayload


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def export_placeholder_json(image_paths, output_dir, target_lang, style):
    json_dir = ensure_dir(os.path.join(output_dir, "_translations_json"))
    for image_path in image_paths:
        page_name = os.path.basename(image_path)
        payload = PagePayload(page=page_name, target_language=target_lang, style=style)
        payload.bubbles.append(
            Bubble(
                id="b1",
                bbox=[40, 40, 240, 120],
                original_text="（OCR placeholder）",
                translation="(translation placeholder)",
                final_text="(edit me)",
            )
        )
        with open(os.path.join(json_dir, f"{Path(page_name).stem}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=lambda o: o.__dict__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with input images")
    ap.add_argument("--output", required=True, help="Folder to save outputs")
    ap.add_argument("--target-lang", default="zh-CN", help="Target language (zh-CN or en)")
    ap.add_argument("--model", default="huihui_ai/qwen3-abliterated:14b", help="Ollama model")
    ap.add_argument("--style", default="modern_casual", help="Translation style preset")
    args = ap.parse_args()

    image_paths = [
        os.path.join(args.input, f)
        for f in sorted(os.listdir(args.input))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    if not image_paths:
        raise SystemExit("No images found in input folder.")

    ensure_dir(args.output)
    export_placeholder_json(image_paths, args.output, args.target_lang, args.style)
    print(f"[scaffold] Exported placeholder JSON to {os.path.join(args.output, '_translations_json')}")


if __name__ == "__main__":
    main()
