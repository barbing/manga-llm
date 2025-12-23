
import os, csv, json, time, argparse
import pandas as pd
import requests

PROMPT_TMPL = """You are a professional Japanese-to-{target_lang} manga translator and typesetter.
Translate the following Japanese text for a speech bubble. Keep natural tone, concise, and suitable for typesetting.
Do not add explanations; return only the translation.
Japanese:
{jp}
"""

def call_ollama(model: str, prompt: str, url="http://localhost:11434/api/generate"):
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to _extractions.csv produced by translate_manga.py")
    ap.add_argument("--model", default="qwen3:8b-q4_K_M", help="Ollama model tag")
    ap.add_argument("--target_lang", default="Chinese", help="Target language description for prompt (e.g., English, Chinese)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite original 'translation' column")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    out_col = "translation_ollama"
    results = []

    for i, row in df.iterrows():
        jp = str(row.get("jp", "")).strip()
        if not jp:
            results.append("")
            continue
        prompt = PROMPT_TMPL.format(target_lang=args.target_lang, jp=jp)
        try:
            tr = call_ollama(args.model, prompt)
        except Exception as e:
            tr = f"[ERROR: {e}]"
        results.append(tr)
        if (i+1) % 20 == 0:
            print(f"Processed {i+1}/{len(df)}")

    if args.overwrite:
        df["translation"] = results
    else:
        df[out_col] = results

    df.to_csv(args.csv, index=False, encoding="utf-8-sig")
    print("Updated:", args.csv)

if __name__ == "__main__":
    main()
