import json
import urllib.request


def translate_texts(texts, model, target_lang, style, glossary=None, endpoint="http://127.0.0.1:11434/api/generate"):
    prompt_prefix = f"Translate Japanese to {target_lang}. Style: {style}."
    if glossary:
        prompt_prefix += f" Use glossary: {json.dumps(glossary, ensure_ascii=False)}."

    results = []
    for text in texts:
        if not text.strip():
            results.append("")
            continue
        payload = {
            "model": model,
            "prompt": f"{prompt_prefix}\n\n{text}",
            "stream": False,
        }
        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results.append(data.get("response", "").strip())
    return results
