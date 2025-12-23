
import os, json, argparse, base64
import cv2
import numpy as np
import requests

def export_masks(image_path, mask_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_img = os.path.join(out_dir, base + "_img.png")
    out_mask = os.path.join(out_dir, base + "_mask.png")
    cv2.imwrite(out_img, cv2.imread(image_path, cv2.IMREAD_COLOR))
    cv2.imwrite(out_mask, cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
    return out_img, out_mask

def comfy_inpaint(url, image_path, mask_path, prompt="clean background", strength=0.6):
    # Simple REST caller assuming a ComfyUI workflow that accepts image+mask b64.
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(mask_path, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "prompt": prompt,
        "strength": strength,
        "image": img_b64,
        "mask": mask_b64
    }
    r = requests.post(url.rstrip("/") + "/inpaint", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    # Expecting {'image': '<b64>'}
    out_b64 = data.get("image", "")
    if not out_b64:
        raise RuntimeError("No image in ComfyUI response")
    out_bytes = base64.b64decode(out_b64)
    return out_bytes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI REST base url")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--prompt", default="clean background")
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        img_bytes = comfy_inpaint(args.url, args.image, args.mask, args.prompt, args.strength)
        with open(args.out, "wb") as f:
            f.write(img_bytes)
        print("Saved:", args.out)
    except Exception as e:
        # If ComfyUI not running, export files for manual workflow
        img, mask = export_masks(args.image, args.mask, os.path.dirname(args.out) or ".")
        print("ComfyUI call failed:", e)
        print("Exported images for manual inpaint:", img, mask)
