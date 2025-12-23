
import argparse, base64, requests, sys

def b64read(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def save_b64(b64s, out):
    with open(out, "wb") as f:
        f.write(base64.b64decode(b64s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI REST base URL")
    ap.add_argument("--mode", choices=["txt2img","img2img","inpaint"], required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", default="")
    ap.add_argument("--mask", default="")
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.mode == "txt2img":
        payload = {"prompt": args.prompt}
        r = requests.post(args.url.rstrip("/") + "/txt2img", json=payload, timeout=600)
    elif args.mode == "img2img":
        if not args.image: sys.exit("--image required for img2img")
        payload = {"prompt": args.prompt, "strength": args.strength, "image": b64read(args.image)}
        r = requests.post(args.url.rstrip("/") + "/img2img", json=payload, timeout=600)
    else:
        if not args.image or not args.mask: sys.exit("--image and --mask required for inpaint")
        payload = {"prompt": args.prompt, "strength": args.strength, "image": b64read(args.image), "mask": b64read(args.mask)}
        r = requests.post(args.url.rstrip("/") + "/inpaint", json=payload, timeout=600)

    r.raise_for_status()
    data = r.json()
    b64s = data.get("image","")
    if not b64s:
        sys.exit("No image returned by ComfyUI endpoint. Adapt endpoints to your workflow.")
    save_b64(b64s, args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
