
import argparse, subprocess, os, sys, tempfile, shutil, requests

def run(cmd):
    print(">", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout); print(p.stderr, file=sys.stderr); raise SystemExit(p.returncode)
    return p.stdout

def trim(args):
    run(["ffmpeg","-y","-ss",args.start,"-to",args.end,"-i",args.input,"-c","copy",args.out]); print("Saved:", args.out)

def concat_cmd(args):
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",args.list,"-c","copy",args.out]); print("Saved:", args.out)

def burn_subs(args):
    vf = f"subtitles={args.srt}:force_style='OutlineColour=&H000000&,BorderStyle=3,Outline=2'"
    run(["ffmpeg","-y","-i",args.input,"-vf",vf,"-c:a","copy",args.out]); print("Saved:", args.out)

def whisper_transcribe(input_path, lang="ja"):
    tmpdir = tempfile.mkdtemp()
    try:
        # Expect whisper.cpp binary 'whisper' in PATH
        run(["whisper", input_path, "--language", lang, "--model", "base", "--output_srt", "--output_dir", tmpdir])
        for f in os.listdir(tmpdir):
            if f.endswith(".srt"):
                return os.path.join(tmpdir, f), tmpdir
    except Exception as e:
        shutil.rmtree(tmpdir)
        raise SystemExit("ASR failed. Install whisper.cpp or faster-whisper CLI and adapt function.")
    return None, None

def translate_srt_with_ollama(srt_in, srt_out, model, target="Chinese"):
    def ask(prompt):
        r = requests.post("http://localhost:11434/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=300)
        r.raise_for_status()
        return r.json()["response"].strip()
    with open(srt_in,"r",encoding="utf-8") as f:
        src=f.read()
    blocks = src.split("\n\n")
    out_blocks = []
    for b in blocks:
        prompt = f"Translate this SRT block to {target}. Keep all timecodes and numbering exactly.\n\n{b}"
        out_blocks.append(ask(prompt))
    with open(srt_out,"w",encoding="utf-8") as f:
        f.write("\n\n".join(out_blocks))

def autosub(args):
    srt_src, tmp = whisper_transcribe(args.input, lang=args.lang)
    srt_zh = os.path.join(tmp, "translated.srt")
    translate_srt_with_ollama(srt_src, srt_zh, model=args.model, target="Chinese")
    burn_subs(argparse.Namespace(input=args.input, srt=srt_zh, out=args.out))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1=sub.add_parser("trim"); s1.add_argument("--input",required=True); s1.add_argument("--start",required=True); s1.add_argument("--end",required=True); s1.add_argument("--out",required=True); s1.set_defaults(func=trim)
    s2=sub.add_parser("concat"); s2.add_argument("--list",required=True); s2.add_argument("--out",required=True); s2.set_defaults(func=concat_cmd)
    s3=sub.add_parser("burn-subs"); s3.add_argument("--input",required=True); s3.add_argument("--srt",required=True); s3.add_argument("--out",required=True); s3.set_defaults(func=burn_subs)
    s4=sub.add_parser("autosub"); s4.add_argument("--input",required=True); s4.add_argument("--lang",default="ja"); s4.add_argument("--model",default="qwen2.5:7b-instruct-q4_K_M"); s4.add_argument("--out",required=True); s4.set_defaults(func=autosub)
    args=ap.parse_args(); args.func(args)

if __name__=="__main__":
    main()
