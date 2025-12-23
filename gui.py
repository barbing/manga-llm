import os, threading, queue
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import subprocess, sys, glob
import yaml, csv, time

APP_TITLE = "Manga Local Translator GUI"

def run_cmd(args, cwd=None):
    return subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

class App:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.csv_path = tk.StringVar()
        self.mode = tk.StringVar(value="full")
        self.log = tk.StringVar()
        self.preview_img = None

        frm = tk.Frame(root, padx=8, pady=8)
        frm.pack(fill="both", expand=True)

        def row(btn_text, var, is_dir=True):
            rowf = tk.Frame(frm)
            rowf.pack(fill="x", pady=3)
            tk.Entry(rowf, textvariable=var).pack(side="left", fill="x", expand=True)
            def choose():
                p = filedialog.askdirectory() if is_dir else filedialog.askopenfilename()
                if p: var.set(p)
            tk.Button(rowf, text=btn_text, command=choose).pack(side="left", padx=6)

        tk.Label(frm, text="Input folder (images)").pack(anchor="w")
        row("Browse", self.input_dir, True)
        tk.Label(frm, text="Output folder").pack(anchor="w")
        row("Browse", self.output_dir, True)
        tk.Label(frm, text="CSV (for typeset mode)").pack(anchor="w")
        row("Pick CSV", self.csv_path, False)

        modes = tk.Frame(frm); modes.pack(anchor="w", pady=4)
        tk.Radiobutton(modes, text="Full (OCR+MT+Typeset)", variable=self.mode, value="full").pack(side="left")
        tk.Radiobutton(modes, text="Typeset only (from CSV)", variable=self.mode, value="typeset").pack(side="left")

        btns = tk.Frame(frm); btns.pack(fill="x", pady=6)
        tk.Button(btns, text="Run", command=self.run).pack(side="left")
        tk.Button(btns, text="Preview boxes", command=self.preview).pack(side="left", padx=6)

        self.canvas = tk.Label(frm)
        self.canvas.pack(fill="both", expand=True, pady=8)

        self.console = tk.Text(frm, height=10)
        self.console.pack(fill="both", expand=True)

    def run(self):
        args = [sys.executable, "translate_manga.py", "--output", self.output_dir.get() or "./output_pages"]
        if self.mode.get()=="full":
            args += ["--input", self.input_dir.get() or "./input_pages"]
        else:
            if not self.csv_path.get():
                messagebox.showerror("Error", "Pick a CSV for typeset mode")
                return
            args += ["--mode", "typeset", "--csv", self.csv_path.get(), "--input", self.input_dir.get() or "./input_pages"]
        self.console.delete("1.0", "end")
        proc = run_cmd(args, cwd=os.path.dirname(__file__))
        threading.Thread(target=self._pipe, args=(proc,), daemon=True).start()

    def _pipe(self, proc):
        for line in proc.stdout:
            self.console.insert("end", line)
            self.console.see("end")
        self.console.insert("end", "\n[Done]\n")

    def preview(self):
        out_dir = self.output_dir.get() or "./output_pages"
        dbg = sorted(glob.glob(os.path.join(out_dir, "_debug_overlays", "*_boxes.png")))
        if not dbg:
            messagebox.showinfo("Info", "No debug overlays yet. Run once first.")
            return
        img = Image.open(dbg[-1])
        img.thumbnail((1000, 1000))
        self.preview_img = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.preview_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
