"""Dark-Mode GUI for AI Audio-to-Tab (.feedpak) Generator."""

import json
import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ai_tab_engine import detect_compute_device, generate_ai_feedpak

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "ai_gui_settings.json"


class AITabGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("fee[dB]ack AI Audio-to-Tab Builder (Demucs + Basic Pitch)")
        self.geometry("760x620")
        self.configure(bg="#0f172a")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#0f172a")
        style.configure("TLabel", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), foreground="#38bdf8")
        style.configure("TCheckbutton", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 10))

        saved = self._load_settings()
        self.audio_path = tk.StringVar(value=saved.get("audio_path", ""))
        self.cover_path = tk.StringVar(value=saved.get("cover_path", ""))
        self.out_dir = tk.StringVar(value=saved.get("out_dir", str(APP_DIR / "output")))
        self.tuning = tk.StringVar(value=saved.get("tuning", "bass_4_std"))
        self.include_guitar = tk.BooleanVar(value=saved.get("include_guitar", False))
        self.lead_in = tk.StringVar(value=saved.get("lead_in", "3.0"))
        self.sensitivity = tk.DoubleVar(value=float(saved.get("sensitivity", 0.55)))

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_settings(self) -> dict:
        if SETTINGS_FILE.is_file():
            try:
                return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_settings(self):
        data = {
            "audio_path": self.audio_path.get(),
            "cover_path": self.cover_path.get(),
            "out_dir": self.out_dir.get(),
            "tuning": self.tuning.get(),
            "include_guitar": self.include_guitar.get(),
            "lead_in": self.lead_in.get(),
            "sensitivity": round(self.sensitivity.get(), 2),
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self):
        self._save_settings()
        self.destroy()

    def _row_picker(self, parent, label_text, var, cmd):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label_text, width=18).pack(side="left")
        tk.Entry(
            row, textvariable=var, bg="#1e293b", fg="#f8fafc",
            insertbackground="#f8fafc", relief="flat", font=("Segoe UI", 10)
        ).pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 8))
        tk.Button(
            row, text="Browse...", command=cmd, bg="#334155", fg="#f8fafc",
            relief="flat", padx=12, pady=4, cursor="hand2"
        ).pack(side="right")

    def _build_ui(self):
        top = ttk.Frame(self, padding=16)
        top.pack(fill="x")

        dev = detect_compute_device().upper()
        badge_color = "#4ade80" if dev == "CUDA" else "#facc15"
        hdr = ttk.Frame(top)
        hdr.pack(fill="x", pady=(0, 8))
        ttk.Label(hdr, text="AI Audio-to-Tab Builder (.mp3/.flac -> .feedpak)", style="Header.TLabel").pack(side="left")
        tk.Label(hdr, text=f"Engine: {dev}", bg="#1e293b", fg=badge_color, font=("Segoe UI", 9, "bold"), padx=10, pady=3).pack(side="right")

        self._row_picker(top, "Audio File:", self.audio_path, self._pick_audio)
        self._row_picker(top, "Cover Art (opt):", self.cover_path, self._pick_cover)
        self._row_picker(top, "Output Folder:", self.out_dir, self._pick_out)

        opts = ttk.Frame(top)
        opts.pack(fill="x", pady=(10, 4))

        ttk.Label(opts, text="Bass Tuning:").pack(side="left", padx=(0, 6))
        t_box = ttk.Combobox(
            opts, textvariable=self.tuning, state="readonly", width=15,
            values=["bass_4_std", "bass_4_dropd", "bass_5_std"]
        )
        t_box.pack(side="left", padx=(0, 16))

        ttk.Label(opts, text="Lead-In (s):").pack(side="left", padx=(0, 4))
        tk.Entry(opts, textvariable=self.lead_in, width=6, bg="#1e293b", fg="#f8fafc", justify="center", relief="flat").pack(side="left", ipady=4, padx=(0, 16))

        ttk.Checkbutton(opts, text="Also Transcribe Guitar (Draft)", variable=self.include_guitar).pack(side="left")

        sens_row = ttk.Frame(top)
        sens_row.pack(fill="x", pady=(6, 0))
        ttk.Label(sens_row, text="Note Sensitivity (Lower = More Notes):", width=34).pack(side="left")
        tk.Scale(
            sens_row, from_=0.30, to=0.80, resolution=0.05, orient="horizontal",
            variable=self.sensitivity, bg="#0f172a", fg="#f8fafc", highlightthickness=0, troughcolor="#1e293b"
        ).pack(side="left", fill="x", expand=True)

        tk.Button(
            top, text="Separate Stems & Generate AI Tab", command=self._run_build,
            bg="#2563eb", fg="#ffffff", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=20, pady=9, cursor="hand2"
        ).pack(anchor="e", pady=(12, 0))

        log_frame = ttk.Frame(self, padding=(16, 0, 16, 16))
        log_frame.pack(fill="both", expand=True)
        ttk.Label(log_frame, text="AI Transcription Log:").pack(anchor="w")
        self.log_box = tk.Text(
            log_frame, bg="#020617", fg="#38bdf8", font=("Consolas", 9),
            relief="flat", state="disabled"
        )
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

    def _log(self, msg: str):
        self.after(0, self._append_log, str(msg))

    def _append_log(self, msg: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _initial_dir(self, val: str) -> str:
        p = Path(val) if val else None
        if p and p.is_dir():
            return str(p)
        if p and p.parent.is_dir():
            return str(p.parent)
        return str(Path.home())

    def _pick_audio(self):
        f = filedialog.askopenfilename(
            initialdir=self._initial_dir(self.audio_path.get()),
            filetypes=[("Audio Files", "*.flac *.mp3 *.wav *.ogg *.m4a")],
        )
        if f:
            self.audio_path.set(f)
            self._save_settings()

    def _pick_cover(self):
        f = filedialog.askopenfilename(
            initialdir=self._initial_dir(self.cover_path.get() or self.audio_path.get()),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")],
        )
        if f:
            self.cover_path.set(f)
            self._save_settings()

    def _pick_out(self):
        d = filedialog.askdirectory(initialdir=self._initial_dir(self.out_dir.get()))
        if d:
            self.out_dir.set(d)
            self._save_settings()

    def _run_build(self):
        if not self.audio_path.get():
            messagebox.showwarning("Missing Audio", "Select an .mp3, .flac, or .wav file first.")
            return
        self._save_settings()
        audio = Path(self.audio_path.get())
        out_dir = Path(self.out_dir.get())
        cover = Path(self.cover_path.get()) if self.cover_path.get() else None
        tuning = self.tuning.get()
        guitar = self.include_guitar.get()
        lead_in = float(self.lead_in.get() or 3.0)
        sens = float(self.sensitivity.get())

        def worker():
            try:
                generate_ai_feedpak(
                    audio_path=audio,
                    output_dir=out_dir,
                    bass_tuning=tuning,
                    include_guitar=guitar,
                    sensitivity=sens,
                    lead_in=lead_in,
                    custom_cover=cover,
                    log_fn=self._log,
                )
            except Exception as exc:
                self._log(f"[ERROR] {exc}")

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = AITabGUI()
    app.mainloop()
