# feedBack AI Audio-to-Tab Builder

Turn raw **`.mp3`, `.flac`, `.wav`, or `.ogg`** audio files (with no existing tab or chart) into multi-stem, beat-synced **`.feedpak`** packages for **fee[dB]ack**.

## How It Works
1. **6-Stem AI Separation (`Demucs htdemucs_6s`):** Splits audio into isolated **Bass, Drums, Guitar, Other (Other + Piano), and Vocals** stems on NVIDIA CUDA GPUs (with automatic Apple Silicon MPS and CPU fallback).
2. **Neural Pitch & Onset Transcription (`Basic Pitch` + `librosa`):** Transcribes timestamped notes directly from the isolated bass and guitar stems so note timing locks onto the audio waveform automatically.
3. **Ergonomic Fretboard Solver:** Uses dynamic programming to map MIDI pitches across 4-string (`E Std` / `Drop D`), 5-string bass, or 6-string guitar while minimizing hand shifts.
4. **Full `.feedpak` Packaging:** Attaches embedded metadata, cover art, configurable lead-in padding, and all 5 OGG stems for full in-game mixer control.

## Setup
1. Install **FFmpeg** (`winget install Gyan.FFmpeg` on Windows).
2. Create a **Python 3.12** virtual environment and install dependencies:
   ```bash
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
   pip install demucs librosa PyYAML soundfile pretty_midi onnxruntime resampy mir-eval
   pip install basic-pitch --no-deps
   python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); x = torch.zeros(1, device='cuda'); print('GPU OK')"
   ```
3. Launch the GUI:
   ```bash
   python ai_tab_gui.py
   ```