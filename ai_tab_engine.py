##part 1
"""AI Audio-to-Tab (.feedpak) Generator using Demucs + Basic Pitch."""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

BASE_DIR = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
FEEDBACK_LIB = BASE_DIR / "feedback_lib"
sys.path.insert(0, str(FEEDBACK_LIB))

import librosa
import numpy as np
import torch
import yaml
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from song import Anchor, Arrangement, Note, arrangement_to_wire

TUNINGS = {
    "bass_4_std": {"label": "Bass", "type": "bass", "open_midi": (28, 33, 38, 43), "offsets": (0, 0, 0, 0)},
    "bass_4_dropd": {"label": "Bass", "type": "bass", "open_midi": (26, 33, 38, 43), "offsets": (-2, 0, 0, 0)},
    "bass_5_std": {"label": "Bass", "type": "bass", "open_midi": (23, 28, 33, 38, 43), "offsets": (0, 0, 0, 0, 0)},
    "guitar_6_std": {"label": "Lead", "type": "guitar", "open_midi": (40, 45, 50, 55, 59, 64), "offsets": (0, 0, 0, 0, 0, 0)},
}


def detect_compute_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def probe_audio_info(audio_path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:format_tags", "-of", "json", str(audio_path)
    ]
    raw = subprocess.check_output(cmd).decode("utf-8", errors="replace")
    data = json.loads(raw).get("format", {})
    duration = round(float(data.get("duration", 0.0)), 3)
    tags = {k.lower(): v for k, v in (data.get("tags") or {}).items()}
    return {
        "duration": duration,
        "title": tags.get("title", "").strip(),
        "artist": tags.get("artist", "").strip(),
        "album": tags.get("album", "").strip(),
        "year": str(tags.get("date", tags.get("year", ""))).strip()[:4],
    }


def extract_cover_art(audio_path: Path, custom_cover: Path | None, dest_jpg: Path, artist: str = "", album: str = "") -> bool:
    candidates: list[Path] = []
    if custom_cover:
        candidates.append(custom_cover)
    parent = audio_path.parent
    stems = [audio_path.stem]
    if artist and album:
        stems.append(f"{artist} - {album}")
    if " - " in audio_path.stem:
        stems.append(audio_path.stem.rsplit(" - ", 1)[0])
    for s in stems:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidates.append(parent / f"{s}{ext}")
    for generic in ("cover.jpg", "cover.png", "folder.jpg", "folder.png"):
        candidates.append(parent / generic)

    for img in candidates:
        if img.is_file():
            if img.suffix.lower() in (".jpg", ".jpeg"):
                shutil.copy2(img, dest_jpg)
            else:
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "quiet", "-i", str(img), str(dest_jpg)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                )
            if dest_jpg.is_file() and dest_jpg.stat().st_size > 0:
                return True

    subprocess.run(
        ["ffmpeg", "-y", "-v", "quiet", "-i", str(audio_path), "-an", "-vcodec", "mjpeg", "-frames:v", "1", str(dest_jpg)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    return dest_jpg.is_file() and dest_jpg.stat().st_size > 0


def separate_stems(audio_path: Path, out_dir: Path, device: str, log_fn=print) -> dict[str, Path]:
    """Run Demucs htdemucs_6s (Bass, Drums, Guitar, Piano, Other, Vocals)."""
    log_fn(f"[Demucs] Separating 6 stems (htdemucs_6s) on {device.upper()}...")
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs_6s", "-d", device,
        "-o", str(out_dir), str(audio_path)
    ]
    subprocess.run(cmd, check=True)
    stem_folder = out_dir / "htdemucs_6s" / audio_path.stem
    return {
        "bass": stem_folder / "bass.wav",
        "drums": stem_folder / "drums.wav",
        "guitar": stem_folder / "guitar.wav",
        "piano": stem_folder / "piano.wav",
        "other": stem_folder / "other.wav",
        "vocals": stem_folder / "vocals.wav",
    }

##part 2
def transcribe_stem_to_notes(
    stem_wav: Path,
    tuning_key: str,
    onset_thresh: float = 0.55,
    frame_thresh: float = 0.35,
    min_note_len_ms: float = 65.0,
    monophonic: bool = True,
    lead_in: float = 0.0,
) -> tuple[list[Note], list[Anchor]]:
    """Run Basic Pitch on isolated stem and solve ergonomic string/fret positions."""
    cfg = TUNINGS[tuning_key]
    open_midi = cfg["open_midi"]
    min_pitch = min(open_midi)
    max_pitch = max(open_midi) + 22
    min_hz = float(librosa.midi_to_hz(min_pitch))
    max_hz = float(librosa.midi_to_hz(max_pitch))

    _, _, note_events = predict(
        str(stem_wav),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_thresh,
        frame_threshold=frame_thresh,
        minimum_note_length=min_note_len_ms,
        minimum_frequency=min_hz,
        maximum_frequency=max_hz,
    )

    raw_events = []
    for start_t, end_t, pitch, amp, _ in sorted(note_events, key=lambda x: x[0]):
        p = int(round(pitch))
        while p < min_pitch:
            p += 12
        while p > max_pitch:
            p -= 12
        raw_events.append((float(start_t) + lead_in, float(end_t) + lead_in, p, float(amp)))

    if monophonic and raw_events:
        cleaned = []
        for ev in raw_events:
            if not cleaned:
                cleaned.append(list(ev))
                continue
            p_start, p_end, p_pitch, p_amp = cleaned[-1]
            c_start, c_end, c_pitch, c_amp = ev
            if c_start - p_start < 0.045:
                if c_amp > p_amp:
                    cleaned[-1] = list(ev)
            else:
                if p_end > c_start:
                    cleaned[-1][1] = max(p_start + 0.03, c_start - 0.005)
                cleaned.append(list(ev))
        raw_events = [tuple(x) for x in cleaned]

    

    if not raw_events:
        return [], []

    # Dynamic-programming fretboard pathfinder (minimizes hand position leaps)
    candidates_per_note = []
    for _, _, pitch, _ in raw_events:
        opts = []
        for s_idx, open_p in enumerate(open_midi):
            fret = pitch - open_p
            if 0 <= fret <= 22:
                opts.append((s_idx, fret))
        if not opts:
            opts.append((0, max(0, min(22, pitch - open_midi[0]))))
        candidates_per_note.append(opts)

        dp_cost = [{opt: (0.0 if opt[1] == 0 else opt[1] * 0.25) for opt in candidates_per_note[0]}]
        dp_back = [{}]

        for i in range(1, len(candidates_per_note)):
            cur_cost = {}
            cur_back = {}
            for opt in candidates_per_note[i]:
                s_cur, f_cur = opt
                best_c = float("inf")
                best_prev = None
                for prev_opt, prev_c in dp_cost[i - 1].items():
                    s_prev, f_prev = prev_opt
                    if f_cur == 0 or f_prev == 0:
                        move_pen = 0.3
                    else:
                        move_pen = (abs(f_cur - f_prev) * 1.1) + (abs(s_cur - s_prev) * 0.4)
                    high_pen = max(0, f_cur - 12) * 0.35
                    total = prev_c + move_pen + high_pen
                    if total < best_c:
                        best_c = total
                        best_prev = prev_opt
                cur_cost[opt] = best_c
                cur_back[opt] = best_prev
            dp_cost.append(cur_cost)
            dp_back.append(cur_back)


    last_opt = min(dp_cost[-1], key=dp_cost[-1].get)
    chosen = [last_opt]
    for i in range(len(candidates_per_note) - 1, 0, -1):
        last_opt = dp_back[i][last_opt]
        chosen.append(last_opt)
    chosen.reverse()

    notes: list[Note] = []
    anchors: list[Anchor] = []
    cur_anchor_fret = -1

    for (start_t, end_t, _, amp), (s_idx, fret) in zip(raw_events, chosen):
        dur = max(0.0, end_t - start_t)
        sustain = dur if dur >= 0.18 else 0.0
        notes.append(Note(
            time=round(start_t, 3),
            string=s_idx,
            fret=fret,
            sustain=round(sustain, 3),
            accent=(amp >= 0.85),
        ))
        target_anchor = 1 if fret <= 4 else max(1, fret - 1)
        if cur_anchor_fret == -1 or abs(fret - cur_anchor_fret) >= 4:
            cur_anchor_fret = target_anchor
            anchors.append(Anchor(time=round(start_t, 3), fret=cur_anchor_fret, width=4))

    return notes, anchors
##part 3
def detect_beat_timeline(audio_path: Path, lead_in: float = 0.0) -> dict:
    """Estimate beats and 4/4 measure downbeats with librosa."""
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beats = []
    for idx, bt in enumerate(beat_times):
        beats.append({
            "time": round(float(bt) + lead_in, 6),
            "measure": (idx // 4) + 1 if idx % 4 == 0 else -1,
        })
    sections = [{"name": "intro", "number": 1, "time": round(lead_in, 3)}] if beats else []
    return {"beats": beats, "sections": sections}


def convert_audio_to_ogg(audio_path: Path, ogg_dest: Path, lead_in: float = 0.0) -> None:
    ogg_dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(audio_path)]
    if lead_in > 0.001:
        ms = int(round(lead_in * 1000))
        cmd += ["-af", f"adelay={ms}|{ms}"]
    cmd += ["-c:a", "libvorbis", "-q:a", "7", str(ogg_dest)]
    subprocess.check_call(cmd)

def mix_other_and_piano_to_ogg(other_wav: Path, piano_wav: Path, ogg_dest: Path, lead_in: float = 0.0) -> None:
    """Blend htdemucs_6s 'other' + 'piano' stems into a single other.ogg so no audio is lost."""
    ogg_dest.parent.mkdir(parents=True, exist_ok=True)
    if other_wav.is_file() and piano_wav.is_file():
        ms = int(round(lead_in * 1000))
        delay_filter = f",adelay={ms}|{ms}" if lead_in > 0.001 else ""
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(other_wav), "-i", str(piano_wav),
            "-filter_complex", f"amix=inputs=2:duration=longest:normalize=0{delay_filter}",
            "-c:a", "libvorbis", "-q:a", "7", str(ogg_dest)
        ]
        subprocess.run(cmd, check=True)
    elif other_wav.is_file():
        convert_audio_to_ogg(other_wav, ogg_dest, lead_in=lead_in)


def generate_ai_feedpak(
    audio_path: Path,
    output_dir: Path,
    bass_tuning: str = "bass_4_std",
    include_guitar: bool = False,
    sensitivity: float = 0.55,
    lead_in: float = 3.0,
    custom_cover: Path | None = None,
    log_fn=print,
) -> Path:
    device = detect_compute_device()
    log_fn(f"[Hardware] Active compute device: {device.upper()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stems = separate_stems(audio_path, tmp_path / "stems_out", device=device, log_fn=log_fn)

        pack_staging = tmp_path / "pack"
        (pack_staging / "arrangements").mkdir(parents=True, exist_ok=True)
        (pack_staging / "stems").mkdir(parents=True, exist_ok=True)

        log_fn("[Audio] Encoding 5 stems (Bass, Drums, Guitar, Other, Vocals)...")
        manifest_stems = []
        for stem_id in ("bass", "drums", "guitar", "other", "vocals"):
            rel_ogg = f"stems/{stem_id}.ogg"
            dest_ogg = pack_staging / rel_ogg
            if stem_id == "other":
                mix_other_and_piano_to_ogg(stems["other"], stems["piano"], dest_ogg, lead_in=lead_in)
            elif stems.get(stem_id) and stems[stem_id].is_file():
                convert_audio_to_ogg(stems[stem_id], dest_ogg, lead_in=lead_in)

            if dest_ogg.is_file():
                manifest_stems.append({
                    "id": stem_id,
                    "name": stem_id.capitalize(),
                    "file": rel_ogg,
                    "default": (stem_id != "bass"),
                })

        timeline = detect_beat_timeline(audio_path, lead_in=lead_in)
        (pack_staging / "song_timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")

        meta = probe_audio_info(audio_path)
        duration = round(meta["duration"] + lead_in, 3)
        if " - " in audio_path.stem:
            fn_artist, fn_title = [p.strip() for p in audio_path.stem.split(" - ", 1)]
        else:
            fn_artist, fn_title = "Unknown", audio_path.stem.strip()

        song_artist = meta["artist"] or fn_artist
        song_title = meta["title"] or fn_title
        song_album = meta["album"]
        song_year = int(meta["year"]) if meta["year"].isdigit() else 0

        has_cover = extract_cover_art(audio_path, custom_cover, pack_staging / "cover.jpg", song_artist, song_album)
        if has_cover:
            log_fn("[Art] Attached cover.jpg")

        manifest_arrangements = []

        # 1. Transcribe Bass Stem (tuned for rapid 8th-note punk/rock chugs)
        log_fn(f"[AI Tab] Transcribing Bass ({bass_tuning})...")
        b_notes, b_anchors = transcribe_stem_to_notes(
            stems["bass"], bass_tuning,
            onset_thresh=sensitivity, frame_thresh=0.28, min_note_len_ms=40.0,
            monophonic=True, lead_in=lead_in
        )
        b_cfg = TUNINGS[bass_tuning]
        b_arr = Arrangement(
            name=b_cfg["label"],
            tuning=list(b_cfg["offsets"]),
            notes=b_notes,
            anchors=b_anchors,
            type="bass",
            path_bass=True,
        )
        (pack_staging / "arrangements" / "bass_0.json").write_text(
            json.dumps(arrangement_to_wire(b_arr), indent=2), encoding="utf-8"
        )
        manifest_arrangements.append({
            "id": "bass_0", "name": "Bass", "type": "bass",
            "file": "arrangements/bass_0.json", "tuning": list(b_cfg["offsets"]), "capo": 0
        })
        log_fn(f"[AI Tab] Generated {len(b_notes)} bass notes.")

        # 2. Optional Guitar Stem (now using isolated guitar.wav from htdemucs_6s!)
        g_stem = stems["guitar"] if stems["guitar"].is_file() else stems["other"]
        if include_guitar and g_stem.is_file():
            log_fn("[AI Tab] Transcribing Guitar (isolated guitar stem)...")
            g_notes, g_anchors = transcribe_stem_to_notes(
                g_stem, "guitar_6_std",
                onset_thresh=min(0.85, sensitivity + 0.08), frame_thresh=0.30, min_note_len_ms=45.0,
                monophonic=False, lead_in=lead_in
            )
            g_offsets = list(TUNINGS["guitar_6_std"]["offsets"])
            g_arr = Arrangement(
                name="Lead",
                tuning=g_offsets,
                notes=g_notes,
                anchors=g_anchors,
                type="guitar",
                path_lead=True,
            )
            (pack_staging / "arrangements" / "lead_1.json").write_text(
                json.dumps(arrangement_to_wire(g_arr), indent=2), encoding="utf-8"
            )
            manifest_arrangements.append({
                "id": "lead_1", "name": "Lead", "type": "guitar",
                "file": "arrangements/lead_1.json", "tuning": g_offsets, "capo": 0
            })
            log_fn(f"[AI Tab] Generated {len(g_notes)} guitar notes.")

        manifest = {
            "feedpak_version": "1.2.0",
            "title": song_title,
            "artist": song_artist,
            "album": song_album,
            "year": song_year,
            "duration": duration,
            "song_timeline": "song_timeline.json",
            "stems": manifest_stems,
            "arrangements": manifest_arrangements,
        }
        if has_cover:
            manifest["cover"] = "cover.jpg"

        (pack_staging / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

        safe_base = f"{song_artist} - {song_title} (AI Tab)".replace("/", "_").replace(":", "")
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{safe_base}.feedpak"

        with zipfile.ZipFile(final_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in pack_staging.rglob("*"):
                if item.is_file():
                    zf.write(item, item.relative_to(pack_staging))

        log_fn(f"[Done] Built 5-Stem AI Tab Pack: {final_path}")
        return final_path




