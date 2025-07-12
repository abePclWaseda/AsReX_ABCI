#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2‑speaker separation → ReazonSpeech‑NeMo ASR → WhisperX alignment  (thread‑pool version)

* 入力:  IN_ROOT 以下の任意階層にある .wav（16 kHz, mono or stereo）
* 出力:  それぞれのファイルと同じ相対パスで
         - separated/  : 分離後ステレオ wav
         - transcripts/ : 話者別 ReazonSpeech‐NeMo 生 ASR (json)
         - text/        : WhisperX アライン後の単語レベル json

ディレクトリ階層を壊さずに保存するため、`wav.relative_to(IN_ROOT)` を
すべての出力パス組み立てに利用する。
"""
import json
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# -------------------- Paths --------------------
IN_ROOT = Path(
    "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/audio/youtube_train"
)
SEP_DIR = Path(
    "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/separated/youtube_train"
)
TXT_DIR = Path(
    "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/transcripts/youtube_train"
)
ALN_DIR = Path(
    "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/text/youtube_train"
)
for p in (SEP_DIR, TXT_DIR, ALN_DIR):
    p.mkdir(parents=True, exist_ok=True)

SPEAKER_LIST = ("A", "B")
GPU_LIST = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

# ---------- スレッドローカルにモデルを保持 ----------
_tls = threading.local()


def init_models(device: str):
    """各スレッド(GPU)につき 1 回だけモデルロード"""
    if getattr(_tls, "ready", False):
        return

    torch.cuda.set_device(device)

    from asteroid.models import ConvTasNet  # 遅延 import で起動を速く
    from reazonspeech.nemo.asr import load_model
    import whisperx

    _tls.sep_model = (
        ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
        .to(device)
        .eval()
        .half()
    )
    _tls.asr_model = load_model(device=device)  # ReazonSpeech‑NeMo
    _tls.align_model, _tls.meta = whisperx.load_align_model("ja", device)

    _tls.device, _tls.ready = device, True


# ------------------ 1 ファイル処理 ------------------
@torch.inference_mode()
def handle_one(wav: Path) -> bool:
    """1 ファイルを分離→ASR→アラインし、階層を保ったまま保存"""

    # 相対パスを基準に出力パスを決定
    rel = wav.relative_to(IN_ROOT)  # e.g.  speaker1/session/abc.wav
    stem = rel.with_suffix("")  # e.g.  speaker1/session/abc
    device = _tls.device
    sep_model = _tls.sep_model
    asr_model = _tls.asr_model
    align_model = _tls.align_model
    meta = _tls.meta
    log_file = f"align_errors_{device[-1]}_youtube_train.log"

    try:
        # ---------- ① separation ----------
        y_mono, _ = librosa.load(wav, sr=16_000, mono=True)
        est = sep_model.separate(torch.tensor(y_mono).unsqueeze(0))  # [1, 2, T]
        stereo = est[0].cpu().numpy().T  # [T, 2]

        sep_path = SEP_DIR / rel.with_suffix(".wav")
        sep_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(sep_path, stereo, 16_000)

        # ---------- ② ASR (speaker‑wise) ----------
        from reazonspeech.nemo.asr import transcribe, audio_from_numpy

        def txt_path(spk: str) -> Path:
            p = TXT_DIR / rel.parent / f"{rel.stem}_{spk}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

        for ch, spk in enumerate(SPEAKER_LIST):
            path = txt_path(spk)
            if path.exists():
                continue  # 既に transcribe 済み
            ret = transcribe(asr_model, audio_from_numpy(stereo[:, ch], 16_000))
            segs = [
                {
                    "start": round(s.start_seconds, 3),
                    "end": round(s.end_seconds, 3),
                    "text": s.text,
                }
                for s in ret.segments
            ]
            json.dump(
                {"text": ret.text, "segments": segs},
                path.open("w"),
                ensure_ascii=False,
                indent=2,
            )

        # ---------- ③ WhisperX align ----------
        aln_path = ALN_DIR / rel.with_suffix(".json")
        aln_path.parent.mkdir(parents=True, exist_ok=True)

        y, _ = sf.read(sep_path, dtype="float32")  # y.shape == [T, 2]
        import whisperx

        merged = []
        for ch, spk in enumerate(SPEAKER_LIST):
            segs_path = txt_path(spk)
            segs = json.load(segs_path.open())["segments"]
            aligned = whisperx.align(
                segs, align_model, meta, y[:, ch], device, return_char_alignments=False
            )
            merged.extend(
                {
                    "speaker": spk,
                    "word": w["word"],
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                }
                for seg in aligned["segments"]
                for w in seg["words"]
            )

        merged.sort(key=lambda x: x["start"])  # 時間順に整列
        json.dump(merged, aln_path.open("w"), ensure_ascii=False, indent=2)

    except Exception:
        with open(log_file, "a") as f:
            traceback.print_exc(file=f)
        return False  # エラーでスキップ

    return True


# ---------------- メイン：スレッドプール ----------------
def is_done(wav: Path) -> bool:
    """アライン済みかを判定"""
    aln_path = ALN_DIR / wav.relative_to(IN_ROOT).with_suffix(".json")
    return aln_path.is_file() and aln_path.stat().st_size > 0


def gpu_worker(device: str, q: "queue.Queue[Path]"):
    init_models(device)
    while True:
        try:
            wav = q.get_nowait()
        except queue.Empty:
            break
        handle_one(wav)
        q.task_done()


if __name__ == "__main__":
    wav_all = [w for w in sorted(IN_ROOT.rglob("*.wav")) if not is_done(w)]
    print(f"{len(wav_all)} files remain.")

    q: "queue.Queue[Path]" = queue.Queue()
    for w in wav_all:
        q.put(w)

    with ThreadPoolExecutor(max_workers=len(GPU_LIST)) as ex:
        futs = [ex.submit(gpu_worker, dev, q) for dev in GPU_LIST]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="GPUs idle"):
            pass

    print("=== all done! ===")
