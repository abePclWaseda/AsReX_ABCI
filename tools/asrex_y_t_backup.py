#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2‑speaker separation → ReazonSpeech‑NeMo ASR → WhisperX alignment  (HPC 多 GPU 対応)
──────────────────────────────────────────────────────────
* 入力:   IN_ROOT 以下にある任意階層の .wav（16 kHz, mono/stereo）
* 出力:   同じ相対パスで
          ├─ separated/   ConvTasNet で分離したステレオ wav
          ├─ transcripts/ 話者別 ReazonSpeech‑NeMo ASR (json)
          └─ text/        WhisperX 単語アライン (json)

**2025‑07‑12 安定版**
  - ConvTasNet を *float32*（デフォルト）のまま使い、dtype ミスマッチを根本回避。
  - ファイル名衝突を避けつつ相対パスを保存。
  - `spawn` が必要な環境でも動くよう `multiprocessing` を使用。
"""
from __future__ import annotations
import json, queue, threading, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
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

# ---------- スレッドローカルにモデル保持 ----------
_tls = threading.local()


def init_models(device: str):
    """各スレッド(GPU)につき 1 回だけモデルをロード"""
    if getattr(_tls, "ready", False):
        return
    torch.cuda.set_device(device)

    from asteroid.models import ConvTasNet
    from reazonspeech.nemo.asr import load_model
    import whisperx

    _tls.sep_model = (
        ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
        .to(device)
        .eval()
    )  # fp32
    _tls.asr_model = load_model(device=device)
    _tls.align_model, _tls.meta = whisperx.load_align_model("ja", device)

    _tls.device, _tls.ready = device, True


# ------------------ 1 ファイル処理 ------------------
@torch.inference_mode()
def handle_one(wav: Path) -> bool:
    rel = wav.relative_to(IN_ROOT)  # speakerX/.../abc.wav
    device = _tls.device
    sep_model = _tls.sep_model
    asr_model = _tls.asr_model
    align_model = _tls.align_model
    meta = _tls.meta
    log_file = f"align_errors_{device[-1]}_youtube_train.log"

    try:
        # ---------- ① Separation ----------
        y_mono, _ = librosa.load(wav, sr=16_000, mono=True)
        wav_tensor = torch.tensor(y_mono, device=device).unsqueeze(0)  # fp32
        stereo = sep_model.separate(wav_tensor)[0].cpu().numpy().T  # [T, 2]

        sep_path = SEP_DIR / rel.with_suffix(".wav")
        sep_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(sep_path, stereo, 16_000)

        # ---------- ② ASR ----------
        from reazonspeech.nemo.asr import transcribe, audio_from_numpy

        def txt_path(spk: str):
            p = TXT_DIR / rel.parent / f"{rel.stem}_{spk}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

        for ch, spk in enumerate(SPEAKER_LIST):
            tpath = txt_path(spk)
            if tpath.exists():
                continue  # skip if already transcribed
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
                tpath.open("w"),
                ensure_ascii=False,
                indent=2,
            )

        # ---------- ③ WhisperX align ----------
        aln_path = ALN_DIR / rel.with_suffix(".json")
        aln_path.parent.mkdir(parents=True, exist_ok=True)
        import whisperx

        y, _ = sf.read(sep_path, dtype="float32")

        merged = []
        for ch, spk in enumerate(SPEAKER_LIST):
            segs = json.load(txt_path(spk).open())["segments"]
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
        merged.sort(key=lambda x: x["start"])
        json.dump(merged, aln_path.open("w"), ensure_ascii=False, indent=2)

    except Exception:
        with open(log_file, "a") as f:
            traceback.print_exc(file=f)
        return False
    return True


# ---------------- メイン ----------------


def is_done(wav: Path) -> bool:
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
    wav_all = [w for w in IN_ROOT.rglob("*.wav") if not is_done(w)]
    print(f"{len(wav_all)} files remain.")

    q: "queue.Queue[Path]" = queue.Queue()
    for w in wav_all:
        q.put(w)

    with ThreadPoolExecutor(max_workers=len(GPU_LIST)) as ex:
        futs = [ex.submit(gpu_worker, dev, q) for dev in GPU_LIST]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="GPUs idle"):
            pass

    print("=== all done! ===")
