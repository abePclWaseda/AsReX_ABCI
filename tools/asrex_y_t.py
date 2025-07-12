#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2-speaker separation ➜ ReazonSpeech-NeMo ASR ➜ WhisperX alignment  (thread-pool version)
"""
import os, json, traceback, queue, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm

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

# ---------- ⚠️ スレッドローカルにモデルを保持 ----------
_tls = threading.local()


def init_models(device: str):
    """スレッドローカルに各モデルを 1 度だけロード"""
    if getattr(_tls, "ready", False):  # 既に初期化済み
        return
    torch.cuda.set_device(device)

    from asteroid.models import ConvTasNet

    _tls.sep_model = (
        ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
        .to(device)
        .eval()
        .half()
    )

    from reazonspeech.nemo.asr import load_model

    _tls.asr_model = load_model(device=device)  # ReazonSpeech-NeMo

    import whisperx

    _tls.align_model, _tls.meta = whisperx.load_align_model("ja", device)

    _tls.device, _tls.ready = device, True


# ------------------ 1 ファイル処理 ------------------
@torch.inference_mode()
def handle_one(wav: Path):
    rel = wav.relative_to(IN_ROOT)
    stem = rel.with_suffix("")
    device = _tls.device
    sep_model = _tls.sep_model
    asr_model = _tls.asr_model
    align_model, meta = _tls.align_model, _tls.meta
    log_file = f"align_errors_{device[-1]}_youtube_train.log"

    try:
        # --- ① separation ---
        y_mono, _ = librosa.load(wav, sr=16_000)
        est = sep_model.separate(torch.tensor(y_mono).unsqueeze(0))
        stereo = est[0].cpu().numpy().T
        sep_path = SEP_DIR / rel.with_suffix(".wav")
        sep_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(sep_path, stereo, 16_000)

        # --- ② ASR (speaker wise) ---
        from reazonspeech.nemo.asr import transcribe, audio_from_numpy

        def txt_path(spk: str) -> Path:
            p = TXT_DIR / (stem.as_posix() + f"_{spk}.json")
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

        for ch, spk in enumerate(SPEAKER_LIST):
            if txt_path.exists():  # 既に終わっていればスキップ
                continue
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
                txt_path(spk).open("w"),
                ensure_ascii=False,
                indent=2,
            )

        # --- ③ WhisperX align ---
        aln_path = ALN_DIR / rel.with_suffix(".json")
        aln_path.parent.mkdir(parents=True, exist_ok=True)
        y, sr = sf.read(sep_path, dtype="float32")
        merged = []
        import whisperx

        for ch, spk in enumerate(SPEAKER_LIST):
            segs = json.load((TXT_DIR / f"{wav.stem}_{spk}.json").open())["segments"]
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
        out_path = ALN_DIR / f"{wav.stem}.json"
        json.dump(merged, aln_path.open("w"), ensure_ascii=False, indent=2)

    except Exception as e:
        with open(log_file, "a") as f:
            traceback.print_exc(file=f)
        # 失敗したファイルはエラーでスキップ
        return False
    return True


# ---------------- メイン：スレッドプール ----------------
def is_done(wav: Path) -> bool:
    aln_path = ALN_DIR / wav.relative_to(IN_ROOT).with_suffix(".json")
    return aln_path.is_file() and aln_path.stat().st_size > 0


if __name__ == "__main__":
    wav_all = [w for w in sorted(IN_ROOT.rglob("*.wav")) if not is_done(w)]
    print(f"{len(wav_all)} files remain.")

    # ★ wav を queue に積む
    q = queue.Queue()
    for w in wav_all:
        q.put(w)

    def gpu_worker(device: str):
        init_models(device)
        while True:
            try:
                wav = q.get_nowait()
            except queue.Empty:
                break
            handle_one(wav)
            q.task_done()

    # ★ ThreadPoolExecutor で GPU 数ぶんスレッド
    with ThreadPoolExecutor(max_workers=len(GPU_LIST)) as ex:
        futs = [ex.submit(gpu_worker, dev) for dev in GPU_LIST]
        # プログレスバー
        for _ in tqdm(as_completed(futs), total=len(futs), desc="GPUs idle"):
            pass

    print("=== all done! ===")
