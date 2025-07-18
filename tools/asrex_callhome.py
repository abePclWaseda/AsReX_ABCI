#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CallHome 2‑channel speech → ReazonSpeech‑NeMo ASR → WhisperX alignment
=====================================================================
* **ステレオ 2ch = 2 話者** 前提（ch‑0=A, ch‑1=B）なので分離不要。
* 8 kHz μ‑law でも自動で 16 kHz PCM へリサンプル。
* 30 s ウィンドウでストリーミング ASR → WhisperX アラインメント。

Directory layout
----------------
CallHome/
  ├── audio/        # 入力 wav
  ├── transcripts/  # スピーカ別 ASR (JSON)
  └── alignment/    # Word‑level マージ JSON

Usage
-----
```bash
python callhome_pipeline.py --dirs .      # audio/ 直下を全処理
python callhome_pipeline.py --dirs 2024_* # サブフォルダを指定
```
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import torch
import torchaudio
from tqdm import tqdm

# ─── Config ─────────────────────────────────────────────────────────
ROOT = Path("/home/acg17145sv/experiments/0162_dialogue_model/CallHome")
IN_ROOT = ROOT / "audio"
TXT_ROOT = ROOT / "transcripts"
ALN_ROOT = ROOT / "alignment"
for p in (TXT_ROOT, ALN_ROOT):
    p.mkdir(parents=True, exist_ok=True)

SPEAKERS = ("A", "B")
SAMPLE_RATE = 16_000
CHUNK_SEC = 30  # seconds

# ─── Helpers ────────────────────────────────────────────────────────


def resample_if_needed(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Return **2‑ch @16 kHz** tensor on CPU."""
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    # 強制 2ch (同じ波形を複製)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0).repeat(2, 1)
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    return wav.cpu()


def chunk_audio(wav: torch.Tensor, sec: int = CHUNK_SEC):
    """Yield (start_s, end_s, chunk) for **1D or 2D tensor**.

    * Accepts shape `(T,)` **or** `(C, T)`; slicing is always on the last dim.
    * Returns chunks with same dimensionality.
    """
    total = wav.shape[-1]
    if total == 0:
        return  # zero‑length guard
    hop = sec * SAMPLE_RATE
    for start in range(0, total, hop):
        end = min(start + hop, total)
        yield start / SAMPLE_RATE, end / SAMPLE_RATE, wav[..., start:end]


# ─── Worker ─────────────────────────────────────────────────────────


def worker(device: str, wav_paths: List[Path], align_threads: int = 2):
    torch.cuda.set_device(device)
    os.environ["OMP_NUM_THREADS"] = "1"  # BLAS threads

    from reazonspeech.nemo.asr import load_model, transcribe, audio_from_numpy
    import whisperx

    asr_model = load_model(device=device)
    align_model, meta = whisperx.load_align_model("ja", device)
    align_exec = ThreadPoolExecutor(max_workers=align_threads)

    job_tag = os.getenv("PBS_ARRAY_INDEX") or os.getenv("SLURM_ARRAY_TASK_ID") or "solo"
    log = Path(f"align_errors_gpu{device[-1]}_{job_tag}.log").open(
        "a", encoding="utf-8"
    )

    def elog(msg: str):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    for wav in tqdm(wav_paths, desc=f"[GPU {device}]"):
        try:
            wav_tensor, sr = torchaudio.load(wav)
            wav_tensor = resample_if_needed(wav_tensor, sr)

            rel = wav.relative_to(IN_ROOT)
            txt_dir = TXT_ROOT / rel.parent
            txt_dir.mkdir(parents=True, exist_ok=True)

            segs_per_spk: list[list[dict]] = []

            # ── ASR per channel ─────────────────────────────────────
            for ch, spk in enumerate(SPEAKERS):
                t_json = txt_dir / f"{wav.stem}_{spk}.json"
                if t_json.exists():
                    segs_per_spk.append(json.load(t_json.open())["segments"])
                    continue

                all_segments, txt_accum = [], []
                for s, e, chunk in chunk_audio(wav_tensor[ch]):
                    ret = transcribe(
                        asr_model,
                        audio_from_numpy(chunk.numpy(), SAMPLE_RATE),
                        offset_seconds=s,
                    )
                    txt_accum.append(ret.text)
                    all_segments.extend(
                        {
                            "start": round(seg.start_seconds, 3),
                            "end": round(seg.end_seconds, 3),
                            "text": seg.text,
                        }
                        for seg in ret.segments
                    )

                json.dump(
                    {"text": "".join(txt_accum), "segments": all_segments},
                    t_json.open("w", encoding="utf-8"),
                    ensure_ascii=False,
                    indent=2,
                )
                segs_per_spk.append(all_segments)

            # ── WhisperX alignment ─────────────────────────────────
            futures = [
                align_exec.submit(
                    whisperx.align,
                    segs_per_spk[idx],
                    align_model,
                    meta,
                    wav_tensor[idx].numpy(),
                    device,
                    False,
                )
                for idx in range(2)
            ]

            merged = []
            for idx, fut in enumerate(futures):
                aligned = fut.result()
                spk = SPEAKERS[idx]
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
            aln_path = ALN_ROOT / rel.parent / f"{wav.stem}.json"
            aln_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(
                merged,
                aln_path.open("w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )

        except Exception:
            elog(f"ERROR processing {wav}")
            traceback.print_exc(file=log)
            continue


# ─── Utility ────────────────────────────────────────────────────────


def is_done(wav: Path) -> bool:
    path = ALN_ROOT / f"{wav.stem}.json"
    return path.is_file() and path.stat().st_size > 0


# ─── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["."], help="sub‑dirs under audio/")
    args = ap.parse_args()

    targets: list[Path] = []
    for d in args.dirs:
        sub = IN_ROOT / d
        if not sub.is_dir():
            print(f"[warn] {sub} not found", file=sys.stderr)
            continue
        targets.extend(p for p in sub.rglob("*.wav") if not is_done(p))

    print(f"{len(targets)} wav files to process.")
    if not targets:
        sys.exit(0)

    gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if not gpus:
        print("No CUDA device detected.", file=sys.stderr)
        sys.exit(1)

    chunks = [targets[i :: len(gpus)] for i in range(len(gpus))]

    procs = [
        mp.Process(target=worker, args=(dev, chunk), daemon=False)
        for dev, chunk in zip(gpus, chunks)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    print("=== all CallHome files processed! ===")
