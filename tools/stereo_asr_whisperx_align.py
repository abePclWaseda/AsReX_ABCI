#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stereo → ReazonSpeech‑ESPnet ASR → WhisperX alignment 〈**whole‑file mode**〉
==============================================================================
🆕 各チャネルを **丸ごと一括推論** することで文脈切れを防止します。
* ch‑0 = Speaker A, ch‑1 = Speaker B
* 8 kHz μ‑law → 16 kHz PCM へ自動リサンプル
* WhisperX で word‑level 整列し、A+B を時系列マージ
* 想定 vRAM: ≈2.0 GB / 15 min / ch (fp16) — GPU に余裕がある場合のみ利用

ディレクトリ構造 / 使い方は変更なし。
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

# ─── Paths ──────────────────────────────────────────────────────────
ROOT = Path("/home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/Tabidachi")
IN_ROOT = ROOT / "audio"
TXT_ROOT = ROOT / "transcripts"
ALN_ROOT = ROOT / "text"
for p in (TXT_ROOT, ALN_ROOT):
    p.mkdir(parents=True, exist_ok=True)

SPEAKERS = ("A", "B")
SAMPLE_RATE = 16_000

# ─── Helper ─────────────────────────────────────────────────────────


def resample_2ch(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Return (2, T) @16 kHz CPU tensor, duplicating mono if needed."""
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0).repeat(2, 1)
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    return wav.cpu()


# ─── Worker ─────────────────────────────────────────────────────────


def worker(device: str, wav_paths: List[Path], align_threads: int = 2):
    torch.cuda.set_device(device)
    os.environ["OMP_NUM_THREADS"] = "1"

    from reazonspeech.espnet.asr import load_model, transcribe, audio_from_numpy
    import whisperx

    asr_model = load_model(device=device)
    align_model, meta = whisperx.load_align_model("ja", device)
    align_exec = ThreadPoolExecutor(max_workers=align_threads)

    tag = os.getenv("PBS_ARRAY_INDEX") or os.getenv("SLURM_ARRAY_TASK_ID") or "solo"
    logfile = Path(f"Tabidachi_align_errors.log").open("a", encoding="utf-8")

    def elog(msg: str):
        print(msg, flush=True)
        logfile.write(msg + "\n")
        logfile.flush()

    for wav in tqdm(wav_paths, desc=f"[GPU {device}]"):
        try:
            wav_tensor, sr = torchaudio.load(wav)
            wav_tensor = resample_2ch(wav_tensor, sr)
            rel = wav.relative_to(IN_ROOT)
            txt_dir = TXT_ROOT / rel.parent
            txt_dir.mkdir(parents=True, exist_ok=True)

            segs_per_spk: list[list[dict]] = []

            # ── ASR: whole file per channel ─────────────────────────
            for ch, spk in enumerate(SPEAKERS):
                t_json = txt_dir / f"{wav.stem}_{spk}.json"
                if t_json.exists():
                    segs_per_spk.append(json.load(t_json.open())["segments"])
                    continue

                ret = transcribe(
                    asr_model, audio_from_numpy(wav_tensor[ch].numpy(), SAMPLE_RATE)
                )
                segments = [
                    {
                        "start": seg.start_seconds,
                        "end": seg.end_seconds,
                        "text": seg.text,
                    }
                    for seg in ret.segments
                ]

                json.dump(
                    {"text": ret.text, "segments": segments},
                    t_json.open("w", encoding="utf-8"),
                    ensure_ascii=False,
                    indent=2,
                )
                segs_per_spk.append(segments)

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
                    # interpolate_method="linear",  # 本当はinterpolate_method のエラーを修正するために，これをつけたいが，そうすると別のエラーが発生し，処理ミスをするファイルが増えるので，コメントアウトしておく．
                    # return_char_alignments=False,
                    # print_progress=False,
                    # combined_progress=False,
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
                        "start": w["start"],
                        "end": w["end"],
                    }
                    for seg in aligned["segments"]
                    for w in seg["words"]
                )

            merged.sort(key=lambda x: x["start"])
            out = ALN_ROOT / rel.parent / f"{wav.stem}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump(
                merged, out.open("w", encoding="utf-8"), ensure_ascii=False, indent=2
            )

        except Exception:
            elog(f"ERROR processing {wav}")
            traceback.print_exc(file=logfile)
            continue


# ─── Utility ────────────────────────────────────────────────────────


def is_done(wav: Path) -> bool:
    p = ALN_ROOT / f"{wav.stem}.json"
    return p.is_file() and p.stat().st_size > 0


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
    chunks = [targets[i :: len(gpus)] for i in range(len(gpus))] if gpus else [targets]

    procs = [
        mp.Process(target=worker, args=(dev, chunk), daemon=False)
        for dev, chunk in zip(gpus, chunks)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    print("=== Tabidachi processing complete ===")
