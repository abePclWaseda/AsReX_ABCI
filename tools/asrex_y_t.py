#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2-speaker separation ➜ ReazonSpeech-NeMo ASR ➜ WhisperX alignment (v2, soundfile-only)
"""

import os, json, traceback, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List
import argparse, sys

import torch, torchaudio
from tqdm import tqdm

# ─── 固定パス ───────────────────────────────────────────────
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


# ─── Worker ────────────────────────────────────────────────
def worker(
    device: str, wav_paths: List[Path], save_sep: bool = True, align_threads: int = 1
):
    """1 GPU ぶんのフルパイプラインを処理"""

    torch.cuda.set_device(device)
    os.environ["OMP_NUM_THREADS"] = "1"  # BLAS の暴走防止

    from asteroid.models import ConvTasNet
    from reazonspeech.nemo.asr import load_model, transcribe, audio_from_numpy
    import whisperx

    sep_model = (
        ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
        .to(device)
        .eval()
    )

    asr_model = load_model(device=device)
    align_model, meta = whisperx.load_align_model("ja", device)

    align_exec = ThreadPoolExecutor(max_workers=align_threads)

    job_tag = os.getenv("PBS_ARRAY_INDEX") or os.getenv("SLURM_ARRAY_TASK_ID") or "solo"
    log_path = Path(f"align_errors_{device[-1]}_podcast_train_{job_tag}.log")
    with log_path.open("a") as LOG:

        def elog(msg: str):
            print(msg)
            LOG.write(msg + "\n")
            LOG.flush()

        for wav in tqdm(wav_paths, desc=f"[GPU {device}]"):
            try:
                # ① Separation ----------------------------------------------
                wav_tensor, sr = torchaudio.load(wav)
                if sr != 16_000:
                    wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16_000)
                mono = wav_tensor.mean(0, keepdim=True).to(device)
                stereo = sep_model.separate(mono)[0].cpu().float()

                # --- 出力用ディレクトリを入力に合わせて再現 -------------
                rel = wav.relative_to(IN_ROOT)  # series1/ep3/foobar.wav
                out_dir = SEP_DIR / rel.parent  # …/separated/…/series1/ep3
                out_dir.mkdir(parents=True, exist_ok=True)
                if save_sep:  # separated wav
                    torchaudio.save(out_dir / rel.name, stereo, 16_000)

                # ② ASR -----------------------------------------------------
                txt_paths = []
                for ch, spk in enumerate(SPEAKER_LIST):
                    txt_dir = TXT_DIR / rel.parent
                    txt_dir.mkdir(parents=True, exist_ok=True)
                    t_path = txt_dir / f"{wav.stem}_{spk}.json"
                    if not t_path.exists():
                        ret = transcribe(
                            asr_model, audio_from_numpy(stereo[ch].numpy(), 16_000)
                        )
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
                            t_path.open("w"),
                            ensure_ascii=False,
                            indent=2,
                        )
                    txt_paths.append(t_path)

                # ③ Align (async) ------------------------------------------
                def _align_job(spk, segs, audio_np):
                    aligned = whisperx.align(
                        segs,
                        align_model,
                        meta,
                        audio_np,
                        device,
                        return_char_alignments=False,
                    )
                    return spk, aligned

                futures = [
                    align_exec.submit(
                        _align_job,
                        spk,
                        json.load(txt_paths[ch].open())["segments"],
                        stereo[ch].numpy(),
                    )
                    for ch, spk in enumerate(SPEAKER_LIST)
                ]

                merged = []
                for fut in futures:
                    spk, aligned = fut.result()
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
                aln_dir = ALN_DIR / rel.parent
                aln_dir.mkdir(parents=True, exist_ok=True)
                json.dump(
                    merged,
                    (aln_dir / f"{wav.stem}.json").open("w"),
                    ensure_ascii=False,
                    indent=2,
                )

            except Exception:
                elog(f"ERROR processing {wav.name}")
                traceback.print_exc(file=LOG)
                continue


# ─── Utility ────────────────────────────────────────────────
def is_done(wav: Path) -> bool:
    rel = wav.relative_to(IN_ROOT)
    out = ALN_DIR / rel.parent / f"{wav.stem}.json"
    return out.is_file() and out.stat().st_size > 0


# ─── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dirs", nargs="+", required=True, help="sub-dirs under IN_ROOT to process"
    )
    args = ap.parse_args()

    targets: list[Path] = []
    for d in args.dirs:
        sub = IN_ROOT / d
        if not sub.is_dir():
            print(f"[warn] {sub} not found", file=sys.stderr)
            continue
        targets.extend(p for p in sub.rglob("*.wav") if not is_done(p))
    print(f"{len(targets)} wav files to process.")

    GPU_LIST = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    chunks = [targets[i :: len(GPU_LIST)] for i in range(len(GPU_LIST))]

    procs = [
        mp.Process(target=worker, args=(dev, chunk), daemon=False)
        for dev, chunk in zip(GPU_LIST, chunks)
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    print("=== all done! ===")
