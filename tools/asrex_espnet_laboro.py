#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaboroTVSpeech tar.gz -> (wav + aligned.json) ペアを WebDataset 形式で自動シャーディング
"""
import argparse, io, json, math, tarfile, traceback, os
from pathlib import Path
import soundfile as sf
import torch, torchaudio
from tqdm import tqdm
import webdataset as wds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archive", required=True, help="input LaboroTVSpeech_v2.1b.tar.gz"
    )
    ap.add_argument("--out_dir", required=True, help="output dir for shard-XXXXX.tar")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples_per_shard", type=int, default=100_000)
    ap.add_argument("--skip_short_sec", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.device)
    import whisperx
    from reazonspeech.espnet.asr import load_model, transcribe, audio_from_numpy

    asr_model = load_model(device=args.device)
    align_model, align_meta = whisperx.load_align_model(
        language_code="ja", device=args.device
    )

    # 全 wav の TarInfo を列挙
    with tarfile.open(args.archive, mode="r:gz") as tf:
        all_wavs = [m for m in tf if m.isfile() and m.name.lower().endswith(".wav")]
    all_wavs.sort(key=lambda m: m.name)
    if args.limit > 0:
        all_wavs = all_wavs[: args.limit]

    total = len(all_wavs)
    num_shards = math.ceil(total / args.samples_per_shard)
    print(f"{total} wavs → {num_shards} shards (≈{args.samples_per_shard}/shard)")

    # シャードごとに作成
    for shard_idx in range(num_shards):
        start = shard_idx * args.samples_per_shard
        end = min(total, (shard_idx + 1) * args.samples_per_shard)
        my_wavs = all_wavs[start:end]
        out_tar = out_dir / f"shard-{shard_idx:05d}.tar"

        if args.resume and out_tar.exists() and out_tar.stat().st_size > 0:
            print(f"[skip] {out_tar}")
            continue

        print(f"[shard {shard_idx+1}/{num_shards}] {len(my_wavs)} samples")

        sink = wds.TarWriter(str(out_tar))
        with tarfile.open(args.archive, mode="r:gz") as tf_in:
            for m in tqdm(my_wavs, desc=f"shard-{shard_idx:05d}"):
                try:
                    # 読み込み
                    fobj = tf_in.extractfile(m)
                    if fobj is None:
                        continue
                    wav_bytes = fobj.read()

                    # wav -> mono16k
                    audio, sr = sf.read(
                        io.BytesIO(wav_bytes), dtype="float32", always_2d=True
                    )
                    wav = audio.mean(axis=1)
                    if sr != 16000:
                        wav = (
                            torchaudio.functional.resample(
                                torch.from_numpy(wav).unsqueeze(0), sr, 16000
                            )
                            .squeeze(0)
                            .numpy()
                        )
                        sr = 16000

                    dur = len(wav) / float(sr)
                    if dur < args.skip_short_sec:
                        json_bytes = json.dumps([], ensure_ascii=False).encode("utf-8")
                        key = Path(m.name).stem
                        sink.write(
                            {"__key__": key, "wav": wav_bytes, "json": json_bytes}
                        )
                        continue

                    # ASR
                    ret = transcribe(asr_model, audio_from_numpy(wav, sr))
                    segments = [
                        {"start": s.start_seconds, "end": s.end_seconds, "text": s.text}
                        for s in ret.segments
                        if s.end_seconds > s.start_seconds
                    ]

                    # WhisperX align
                    aligned = whisperx.align(
                        transcript=segments,
                        model=align_model,
                        align_model_metadata=align_meta,
                        audio=wav,
                        device=args.device,
                        return_char_alignments=False,
                    )

                    merged = []
                    for seg in aligned.get("segments", []):
                        for w in seg.get("words", []):
                            if {"word", "start", "end"} <= set(w):
                                merged.append(
                                    {
                                        "speaker": "A",
                                        "word": w["word"],
                                        "start": w["start"],
                                        "end": w["end"],
                                    }
                                )

                    merged.sort(key=lambda x: x["start"])
                    json_bytes = json.dumps(
                        merged, ensure_ascii=False, indent=2
                    ).encode("utf-8")

                    # WebDataset のサンプルとして書き込み
                    key = Path(m.name).stem
                    sink.write({"__key__": key, "wav": wav_bytes, "json": json_bytes})

                except Exception:
                    print(f"[error] {m.name}")
                    traceback.print_exc()
                    continue

        sink.close()
        print(f"[done] {out_tar}")


if __name__ == "__main__":
    main()
