#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import torch
import torchaudio
from pathlib import Path
from reazonspeech.espnet.asr import load_model, transcribe, audio_from_numpy
import whisperx


# ─── Constants ─────────────────────────────
SPEAKERS = ("A", "B")
SAMPLE_RATE = 16_000


# 🔁 ヘルパー: モノラル→2ch変換＋リサンプル
def resample_2ch(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0).repeat(2, 1)
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    return wav.cpu()


def process_single_file(wav_path: Path, output_dir: Path, device: str = "cuda:0"):
    print(f"Processing: {wav_path.name} on {device}")

    wav_tensor, sr = torchaudio.load(wav_path)
    wav_tensor = resample_2ch(wav_tensor, sr)

    # 出力パスの設定
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = output_dir / "transcripts"
    aln_dir = output_dir / "text"
    txt_dir.mkdir(parents=True, exist_ok=True)
    aln_dir.mkdir(parents=True, exist_ok=True)

    # モデルロード
    asr_model = load_model(device=device)
    align_model, meta = whisperx.load_align_model("ja", device)

    segs_per_spk = []

    for ch, spk in enumerate(SPEAKERS):
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
            (txt_dir / f"{wav_path.stem}_{spk}.json").open("w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        segs_per_spk.append(segments)

    # WhisperXで整列
    merged = []
    for idx in range(2):
        aligned = whisperx.align(
            segs_per_spk[idx],
            align_model,
            meta,
            wav_tensor[idx].numpy(),
            device,
            return_char_alignments=False,
        )

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

    # 保存
    json.dump(
        merged,
        (aln_dir / f"{wav_path.stem}.json").open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    print("✅ Done:", wav_path.name)


if __name__ == "__main__":
    # ✏️ここを書き換えてください
    wav_path = Path(
        "/home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/Tabidachi/audio/320_6_1.wav"
    )  # 対象のWAVファイル
    output_dir = Path("data/Tabidachi")  # 出力先ディレクトリ

    process_single_file(wav_path, output_dir)
