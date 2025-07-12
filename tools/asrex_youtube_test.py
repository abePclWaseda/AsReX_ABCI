#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2-speaker separation ➜ ReazonSpeech-NeMo ASR ➜ WhisperX alignment
"""

import os, json, tempfile, traceback, multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch, soundfile as sf, librosa
from dotenv import load_dotenv

IN_ROOT = Path(
    "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/audio/youtube_test"
)
SEP_DIR = Path(
    "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/separated/youtube_test"
)
TXT_DIR = Path(
    "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/transcripts/youtube_test"
)
ALN_DIR = Path(
    "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/text/youtube_test"
)
for p in (SEP_DIR, TXT_DIR, ALN_DIR):
    p.mkdir(parents=True, exist_ok=True)

SPEAKER_LIST = ("A", "B")
COMPUTE_TYPE = "float16"
HF_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN")


def worker(device: str, wav_paths: list[Path]) -> None:
    """フルパイプラインを 1GPU で処理"""
    torch.cuda.set_device(device)

    # ① ConvTasNet
    from asteroid.models import ConvTasNet

    print(f"[GPU {device}] loading ConvTasNet …")
    sep_model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    sep_model.to(device).eval()

    # ② ReazonSpeech-NeMo
    from reazonspeech.nemo.asr import load_model, transcribe, audio_from_numpy

    print(f"[GPU {device}] loading ReazonSpeech-NeMo …")
    asr_model = load_model(device=device)

    def nemo_asr_numpy(audio_np: np.ndarray, sr: int = 16_000) -> dict:
        ret = transcribe(asr_model, audio_from_numpy(audio_np, sr))
        segs = [
            {
                "start": round(s.start_seconds, 3),
                "end": round(s.end_seconds, 3),
                "text": s.text,
            }
            for s in ret.segments
        ]
        return {"text": ret.text, "segments": segs}

    # ③ WhisperX
    import whisperx

    print(f"[GPU {device}] loading WhisperX align model …")
    align_model, meta = whisperx.load_align_model("ja", device)

    log_path = f"align_errors_{device}_podcast_train.log"
    with open(log_path, "a") as LOG_FILE:

        def log(msg: str):
            print(msg)
            LOG_FILE.write(msg + "\n")
            LOG_FILE.flush()

        for wav in tqdm(wav_paths, desc=f"[GPU {device}] processing"):
            try:
                # ---------- ① separation ----------
                y_mono, _ = librosa.load(wav, sr=16_000)
                est = sep_model.separate(torch.tensor(y_mono).unsqueeze(0))
                stereo = est[0].cpu().numpy().T  # (T, 2)
                sep_path = SEP_DIR / f"{wav.stem}.wav"
                sf.write(sep_path, stereo, 16_000)

                # ---------- ② ASR ----------
                for ch, spk in enumerate(SPEAKER_LIST):
                    txt_path = TXT_DIR / f"{wav.stem}_{spk}.json"
                    if not txt_path.exists():
                        res = nemo_asr_numpy(stereo[:, ch])
                        json.dump(res, txt_path.open("w"), ensure_ascii=False, indent=2)

                # ---------- ③ align ----------
                y, sr = sf.read(sep_path, dtype="float32")
                merged = []
                for ch, spk in enumerate(SPEAKER_LIST):
                    txt_path = TXT_DIR / f"{wav.stem}_{spk}.json"
                    segs = json.load(txt_path.open())["segments"]
                    aligned = whisperx.align(
                        segs,
                        align_model,
                        meta,
                        y[:, ch],
                        device,
                        return_char_alignments=False,
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
                with out_path.open("w", encoding="utf-8") as f:
                    f.write("[\n")
                    for i, obj in enumerate(merged):
                        f.write('   ') 
                        json.dump(obj, f, ensure_ascii=False, separators=(",", ": "))
                        if i != len(merged) - 1:
                            f.write(",\n")
                    f.write("\n]\n")

            except Exception:
                log(f"\n!! ERROR while processing {wav.name} !!")
                traceback.print_exc(file=LOG_FILE)
                continue


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    GPU_LIST = ["cuda:0"]
    wav_all = sorted(IN_ROOT.rglob("*.wav"))

    chunks = [wav_all[i :: len(GPU_LIST)] for i in range(len(GPU_LIST))]

    procs = []
    for dev, paths in zip(GPU_LIST, chunks):
        p = mp.Process(target=worker, args=(dev, paths), daemon=False)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("=== all done! ===")
