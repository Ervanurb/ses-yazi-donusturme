"""Ses dosyasını yazıya döker: ön işleme -> VAD -> Whisper -> txt/srt/json.

Kullanım:
    python transcribe.py kayit.mp3
    python transcribe.py kayit.mp3 --model small --dil tr
    python transcribe.py kayit.mp3 --no-denoise --no-vad --grafik
"""

import argparse
import json
from pathlib import Path

from ses_yazi.audio import WHISPER_SR, detect_speech, preprocess
from ses_yazi.transcribe import load_model, to_dict, to_srt, to_txt, transcribe


def main():
    parser = argparse.ArgumentParser(description="Sesi yazıya dönüştür")
    parser.add_argument("dosyalar", type=Path, nargs="+")
    parser.add_argument("--model", help="tiny/base/small/medium/large-v3/turbo (varsayılan: GPU'da turbo, CPU'da small)")
    parser.add_argument("--dil", help="Dil kodu, ör. tr, en (boş bırakılırsa otomatik algılanır)")
    parser.add_argument("--cihaz", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-denoise", action="store_true", help="Gürültü azaltmayı kapat")
    parser.add_argument("--no-vad", action="store_true", help="Konuşma tespitini kapat, tüm sesi işle")
    parser.add_argument("--grafik", action="store_true", help="Analiz grafiğini de kaydet")
    parser.add_argument("--cikti", type=Path, default=Path("output"))
    args = parser.parse_args()

    print("Model yükleniyor...")
    model, device = load_model(args.model, args.cihaz)
    print(f"Cihaz: {device}\n")

    args.cikti.mkdir(exist_ok=True)
    for path in args.dosyalar:
        original, audio = preprocess(path, denoise=not args.no_denoise)
        duration = len(audio) / WHISPER_SR
        # 1 s'den kısa duraklamaları birleştir: Whisper cümle bağlamını görürse
        # çok daha doğru yazar; yalnızca uzun sessizlikler atlanır.
        speech = None if args.no_vad else detect_speech(audio, WHISPER_SR, min_silence=1.0)

        result = transcribe(model, audio, speech, language=args.dil)

        stem = args.cikti / path.stem
        stem.with_suffix(".txt").write_text(to_txt(result), encoding="utf-8")
        stem.with_suffix(".srt").write_text(to_srt(result), encoding="utf-8")
        stem.with_suffix(".json").write_text(
            json.dumps(to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.grafik:
            from ses_yazi.viz import plot_analysis

            plot_analysis(original, audio, speech or [], f"{stem}_analiz.png", title=path.name)

        print(f"=== {path.name} ===")
        print(f"Dil: {result.language} ({result.language_prob:.0%}), süre {duration:.1f} s, "
              f"işlem {result.seconds:.1f} s ({duration / max(result.seconds, 1e-3):.1f}x gerçek zaman)")
        print(to_txt(result, timestamps=True))
        print(f"Kaydedildi: {stem}.txt / .srt / .json\n")


if __name__ == "__main__":
    main()
