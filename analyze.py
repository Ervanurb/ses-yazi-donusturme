"""Bir ses dosyasını analiz eder: ön işler, konuşmaları bulur, grafik çizer.

Kullanım:
    python analyze.py kayit.mp3
    python analyze.py kayit.mp3 --no-denoise
"""

import argparse
from pathlib import Path

from ses_yazi.audio import WHISPER_SR, detect_speech, load_audio, preprocess, save_wav
from ses_yazi.viz import plot_analysis


def main():
    parser = argparse.ArgumentParser(description="Ses dosyası analizi ve ön işleme")
    parser.add_argument("dosya", type=Path)
    parser.add_argument("--no-denoise", action="store_true", help="Gürültü azaltmayı kapat")
    parser.add_argument("--cikti", type=Path, default=Path("output"), help="Çıktı klasörü")
    args = parser.parse_args()

    raw, raw_sr = load_audio(args.dosya)
    print(f"Dosya       : {args.dosya.name}")
    print(f"Orijinal    : {raw_sr} Hz, {len(raw) / raw_sr:.2f} s")

    original, processed = preprocess(args.dosya, denoise=not args.no_denoise)
    segments = detect_speech(processed, WHISPER_SR)
    speech_total = sum(s.duration for s in segments)
    print(f"İşlenmiş    : {WHISPER_SR} Hz mono, {len(processed) / WHISPER_SR:.2f} s")
    print(f"Konuşma     : {len(segments)} bölüm, toplam {speech_total:.2f} s")
    for i, s in enumerate(segments, 1):
        print(f"  {i:2d}. {s.start:6.2f}s - {s.end:6.2f}s  ({s.duration:.2f} s)")

    args.cikti.mkdir(exist_ok=True)
    stem = args.dosya.stem
    wav_path = args.cikti / f"{stem}_temiz.wav"
    png_path = args.cikti / f"{stem}_analiz.png"
    save_wav(wav_path, processed)
    plot_analysis(original, processed, segments, png_path, title=args.dosya.name)
    print(f"Kaydedildi  : {wav_path}, {png_path}")


if __name__ == "__main__":
    main()
